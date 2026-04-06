# panorama2gaussian/panorama2gaussian_converter.py
"""
Panorama2Gaussian 高斯转换器：深度图 + ERP 图像 -> 高斯参数。

将等距柱状深度图直接转为 3D 高斯泼溅参数，使用球面投影。
每个像素（步长分辨率下）对应一个高斯，位置在 depth * 射线方向。

颜色直接来自全景像素（sRGB），PLY 导出路径无需额外 gamma 校正。
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from .spherical_grid import create_spherical_grid, rotation_matrix_to_quaternion
from .scene_filter import filter_gaussian_candidates, SkyMode


@dataclass
class Panorama2GaussianParams:
    """Panorama2Gaussian 深度转高斯的参数。"""
    stride: int = 2
    depth_min: float = 0.1
    depth_max: float = 100.0
    default_opacity: float = 0.9
    disc_thickness: float = 0.1       # 相对 xy 尺度的 z 向尺度
    sky_detection: str = "depth"      # "depth"、"gradient"、"none"
    sky_threshold: float = 80.0       # 超过此深度视为天空
    pole_thinning: bool = True
    min_density_ratio: float = 0.30


def depth_to_gaussians(
    erp_image: torch.Tensor,
    depth_map: torch.Tensor,
    params: Optional[Panorama2GaussianParams] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    将 ERP 深度图与图像转为高斯泼溅参数。

    Args:
        erp_image: [H, W, 3] uint8 或 [0,1] float RGB 全景
        depth_map: [H, W] float 深度，单位米（径向距离）
        params: Panorama2Gaussian 转换参数
        device: 输出张量所在设备

    Returns:
        字典键：means [N,3]、scales [N,3]、quats [N,4]（XYZW）、
        colors [N,3]（sRGB 0–1）、opacities [N,1]
    """
    if params is None:
        params = Panorama2GaussianParams()

    if device is None:
        device = depth_map.device if isinstance(depth_map, torch.Tensor) else torch.device('cpu')

    H, W = depth_map.shape[:2]
    stride = params.stride

    # 输入转为 numpy 以便滤波
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_map, dtype=np.float32)

    if isinstance(erp_image, torch.Tensor):
        img_np = erp_image.detach().cpu().numpy()
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img_np.astype(np.uint8)
    else:
        img_uint8 = np.asarray(erp_image, dtype=np.uint8)

    # ── 1. 滤波：天空检测 + 深度范围 + 极点稀疏化 ──
    keep_mask, sky_mask = filter_gaussian_candidates(
        depth_map=depth_np,
        erp_image=img_uint8,
        stride=stride,
        sky_mode=SkyMode.SKIP,
        pole_thinning=params.pole_thinning,
        depth_min=params.depth_min,
        depth_max=params.depth_max if params.sky_threshold <= 0 else params.sky_threshold,
        sky_detection=params.sky_detection,
        min_density_ratio=params.min_density_ratio,
    )

    # ── 2. 构建球面网格 ──
    grid = create_spherical_grid(H, W, device=device, stride=stride)

    # 网格尺寸
    H_s = grid.rhat.shape[0]
    W_s = grid.rhat.shape[1]

    # keep_mask 转为 torch 并展平
    keep_mask_t = torch.from_numpy(keep_mask).to(device)
    # 保证形状一致（滤波返回维度可能略有差异）
    keep_mask_t = keep_mask_t[:H_s, :W_s]
    valid_idx = keep_mask_t.nonzero(as_tuple=False)  # [N, 2] (row, col)

    if valid_idx.shape[0] == 0:
        return _empty_gaussians(device)

    rows = valid_idx[:, 0]
    cols = valid_idx[:, 1]
    N = rows.shape[0]

    # ── 3. 在有效网格位置采样深度 ──
    # 将步长网格索引映射回原始像素坐标
    pixel_rows = rows * stride + stride // 2
    pixel_cols = cols * stride + stride // 2
    pixel_rows = pixel_rows.clamp(0, H - 1)
    pixel_cols = pixel_cols.clamp(0, W - 1)

    depth_sampled = depth_map[pixel_rows.cpu(), pixel_cols.cpu()].to(device)  # [N]

    # ── 4. 位置：p = depth * r̂ ──
    rhat = grid.rhat[rows, cols]  # [N, 3]
    positions = depth_sampled.unsqueeze(-1) * rhat  # [N, 3]

    # ── 5. 颜色：直接来自全景（已是 sRGB）──
    if isinstance(erp_image, torch.Tensor):
        img_for_sample = erp_image.float()
        if erp_image.dtype == torch.uint8:
            img_for_sample = img_for_sample / 255.0
    else:
        img_for_sample = torch.from_numpy(img_uint8).float() / 255.0

    colors = img_for_sample[pixel_rows.cpu(), pixel_cols.cpu()].to(device)  # [N, 3]

    # ── 6. 尺度：随纬度各向异性 ──
    # 网格采样之间的角间距
    angular_spacing = math.pi / H  # 每行像素弧度
    base_scale = depth_sampled * angular_spacing * stride  # [N]

    # 纬度修正：sin(phi) 使极点附近尺度变小
    phi = grid.phi[rows, cols]  # [N]
    sin_phi = torch.sin(phi).clamp(min=0.05)
    scale_xy = base_scale * sin_phi

    # 盘状：法向方向较薄
    scale_z = base_scale * params.disc_thickness

    scales = torch.stack([scale_xy, scale_xy, scale_z], dim=-1)  # [N, 3]

    # ── 7. 旋转：用切向基与法线对齐 ──
    # 由 [tangent_right, tangent_up, -rhat] 构建旋转矩阵
    # 构成正交标架，其中 -rhat 指向相机
    tangent_right = grid.tangent_right[rows, cols]  # [N, 3]
    tangent_up = grid.tangent_up[rows, cols]        # [N, 3]
    normal = -rhat                                   # [N, 3]

    # R = [right | up | normal] 为列向量
    R = torch.stack([tangent_right, tangent_up, normal], dim=-1)  # [N, 3, 3]
    quats = rotation_matrix_to_quaternion(R)  # [N, 4] XYZW

    # ── 8. 不透明度：默认均匀 ──
    opacities = torch.full((N, 1), params.default_opacity, device=device)

    return {
        'means': positions,
        'scales': scales,
        'quats': quats,
        'colors': colors,
        'opacities': opacities,
    }


def _empty_gaussians(device: torch.device) -> dict:
    """返回形状正确的空高斯字典。"""
    return {
        'means': torch.zeros(0, 3, device=device),
        'scales': torch.zeros(0, 3, device=device),
        'quats': torch.zeros(0, 4, device=device),
        'colors': torch.zeros(0, 3, device=device),
        'opacities': torch.zeros(0, 1, device=device),
    }
