"""SPAG-4D 与 GSFix3D 之间的 PLY 格式转换。"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


def load_gaussians_from_ply(ply_path: str, device: str = "cuda"):
    """将 SPAG-4D 的 PLY 加载为 GSFix3D 的 GaussianModel。

    处理格式差异：补全缺失法线、为 SH 阶 0 创建空 f_rest、直接填充张量。
    """
    from plyfile import PlyData
    from gs.gaussian_model import GaussianModel

    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    # 读取位置
    xyz = np.stack([
        np.asarray(vertex["x"]),
        np.asarray(vertex["y"]),
        np.asarray(vertex["z"]),
    ], axis=1).astype(np.float32)

    # 读取 SH DC 系数
    f_dc = np.stack([
        np.asarray(vertex["f_dc_0"]),
        np.asarray(vertex["f_dc_1"]),
        np.asarray(vertex["f_dc_2"]),
    ], axis=1).astype(np.float32)

    # 读取不透明度、尺度、旋转
    opacity = np.asarray(vertex["opacity"]).astype(np.float32)[:, np.newaxis]

    scale = np.stack([
        np.asarray(vertex["scale_0"]),
        np.asarray(vertex["scale_1"]),
        np.asarray(vertex["scale_2"]),
    ], axis=1).astype(np.float32)

    rot = np.stack([
        np.asarray(vertex["rot_0"]),
        np.asarray(vertex["rot_1"]),
        np.asarray(vertex["rot_2"]),
        np.asarray(vertex["rot_3"]),
    ], axis=1).astype(np.float32)

    n = xyz.shape[0]

    # 创建 GaussianModel 并直接填充
    gaussians = GaussianModel(sh_degree=0)
    gaussians.active_sh_degree = 0

    # 原始参数（CUDA 上的 nn.Parameter）
    gaussians._xyz = nn.Parameter(
        torch.from_numpy(xyz).cuda().requires_grad_(True)
    )
    gaussians._features_dc = nn.Parameter(
        torch.from_numpy(f_dc.reshape(n, 1, 3)).cuda().requires_grad_(True)
    )
    gaussians._features_rest = nn.Parameter(
        torch.zeros(n, 0, 3).cuda().requires_grad_(True)
    )
    gaussians._opacity = nn.Parameter(
        torch.from_numpy(opacity).cuda().requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(
        torch.from_numpy(scale).cuda().requires_grad_(True)
    )
    gaussians._rotation = nn.Parameter(
        torch.from_numpy(rot).cuda().requires_grad_(True)
    )

    # 辅助张量初始化
    gaussians.max_radii2D = torch.zeros(n).cuda()
    gaussians.xyz_gradient_accum = torch.zeros(n, 1).cuda()
    gaussians.denom = torch.zeros(n, 1).cuda()

    logger.info(f"已从 {ply_path} 加载 {n} 个高斯")
    return gaussians


def save_gaussians_to_ply(gaussians, output_path: str):
    """将 GaussianModel 保存为标准 3DGS PLY。"""
    if gaussians is None:
        logger.warning("save_gaussians_to_ply：gaussians 为 None，已跳过")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(output_path)
    logger.info(f"已保存高斯至 {output_path}")
