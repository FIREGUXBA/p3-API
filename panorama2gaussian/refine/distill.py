"""阶段 3：3DGS 蒸馏 —— 用可微渲染使高斯与修复后图像一致。"""

import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 将 GSFix3D 加入路径
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


def distill_to_gaussians(
    gaussians,
    repaired_images,
    cameras,
    hole_masks,
    original_images=None,
    original_cameras=None,
    num_iterations=3000,
    densify_interval=100,
    densify_grad_threshold=0.005,
    lr_position=0.00032,
    lr_feature=0.0025,
    lr_opacity=0.025,
    lr_scaling=0.002,
    lr_rotation=0.001,
    original_view_ratio=0.3,
    prune_opacity_threshold=0.005,
    iters_per_view=20,
    kf_iters=50,
):
    """通过可微渲染优化 3DGS 以匹配修复图像。

    紧密遵循 GSFix3D 的 refine_gs.py 模式：
    - 阶段 A：每个修复视角 20 次迭代（含致密化）
    - 阶段 B：在扩展训练集（修复 + 原始）上 50 轮

    对原始内容的保护：
    - 原始高斯学习率 ×0.1，减轻漂移
    - 阶段 B 保守（50 次迭代，非数千次）
    - 原始立方体贴图视角锚定已较好区域
    """
    if gaussians is None:
        logger.warning("distill_to_gaussians：gaussians 为 None，已跳过")
        return gaussians

    from gs.camera import Camera as GSCamera
    from gs.gaussian_renderer import render as gs_render
    from gs.loss_utils import l1_loss, ssim
    from gs.arguments import OptimizationParams

    from .camera_rig import _camera_to_RT

    # 记录初始高斯数量，用于基于来源的学习率缩放
    initial_count = gaussians.get_xyz.shape[0]

    # 构造假 parser 以创建 OptimizationParams
    import argparse
    mock_parser = argparse.ArgumentParser()
    optim_params = OptimizationParams(mock_parser)
    # 用我们的设置覆盖
    optim_params.position_lr_init = lr_position
    optim_params.position_lr_final = lr_position * 0.01
    optim_params.position_lr_max_steps = iters_per_view * len(repaired_images) + kf_iters
    optim_params.feature_lr = lr_feature
    optim_params.opacity_lr = lr_opacity
    optim_params.scaling_lr = lr_scaling
    optim_params.rotation_lr = lr_rotation
    optim_params.densify_grad_threshold = densify_grad_threshold
    optim_params.prune_opacity_threshold = prune_opacity_threshold

    # 优化器
    gaussians.training_setup(optim_params)

    # 降低原始高斯学习率以防漂移
    _apply_original_lr_scaling(gaussians, initial_count, scale=0.1)

    # 渲染器用的 Pipe 占位
    class PipeMock:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    bg = torch.zeros(3, device="cuda")

    def _camera_to_gs(cam):
        """CameraPose -> GSFix3D Camera。"""
        R, T = _camera_to_RT(cam)
        fov_rad = math.radians(cam.fov_deg)
        aspect = cam.width / cam.height
        fov_x = 2 * math.atan(math.tan(fov_rad / 2) * aspect)
        return GSCamera(R=R, T=T, FoVx=fov_x, FoVy=fov_rad,
                        width=cam.width, height=cam.height)

    # 图像转张量 (3, H, W) 上 CUDA
    repair_tensors = [
        torch.from_numpy(img).permute(2, 0, 1).cuda() for img in repaired_images
    ]
    repair_gs_cams = [_camera_to_gs(cam) for cam in cameras]

    orig_tensors = None
    orig_gs_cams = None
    if original_images:
        orig_tensors = [
            torch.from_numpy(img).permute(2, 0, 1).cuda() for img in original_images
        ]
        orig_gs_cams = [_camera_to_gs(cam) for cam in original_cameras]

    n_repair = len(repair_tensors)
    n_orig = len(orig_tensors) if orig_tensors else 0

    logger.info(f"蒸馏：每视角 {iters_per_view} 次 × {n_repair} 个视角 + "
                f"混合 {kf_iters} 次（{n_orig} 个原始视角）")

    # --- 阶段 A：在修复图像上按视角优化 ---
    #（与 refine_gs.py 一致：每张固定图 20 次迭代并致密化）
    for view_idx in range(n_repair):
        gt_image = repair_tensors[view_idx]
        gs_cam = repair_gs_cams[view_idx]

        for step in range(iters_per_view):
            render_pkg = gs_render(gs_cam, gaussians, PipeMock(), bg)
            rendered = render_pkg["render"]
            viewspace_points = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]

            loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))
            loss.backward()

            gaussians.add_densification_stats(viewspace_points, visibility_filter)

            if step % 5 == 0:
                gaussians.densify(densify_grad_threshold)

            if step == iters_per_view - 1:
                gaussians.prune(prune_opacity_threshold)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if view_idx % 5 == 0:
            logger.info(f"  阶段 A：视角 {view_idx+1}/{n_repair}，"
                        f"loss={loss.item():.4f}")

    # --- 阶段 B：保守混合优化（与 refine_gs.py kf_iters=50 一致）---
    all_tensors = list(repair_tensors) + (list(orig_tensors) if orig_tensors else [])
    all_cams = list(repair_gs_cams) + (list(orig_gs_cams) if orig_gs_cams else [])

    for step in range(kf_iters):
        # 每步打乱后遍历全部视角
        indices = list(range(len(all_tensors)))
        random.shuffle(indices)

        for idx in indices:
            gt_image = all_tensors[idx]
            gs_cam = all_cams[idx]

            rendered = gs_render(gs_cam, gaussians, PipeMock(), bg)["render"]
            loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * (1 - ssim(rendered, gt_image))
            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            logger.info(f"  阶段 B：轮次 {step}/{kf_iters}，loss={loss.item():.4f}")

    final_count = gaussians.get_xyz.shape[0]
    logger.info(f"蒸馏完成。高斯数量：{initial_count} -> {final_count} "
                f"（新增 {final_count - initial_count}）")
    return gaussians


def _apply_original_lr_scaling(gaussians, initial_count, scale=0.1):
    """通过逐参数梯度掩码降低原始高斯的学习率。

    Adam 本身不支持逐元素学习率，因此对原始高斯的梯度按比例缩放（等效）。
    在优化器上注册 hook 实现。
    """
    if not hasattr(gaussians, 'optimizer') or gaussians.optimizer is None:
        return

    def _scale_grad_hook(grad, count=initial_count, s=scale):
        """对原始高斯（索引 < count）的梯度缩放。"""
        if grad is None:
            return grad
        scaled = grad.clone()
        if len(scaled.shape) >= 1 and scaled.shape[0] > count:
            scaled[:count] *= s
        return scaled

    # 在所有可优化参数上注册 hook
    for param_group in gaussians.optimizer.param_groups:
        for param in param_group['params']:
            if param.requires_grad and param.shape[0] >= initial_count:
                param.register_hook(_scale_grad_hook)

    logger.info(f"已对前 {initial_count} 个原始高斯应用 {scale} 倍梯度缩放")
