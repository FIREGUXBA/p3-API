# panorama2gaussian/ply_writer.py
"""
3D 高斯泼溅的 PLY 导出。

输出格式与 gsplat、SuperSplat 等 3DGS 查看器兼容。
"""

import numpy as np
from plyfile import PlyData, PlyElement
from typing import Optional

# 球谐 DC 项归一化常数
SH_C0 = 0.28209479177387814


def _linearRGB_to_sRGB(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF，与 Apple Metal 规范 7.7.7 一致。"""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.clip(linear, 0.0031308, None) ** (1.0 / 2.4) - 0.055
    )


def save_ply_gsplat(
    gaussians: dict,
    path: str,
    sh_degree: int = 0,
    colors_linear: bool = True,
) -> None:
    """
    将高斯保存为与 gsplat 查看器兼容的 PLY 格式。

    Args:
        gaussians: 含 means、scales、quats、colors、opacities 的字典，
                   可选 'sh1' [N,9] 为 SH 一阶系数
        path: 输出 PLY 路径
        sh_degree: SH 阶数（0=仅 DC，1=含一阶带…）
                   若 gaussians 含 'sh1' 会自动提升到 1
        colors_linear: True（SHARP 路径）时应用 linearRGB→sRGB；
                       False（Panorama2Gaussian 路径）时颜色已是 sRGB，跳过 gamma
    """
    # 辅助：确保为 numpy
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return np.array(x)

    # 转到 CPU numpy
    means = to_numpy(gaussians['means'])
    scales = to_numpy(gaussians['scales'])
    quats = to_numpy(gaussians['quats'])      # 内部为 XYZW 顺序
    colors = to_numpy(gaussians['colors'])
    opacities = to_numpy(gaussians['opacities'])

    N = means.shape[0]
    if N == 0:
        raise ValueError("No valid Gaussians to save")

    # 坐标系：Y 轴向上（OpenGL 约定）— 无需变换
    means_out = means
    quats_out = quats


    # ─────────────────────────────────────────────────────────────────
    # 编码为 PLY 存储格式
    # ─────────────────────────────────────────────────────────────────

    # 尺度：对数空间
    log_scales = np.log(np.clip(scales, 1e-7, None))

    # 颜色 -> SH DC 系数
    # SHARP 输出 linearRGB；Panorama2Gaussian 直接从像素得到 sRGB
    colors_clamped = np.clip(colors, 0.0, 1.0)
    if colors_linear:
        # SHARP 路径：需要 linearRGB -> sRGB
        colors_srgb = _linearRGB_to_sRGB(colors_clamped)
    else:
        # Panorama2Gaussian 路径：已是 sRGB，跳过 gamma
        colors_srgb = colors_clamped
    sh_dc = (colors_srgb - 0.5) / SH_C0

    # 不透明度：logit 空间
    opacities_clamped = np.clip(opacities, 1e-6, 1 - 1e-6)
    opacity_logit = np.log(opacities_clamped / (1 - opacities_clamped))

    # ─────────────────────────────────────────────────────────────────
    # 构建 PLY 结构
    # ─────────────────────────────────────────────────────────────────

    # 基础属性（始终存在）
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # 若阶数 > 0 则添加 SH 其余系数
    if sh_degree >= 1:
        # 总 SH 系数：(degree+1)^2 * 3 通道
        # DC 占 3 个，其余为总数减 3
        num_rest = (sh_degree + 1) ** 2 * 3 - 3
        for i in range(num_rest):
            dtype_list.append((f'f_rest_{i}', 'f4'))

    dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    data = np.zeros(N, dtype=dtype_list)

    # 填充数据
    data['x'], data['y'], data['z'] = means_out.T
    data['nx'] = data['ny'] = data['nz'] = 0  # 未使用
    data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = sh_dc.T

    # 若存在则填充 SH 其余系数
    if sh_degree >= 1:
        sh1 = gaussians.get('sh1', None)
        if sh1 is not None:
            sh1_np = to_numpy(sh1)  # [N, 9]
            # 写入每个高斯的 9 个一阶带系数
            # 3DGS PLY 中 f_rest 顺序：每个 SH 基函数按通道 R,G,B
            # f_rest_{0..2} = Y_1^-1 的 R,G,B
            # f_rest_{3..5} = Y_1^0 的 R,G,B
            # f_rest_{6..8} = Y_1^1 的 R,G,B
            for i in range(min(9, sh1_np.shape[1])):
                data[f'f_rest_{i}'] = sh1_np[:, i]
            # 更高阶其余系数置零
            num_rest = (sh_degree + 1) ** 2 * 3 - 3
            for i in range(9, num_rest):
                data[f'f_rest_{i}'] = 0
        else:
            num_rest = (sh_degree + 1) ** 2 * 3 - 3
            for i in range(num_rest):
                data[f'f_rest_{i}'] = 0

    data['opacity'] = opacity_logit.squeeze()
    data['scale_0'], data['scale_1'], data['scale_2'] = log_scales.T

    # 四元数：PLY 为 WXYZ，内部为 XYZW
    data['rot_0'] = quats_out[:, 3]  # W
    data['rot_1'] = quats_out[:, 0]  # X
    data['rot_2'] = quats_out[:, 1]  # Y
    data['rot_3'] = quats_out[:, 2]  # Z

    # 写入
    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(path)
    print(f"Saved {N:,} Gaussians to {path} (SH degree {sh_degree})")


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法（XYZW 顺序）。

    Args:
        q1: 第一个四元数 [4] 或 [1, 4]
        q2: 第二个四元数 [..., 4]

    Returns:
        乘积 q1 * q2 [..., 4]
    """
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # X
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # Y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # Z
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # W
    ], axis=-1)


def load_ply_gaussians(path: str) -> dict:
    """
    从 PLY 加载高斯。

    Args:
        path: PLY 文件路径

    Returns:
        解码后的 means、scales、quats、colors、opacities 字典
    """
    import torch

    ply = PlyData.read(path)
    vertex = ply['vertex'].data

    # 位置（已在 OpenCV 坐标系下）
    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

    # 从对数恢复尺度
    scales = np.exp(np.stack([
        vertex['scale_0'], vertex['scale_1'], vertex['scale_2']
    ], axis=-1))

    # 从 SH DC 恢复颜色
    colors = np.stack([
        vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']
    ], axis=-1) * SH_C0 + 0.5
    colors = np.clip(colors, 0, 1)

    # 从 logit 恢复不透明度
    opacity_logit = vertex['opacity']
    opacities = 1 / (1 + np.exp(-opacity_logit))

    # 四元数：PLY WXYZ → 内部 XYZW
    quats = np.stack([
        vertex['rot_1'],  # X
        vertex['rot_2'],  # Y
        vertex['rot_3'],  # Z
        vertex['rot_0'],  # W
    ], axis=-1)

    return {
        'means': torch.from_numpy(means).float(),
        'scales': torch.from_numpy(scales).float(),
        'quats': torch.from_numpy(quats).float(),
        'colors': torch.from_numpy(colors).float(),
        'opacities': torch.from_numpy(opacities[:, np.newaxis]).float(),
    }
