# panorama2gaussian/spherical_grid.py
"""
等距柱状全景图的球面网格计算。

坐标系（Y 向上，右手系）：
- θ（方位角）：+X 处为 0，从 +Y 向看逆时针增大
- φ（仰角）：+Y（北极）处为 0，-Y（南极）处为 π
"""

import torch
from dataclasses import dataclass
import math


@dataclass
class SphericalGrid:
    """
    预计算的 ERP → 高斯转换用球面网格。

    Attributes:
        theta: 方位角 [H, W]，弧度 [0, 2π]
        phi: 仰角 [H, W]，弧度 [0, π]
        rhat: 单位方向向量 [H, W, 3]（Y 向上坐标系）
        tangent_right: 切向基「右」向量 [H, W, 3]
        tangent_up: 切向基「上」向量 [H, W, 3]
        device: Torch 设备
        stride: 使用的下采样因子
    """
    theta: torch.Tensor
    phi: torch.Tensor
    rhat: torch.Tensor
    tangent_right: torch.Tensor
    tangent_up: torch.Tensor
    device: torch.device
    stride: int
    original_H: int
    original_W: int


def create_spherical_grid(
    H: int,
    W: int,
    device: torch.device,
    stride: int = 1,
    pole_rows: int = 3
) -> SphericalGrid:
    """
    为等距柱状图像创建球面网格。

    Args:
        H: 原始图像高度
        W: 原始图像宽度
        device: Torch 设备
        stride: 空间下采样因子（1、2、4、8）
        pole_rows: 极点处排除的行数

    Returns:
        含预计算几何的 SphericalGrid
    """
    # 计算步长后的尺寸
    H_strided = H // stride
    W_strided = W // stride

    # 创建像素索引（每个步长单元的中心）
    # u：水平 [0, W)，v：垂直 [0, H)
    u = torch.arange(W_strided, device=device, dtype=torch.float32) * stride + stride / 2
    v = torch.arange(H_strided, device=device, dtype=torch.float32) * stride + stride / 2

    # Meshgrid：dim 0 为 v，dim 1 为 u
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    # ERP 像素 → 球面角
    # θ：方位角 [0, 2π]，从左到右递减，使图像中心偏右处 θ=0
    # 注：uu 已为单元中心（已加 stride/2），无需再 +0.5
    theta = (1 - uu / W) * 2 * math.pi

    # φ：仰角 [0, π]，顶部（北极）为 0，底部（南极）为 π
    phi = vv / H * math.pi

    # 球面 → 笛卡尔方向（Y 向上）
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # 单位方向向量 r̂ = [sin(φ)cos(θ), cos(φ), -sin(φ)sin(θ)]
    # Y 为向上（= cos(φ)）：
    # φ=0（北极）-> cos(0)=1 -> +Y
    # φ=π（南极）-> cos(π)=-1 -> -Y
    rhat = torch.stack([
        sin_phi * cos_theta,   # X
        cos_phi,               # Y（上）
        -sin_phi * sin_theta   # Z
    ], dim=-1)

    # 计算切向基
    # 法线指向相机（原点）= -rhat
    normal = -rhat

    # 世界 up 参考
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device)

    # 广播
    up_world_expanded = up_world.view(1, 1, 3).expand(H_strided, W_strided, 3)

    # right = normalize(up_world × normal)
    right = torch.cross(up_world_expanded, normal, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    right = right / right_norm

    # 处理极点：法线 ≈ ±Y 时叉积退化
    # 极点处改用 Z 作为替代 up 参考
    pole_mask = (torch.abs(cos_phi) > 0.99).unsqueeze(-1).expand(-1, -1, 3)
    z_world = torch.tensor([0.0, 0.0, 1.0], device=device)
    z_world_expanded = z_world.view(1, 1, 3).expand(H_strided, W_strided, 3)

    right_pole = torch.cross(z_world_expanded, normal, dim=-1)
    right_pole_norm = right_pole.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    right_pole = right_pole / right_pole_norm

    right = torch.where(pole_mask, right_pole, right)

    # up = normal × right
    up = torch.cross(normal, right, dim=-1)

    return SphericalGrid(
        theta=theta,
        phi=phi,
        rhat=rhat,
        tangent_right=right,
        tangent_up=up,
        device=device,
        stride=stride,
        original_H=H,
        original_W=W
    )


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    用 Shepperd 方法将旋转矩阵转为四元数。

    比简单方法数值更稳定，避免奇点。

    Args:
        R: 旋转矩阵 [..., 3, 3]

    Returns:
        四元数 [..., 4]，XYZW 顺序
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]

    # Shepperd：选最大对角元对应的 case，避免除以很小的数
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # 预分配输出
    quat = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    # Case 1：trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        quat[mask1, 3] = 0.25 * s  # W
        quat[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # X
        quat[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # Y
        quat[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # Z

    # Case 2：R[0,0] > R[1,1] 且 R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quat[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        quat[mask2, 0] = 0.25 * s
        quat[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        quat[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3：R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quat[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        quat[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        quat[mask3, 1] = 0.25 * s
        quat[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4：其余
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quat[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        quat[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        quat[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        quat[mask4, 2] = 0.25 * s

    # 归一化
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return quat.reshape(*batch_shape, 4)
