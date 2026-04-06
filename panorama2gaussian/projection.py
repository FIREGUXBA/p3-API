# panorama2gaussian/projection.py
"""
ERP ↔ 切平面投影工具。

支持多种投影模式：
- 立方体贴图（6 面，90° 视场）
- 二十面体（20 面，约 72° 视场并带重叠）
"""

import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Literal
import math


class BaseProjector(ABC):
    """ERP ↔ 切平面投影器基类。"""

    def __init__(self, face_size: int, device: torch.device):
        self.face_size = face_size
        self.device = device
        self._sampling_grids = None  # 缓存的采样网格

    @property
    @abstractmethod
    def num_faces(self) -> int:
        """投影面数量。"""
        pass

    @property
    @abstractmethod
    def face_directions(self) -> List[np.ndarray]:
        """各面中心方向（单位向量）列表。"""
        pass

    @property
    @abstractmethod
    def face_ups(self) -> List[np.ndarray]:
        """各面向上向量列表。"""
        pass

    @property
    @abstractmethod
    def face_fov(self) -> float:
        """每面视场角（弧度）。"""
        pass

    def project_erp_to_faces(self, erp_image: np.ndarray) -> List[np.ndarray]:
        """将 ERP 投影为 N 张切平面图像。"""
        H, W = erp_image.shape[:2]
        erp_tensor = torch.from_numpy(erp_image).float().to(self.device)

        faces = []
        for i in range(self.num_faces):
            face = self._sample_tangent_plane(
                erp_tensor,
                self.face_directions[i],
                self.face_ups[i],
                H, W
            )
            faces.append(face.cpu().numpy().astype(np.uint8))

        return faces

    def _sample_tangent_plane(
        self,
        erp: torch.Tensor,
        center_dir: np.ndarray,
        up_dir: np.ndarray,
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """
        用日晷投影（gnomonic）从 ERP 采样切平面。
        """
        # 建立切平面坐标系
        forward = center_dir / np.linalg.norm(center_dir)
        up = up_dir / np.linalg.norm(up_dir)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # 在切平面坐标中创建采样网格
        half_fov = self.face_fov / 2
        tan_half = math.tan(half_fov)

        # 网格在 [-tan_half, tan_half]
        u = torch.linspace(-tan_half, tan_half, self.face_size, device=self.device)
        v = torch.linspace(-tan_half, tan_half, self.face_size, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='xy')

        # 切平面内三维方向
        right_t = torch.from_numpy(right).float().to(self.device)
        up_t = torch.from_numpy(up).float().to(self.device)
        forward_t = torch.from_numpy(forward).float().to(self.device)

        # 射线方向：forward + u*right + v*up
        dirs = (forward_t.view(1, 1, 3) +
                uu.unsqueeze(-1) * right_t.view(1, 1, 3) +
                vv.unsqueeze(-1) * up_t.view(1, 1, 3))

        # 归一化到单位球
        dirs = F.normalize(dirs, dim=-1)

        # 转为球坐标 (theta, phi) 再映射到 ERP 像素坐标
        # theta = 方位角 [0, 2π]，phi = 极角 [0, π]
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        theta = torch.atan2(-z, x)  # [-π, π] -> [0, 2π]
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        phi = torch.acos(y.clamp(-1, 1))  # [0, π]

        # 归一化到 [-1, 1] 供 grid_sample 使用
        u_erp = 1 - (theta / math.pi)  # theta=0 -> u_erp=1（右），theta=2pi -> u_erp=-1（左）
        v_erp = (phi / math.pi) * 2 - 1          # [0, π] -> [-1, 1]

        grid = torch.stack([u_erp, v_erp], dim=-1).unsqueeze(0)

        # 采样 ERP
        erp_chw = erp.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        sampled = F.grid_sample(
            erp_chw, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # [H, W, 3]
        return sampled.squeeze(0).permute(1, 2, 0)

    def reproject_to_erp(
        self,
        face_features: List[torch.Tensor],
        erp_h: int,
        erp_w: int
    ) -> torch.Tensor:
        """
        将 N 个面的特征按距离加权混合重投影回 ERP。
        """
        C = face_features[0].shape[-1] if face_features[0].dim() == 3 else 1

        # 累加加权特征
        result = torch.zeros(erp_h, erp_w, C, device=self.device)
        weights = torch.zeros(erp_h, erp_w, device=self.device)

        for i in range(self.num_faces):
            feat = face_features[i]
            if feat.dim() == 2:
                feat = feat.unsqueeze(-1)

            # 在每个 ERP 像素处采样该面的贡献
            contribution, weight = self._sample_face_to_erp(
                feat,
                self.face_directions[i],
                self.face_ups[i],
                erp_h, erp_w
            )

            result += contribution * weight.unsqueeze(-1)
            weights += weight

        # 按总权重归一化
        result = result / weights.unsqueeze(-1).clamp(min=1e-6)

        if C == 1:
            result = result.squeeze(-1)

        return result

    def _sample_face_to_erp(
        self,
        face_feat: torch.Tensor,
        center_dir: np.ndarray,
        up_dir: np.ndarray,
        erp_h: int,
        erp_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将面特征按角权重采样到 ERP 网格上。
        """
        C = face_feat.shape[-1]

        # 建立切平面坐标系
        forward = center_dir / np.linalg.norm(center_dir)
        up = up_dir / np.linalg.norm(up_dir)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # 创建 ERP 像素方向
        u_erp = torch.linspace(0, 1, erp_w, device=self.device)
        v_erp = torch.linspace(0, 1, erp_h, device=self.device)
        uu, vv = torch.meshgrid(u_erp, v_erp, indexing='xy')

        theta = (1 - uu) * 2 * math.pi  # uu=0（左）-> theta=2pi，uu=1（右）-> theta=0
        phi = vv * math.pi        # [0, π]

        # 球面到笛卡尔
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.cos(phi)
        z = -torch.sin(phi) * torch.sin(theta)

        dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]

        # 投影到切平面
        forward_t = torch.from_numpy(forward).float().to(self.device)
        right_t = torch.from_numpy(right).float().to(self.device)
        up_t = torch.from_numpy(up).float().to(self.device)

        # 与 forward 点积得深度
        depth = (dirs * forward_t).sum(dim=-1)  # [H, W]

        # 仅在 depth > 0（在切平面前方）时有效
        valid = depth > 0.1

        # 投影到切平面坐标
        proj_right = (dirs * right_t).sum(dim=-1) / depth.clamp(min=0.1)
        proj_up = (dirs * up_t).sum(dim=-1) / depth.clamp(min=0.1)

        # 转为 grid_sample 的归一化坐标
        half_fov = self.face_fov / 2
        tan_half = math.tan(half_fov)

        u_face = proj_right / tan_half  # [-1, 1]
        v_face = proj_up / tan_half     # [-1, 1]

        # 软权重：严格在后方或远出面外的像素权重为 0。
        # 面内使用平滑 cos² 衰减，从中心 1 衰减到边界 0 — 消除硬接缝。
        in_front = valid  # 半球可见性

        # 与面中心的角距离（类高斯衰减）
        cos_center = (dirs * forward_t).sum(dim=-1).clamp(-1, 1)
        angular_dist = torch.acos(cos_center)
        gaussian_weight = torch.exp(-angular_dist ** 2 / (self.face_fov ** 2 / 4))

        # 基于面内归一化位置的 cos² 锥形
        # u_face、v_face 在采样系中为 [-1, 1]；重叠区略超出 ±1，
        # 故在 ±base_half 外开始衰减。
        base_half = 1.0 / (1.0 + self.overlap_ratio)  # overlap=0.25 时约 0.8
        edge_range = max(1 - base_half, 1e-6)  # python float，安全除法
        edge_dist_u = (u_face.abs() - base_half).clamp(0, edge_range)
        edge_dist_v = (v_face.abs() - base_half).clamp(0, edge_range)
        edge_dist = torch.max(edge_dist_u, edge_dist_v) / edge_range
        taper = torch.cos(edge_dist * math.pi / 2) ** 2  # 从 base 到边缘 1 → 0

        # 超出完整面范围 → 零
        in_bounds = (u_face.abs() <= 1) & (v_face.abs() <= 1) & in_front
        weight = gaussian_weight * taper * in_bounds.float()

        # 采样面特征
        grid = torch.stack([u_face, v_face], dim=-1).unsqueeze(0)
        face_chw = face_feat.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        sampled = F.grid_sample(
            face_chw, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        contribution = sampled.squeeze(0).permute(1, 2, 0)  # [H, W, C]

        return contribution, weight


class CubemapProjector(BaseProjector):
    """6 面立方体贴图投影，带混合重叠。"""

    # 标准立方体贴图面方向（OpenGL 约定）
    FACE_DIRS = [
        np.array([1, 0, 0]),   # +X（右）
        np.array([-1, 0, 0]),  # -X（左）
        np.array([0, 1, 0]),   # +Y（上）
        np.array([0, -1, 0]),  # -Y（下）
        np.array([0, 0, 1]),   # +Z（前）
        np.array([0, 0, -1]), # -Z（后）
    ]

    FACE_UPS = [
        np.array([0, 1, 0]),   # +X
        np.array([0, 1, 0]),   # -X
        np.array([0, 0, -1]),  # +Y
        np.array([0, 0, 1]),   # -Y
        np.array([0, 1, 0]),   # +Z
        np.array([0, 1, 0]),   # -Z
    ]

    def __init__(self, face_size: int, device: torch.device, overlap_ratio: float = 0.55):
        super().__init__(face_size, device)
        self.overlap_ratio = overlap_ratio

    @property
    def num_faces(self) -> int:
        return 6

    @property
    def face_directions(self) -> List[np.ndarray]:
        return self.FACE_DIRS

    @property
    def face_ups(self) -> List[np.ndarray]:
        return self.FACE_UPS

    @property
    def face_fov(self) -> float:
        return (math.pi / 2) * (1 + self.overlap_ratio)  # 90° + 重叠


class IcosahedralProjector(BaseProjector):
    """
    20 面二十面体切平面投影。

    在二十面体各面心放置重叠的方形切平面。更大重叠可在接缝处平滑混合。
    """

    def __init__(self, face_size: int, device: torch.device, overlap_ratio: float = 0.3):
        super().__init__(face_size, device)
        self.overlap_ratio = overlap_ratio
        self._face_dirs = None
        self._face_ups = None
        self._compute_icosahedron_geometry()

    def _compute_icosahedron_geometry(self):
        """计算二十面体各面中心与朝向。"""
        # 黄金比例
        phi = (1 + math.sqrt(5)) / 2

        # 正二十面体 12 个顶点
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        # 归一化到单位球
        vertices = vertices / np.linalg.norm(vertices[0])

        # 20 个三角形面（顶点索引）
        faces = [
            (0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 9), (0, 9, 1),
            (3, 2, 10), (3, 10, 6), (3, 6, 7), (3, 7, 11), (3, 11, 2),
            (1, 6, 8), (8, 6, 10), (8, 10, 4), (4, 10, 2), (4, 2, 5),
            (5, 2, 11), (5, 11, 9), (9, 11, 7), (9, 7, 1), (1, 7, 6)
        ]

        # 计算面心（质心）
        self._face_dirs = []
        self._face_ups = []

        for f in faces:
            center = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3
            center = center / np.linalg.norm(center)  # 归一化
            self._face_dirs.append(center)

            # 向上向量：与 center 垂直，大致指向 +Y
            world_up = np.array([0, 1, 0])
            right = np.cross(world_up, center)
            if np.linalg.norm(right) < 0.01:  # 近极点
                right = np.cross(np.array([1, 0, 0]), center)
            right = right / np.linalg.norm(right)
            up = np.cross(center, right)
            up = up / np.linalg.norm(up)
            self._face_ups.append(up)

    @property
    def num_faces(self) -> int:
        return 20

    @property
    def face_directions(self) -> List[np.ndarray]:
        return self._face_dirs

    @property
    def face_ups(self) -> List[np.ndarray]:
        return self._face_ups

    @property
    def face_fov(self) -> float:
        # 约 72° 基底 + 重叠
        base_fov = 2 * math.atan(1 / math.sqrt(5))  # 内接约 63.4°
        return base_fov * (1 + self.overlap_ratio)


def get_projector(
    mode: Literal["cubemap", "icosahedral"],
    face_size: int,
    device: torch.device
) -> BaseProjector:
    """按模式名返回投影器工厂函数。"""
    if mode == "cubemap":
        return CubemapProjector(face_size, device)
    elif mode == "icosahedral":
        return IcosahedralProjector(face_size, device)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
