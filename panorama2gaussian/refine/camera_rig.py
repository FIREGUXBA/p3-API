"""阶段 1：相机布置、渲染与空洞检测。"""
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 确保可导入 GSFix3D
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


@dataclass
class CameraPose:
    """用于新视角渲染的透视相机。"""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int

    @property
    def intrinsics(self):
        f = self.height / (2 * np.tan(np.radians(self.fov_deg) / 2))
        cx, cy = self.width / 2, self.height / 2
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def generate_camera_rig(
    origin: np.ndarray,
    depth_map: np.ndarray,
    num_directions: int = 12,
    num_depths: int = 3,
    fov_deg: float = 60.0,
    translation_fracs: tuple = (0.05, 0.15, 0.30),
    resolution: int = 512,
) -> list:
    """生成能暴露遮挡空洞的新视角相机。"""
    logger.info(f"[占位] generate_camera_rig: {num_directions} 方向 × {num_depths} 深度层")
    cameras = []
    median_depth = float(np.median(depth_map[depth_map > 0]))
    for azi_idx in range(num_directions):
        azimuth = (2 * np.pi * azi_idx) / num_directions
        for frac in translation_fracs:
            t = frac * median_depth
            cam_pos = origin + t * np.array([np.cos(azimuth), 0.0, np.sin(azimuth)])
            cam = CameraPose(
                position=cam_pos, look_at=origin.copy(),
                up=np.array([0.0, 1.0, 0.0]),
                fov_deg=fov_deg, width=resolution, height=resolution,
            )
            cameras.append(cam)
    return cameras


def _camera_to_RT(camera: "CameraPose"):
    """将 CameraPose 转为 GSFix3D Camera 所需的 (R, T)。

    返回:
        R: (3, 3) 世界到相机旋转矩阵（numpy float32）。
        T: (3,)   W2C 坐标系下的平移向量（numpy float32）。

    GSFix3D / Inria 约定：4×4 W2C 矩阵为 [[R, T], [0, 1]]。
    相机轴：+X 右，+Y 下，+Z 前（OpenCV / COLMAP 风格）。
    """
    forward = camera.look_at - camera.position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, camera.up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # 重新正交化 up，使其与 forward 严格垂直
    up = np.cross(right, forward)

    # R 的行是相机坐标系基向量在世界坐标中的表达
    # OpenCV：X=右，Y=下（=-up），Z=前
    R = np.stack([right, -up, forward], axis=0).astype(np.float32)

    # T = -R @ position（W2C 下的相机平移）
    T = (-R @ camera.position).astype(np.float32)

    return R, T


def render_with_hole_mask(gaussians, camera, alpha_threshold=0.1):
    """用 GSFix3D 渲染相机视角，并通过低 alpha 区域检测空洞。

    参数:
        gaussians: 已加载到 CUDA 的 GaussianModel 实例。
        camera: 描述视点的 CameraPose。
        alpha_threshold: RGB L2 范数低于此阈值的像素标为空洞
                         （背景为黑色，范数低表示无高斯覆盖）。

    返回:
        rgb: (H, W, 3) float32，取值 [0, 1]。
        hole_mask: (H, W) float32，1.0 为空洞，0.0 为已覆盖。
    """
    import torch
    from gs.camera import Camera as GSCamera
    from gs.gaussian_renderer import render as gs_render

    # CameraPose -> GSFix3D Camera
    R, T = _camera_to_RT(camera)
    fov_y_rad = np.radians(camera.fov_deg)
    aspect = camera.width / camera.height
    fov_x_rad = 2 * math.atan(math.tan(fov_y_rad / 2) * aspect)

    gs_cam = GSCamera(
        R=R, T=T,
        FoVx=fov_x_rad, FoVy=fov_y_rad,
        width=camera.width, height=camera.height,
    )

    # gs_render 所需的最小管线配置
    class _Pipe:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False

    bg = torch.zeros(3, device="cuda")

    with torch.no_grad():
        result = gs_render(gs_cam, gaussians, _Pipe(), bg)

    # result["render"] 为 (3, H, W)，值钳制在 [0, 1]
    rgb = result["render"].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # 空洞检测：背景为黑（bg=0），RGB 范数低表示该处投影的高斯很少或没有
    alpha_proxy = np.sqrt((rgb ** 2).sum(axis=2))
    hole_mask = (alpha_proxy < alpha_threshold).astype(np.float32)

    return rgb, hole_mask


def select_repair_cameras(cameras, hole_masks, min_hole_fraction=0.03, max_cameras=20):
    """筛选空洞覆盖显著的相机。"""
    scored = []
    for i, mask in enumerate(hole_masks):
        frac = float(mask.mean())
        if frac >= min_hole_fraction:
            scored.append((i, frac))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:max_cameras]]


def extract_cubemap_views(panorama, depth_map, face_size=512):
    """从等距柱状全景提取 6 个立方体面图像及对应相机。

    每个面由立方体面网格向 ERP 图像投射射线并双线性采样得到。

    参数:
        panorama: (H, W, 3) float32 等距柱状图。
        depth_map: (H, W) float32 深度图（同一投影；暂保留 API 供后续深度面提取）。
        face_size: 每个正方形面的像素边长。

    返回:
        faces:   6 个 (face_size, face_size, 3) float32 数组的列表。
        cameras: 6 个 CameraPose（原点、90° FOV）。
    """
    from scipy.ndimage import map_coordinates

    h, w = panorama.shape[:2]
    faces = []
    cameras = []

    # 每个立方体面的 (前向向量, 上向量)
    face_defs = [
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),   # 前 (-Z)
        (np.array([0.0, 0.0, 1.0]),  np.array([0.0, 1.0, 0.0])),   # 后 (+Z)
        (np.array([1.0, 0.0, 0.0]),  np.array([0.0, 1.0, 0.0])),   # 右 (+X)
        (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),   # 左 (-X)
        (np.array([0.0, 1.0, 0.0]),  np.array([0.0, 0.0, -1.0])),  # 顶 (+Y)
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # 底 (-Y)
    ]

    for forward, up in face_defs:
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)  # 重新正交化

        # 在 90° FOV 网格上构建每像素射线方向
        # v 自上而下 = +up 到 -up（与 OpenCV 相机约定一致：第 0 行朝世界上方，最后一行朝下方）
        u = np.linspace(-1.0, 1.0, face_size)
        v = np.linspace(1.0, -1.0, face_size)
        uu, vv = np.meshgrid(u, v)  # (face_size, face_size)

        # 每像素方向 = forward + u*right + v*up（未单位化）
        dirs = (uu[..., None] * right
                + vv[..., None] * up
                + forward)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # 三维方向 -> 等距柱状像素坐标
        # 需与 SPAG 的 spherical_grid.py 一致：
        #   θ = atan2(-Z, X)，映射到 [0, 2π]
        #   φ = acos(Y)，映射到 [0, π]
        #   pixel_x = (1 - θ/(2π)) * (w - 1)
        #   pixel_y = φ/π * (h - 1)
        theta_spag = np.arctan2(-dirs[..., 2], dirs[..., 0])  # [-pi, pi]
        theta_spag = theta_spag % (2 * np.pi)                 # [0, 2pi]
        phi_spag = np.arccos(np.clip(dirs[..., 1], -1.0, 1.0))  # [0, pi]

        px = (1.0 - theta_spag / (2 * np.pi)) * (w - 1)      # [0, w-1]
        py = phi_spag / np.pi * (h - 1)                       # [0, h-1]

        # 各通道双线性采样（mode='wrap' 处理水平接缝）
        face = np.zeros((face_size, face_size, 3), dtype=np.float32)
        for c in range(3):
            face[..., c] = map_coordinates(
                panorama[..., c].astype(np.float64),
                [py, px],
                order=1,
                mode="wrap",
            ).astype(np.float32)

        faces.append(face)
        cameras.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=forward.astype(np.float64),
            up=up.astype(np.float64),
            fov_deg=90.0,
            width=face_size,
            height=face_size,
        ))

    return faces, cameras
