# panorama2gaussian/scene_filter.py
"""
Panorama2Gaussian 的天空检测与极点密度归一化。

提供控制哪些像素生成高斯的掩码：
  - 天空检测：剔除近无穷深度处的无效几何
  - 极点稀疏化：使泼溅密度在球面上近似均匀

等距柱状投影在极点过采样 — 赤道处每像素约覆盖 1 球面度，极点处约 0。
若不校正，每个泼溅场景顶部/底部都会出现致密团块。
"""

import numpy as np
import torch
from enum import Enum
from typing import Optional, Tuple


class SkyMode(Enum):
    """天空像素处理方式。"""
    SKIP = "skip"                # 完全丢弃（透明空洞）
    BACKGROUND_SPHERE = "sphere" # 放在大背景球上
    LOW_OPACITY = "low_opacity"  # 保留但不透明度降至 0.15


# ──────────────────────────────────────────────────────────────────────────────
# 天空检测
# ──────────────────────────────────────────────────────────────────────────────

def detect_sky_depth(
    depth_map: np.ndarray,
    depth_max: float = 100.0,
    threshold_ratio: float = 0.90,
) -> np.ndarray:
    """
    仅用深度阈值将像素分类为天空。

    快速且无额外模型。当深度模型给天空区域赋接近最大深度时效果很好
    （如 DAP 与 PanDA）。

    Args:
        depth_map: (H, W) float32，深度单位米
        depth_max: 预期最大场景深度
        threshold_ratio: 深度 > depth_max * 该比例 视为天空

    Returns:
        sky_mask: (H, W) bool
    """
    return depth_map > (depth_max * threshold_ratio)


def detect_sky_gradient(
    depth_map: np.ndarray,
    erp_image: np.ndarray,
    depth_max: float = 100.0,
    depth_ratio: float = 0.70,
    grad_percentile: float = 30.0,
    color_var_percentile: float = 20.0,
    patch_size: int = 32,
) -> np.ndarray:
    """
    用三种互补信号将像素分类为天空（无需额外模型）：

      1. 高深度值（> depth_max * depth_ratio）
      2. 低深度梯度幅值（深度表面平坦）
      3. 低局部颜色方差（颜色块均匀）

    Args:
        depth_map: (H, W) float32，深度单位米
        erp_image: (H, W, 3) uint8 RGB 等距柱状图
        depth_max: 预期最大场景深度
        depth_ratio: 天空候选为 depth_max 的倍数下界
        grad_percentile: 梯度分位阈值（低于 = 低梯度）
        color_var_percentile: 颜色方差分位阈值（低于 = 均匀）
        patch_size: 局部颜色方差窗口边长

    Returns:
        sky_mask: (H, W) bool
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        raise ImportError("scipy is required for detect_sky_gradient. Install with: pip install scipy")

    H, W = depth_map.shape

    # 信号 1：高深度
    high_depth = depth_map > (depth_max * depth_ratio)

    # 信号 2：低深度梯度（天空几何平坦）
    dy = np.gradient(depth_map, axis=0)
    dx = np.gradient(depth_map, axis=1)
    grad_mag = np.sqrt(dx**2 + dy**2)
    # 阈值仅在高深度区域上计算，使分位有意义（天空梯度很低；否则前景会主导）
    if high_depth.any():
        grad_thresh = np.percentile(grad_mag[high_depth], grad_percentile)
    else:
        grad_thresh = np.percentile(grad_mag, grad_percentile)
    low_grad = grad_mag < grad_thresh

    # 信号 3：低局部颜色方差（天空颜色均匀）
    gray = erp_image.mean(axis=-1).astype(np.float32)
    local_mean = uniform_filter(gray, size=patch_size)
    local_sq   = uniform_filter(gray**2, size=patch_size)
    local_var  = np.maximum(local_sq - local_mean**2, 0.0)
    var_thresh = np.percentile(local_var, color_var_percentile)
    low_var = local_var < var_thresh

    # 合并 — 三种信号须同时满足
    sky_mask = high_depth & low_grad & low_var
    return sky_mask


# ──────────────────────────────────────────────────────────────────────────────
# 极点稀疏化
# ──────────────────────────────────────────────────────────────────────────────

def compute_pole_thinning_mask(
    H: int,
    W: int,
    stride: int,
    min_density_ratio: float = 0.30,
    seed: int = 42,
) -> np.ndarray:
    """
    随机掩码，使高斯在球面上密度归一化。

    在纬度 θ 处每像素对应的立体角与 sin(θ) 成正比。
    以概率 sin(θ) 保留像素，并下限为 min_density_ratio，避免极点完全无覆盖。

    Args:
        H: 全分辨率图像高度
        W: 全分辨率图像宽度
        stride: 网格步长（掩码与步长分辨率一致）
        min_density_ratio: 最小保留概率（防止极点零覆盖）
        seed: 随机种子，可复现

    Returns:
        keep_mask: (n_rows, n_cols) bool，步长分辨率
    """
    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    n_rows = len(rows)
    n_cols = len(cols)

    # sin(θ)：极点为 0，赤道为 1（θ 为余纬度 0…π）
    theta = rows / H * np.pi
    keep_prob_1d = np.sin(theta).clip(min_density_ratio, 1.0)  # (n_rows,)

    # 广播到二维
    keep_prob_2d = np.tile(keep_prob_1d[:, None], (1, n_cols))  # (n_rows, n_cols)

    rng = np.random.default_rng(seed)
    keep_mask = rng.random(keep_prob_2d.shape) < keep_prob_2d

    return keep_mask


def get_adaptive_row_stride(
    H: int,
    base_stride: int,
    max_stride_factor: int = 4,
) -> np.ndarray:
    """
    按行自适应步长，通过纬度相关间距归一化高斯密度。

    赤道：stride = base_stride。
    近极点：stride = base_stride * max_stride_factor（更稀疏）。

    可作为随机稀疏化的替代；确定性强，极点图案更干净。

    Args:
        H: 全分辨率图像高度
        base_stride: 赤道处（θ = π/2）的步长
        max_stride_factor: 极点最大步长为 base_stride × 该因子

    Returns:
        row_strides: (H,) int，每行有效步长
    """
    rows = np.arange(H)
    theta = rows / H * np.pi  # 余纬度 0…π
    sin_theta = np.sin(theta).clip(1.0 / max_stride_factor, 1.0)

    row_strides = np.round(base_stride / sin_theta).astype(int)
    row_strides = np.clip(row_strides, base_stride, base_stride * max_stride_factor)
    return row_strides


# ──────────────────────────────────────────────────────────────────────────────
# 组合滤波
# ──────────────────────────────────────────────────────────────────────────────

def filter_gaussian_candidates(
    depth_map: np.ndarray,
    erp_image: np.ndarray,
    stride: int,
    sky_mode: SkyMode = SkyMode.SKIP,
    pole_thinning: bool = True,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    sky_detection: str = "gradient",  # "gradient" | "depth" | "none"
    min_density_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算有效高斯像素位置的布尔掩码。

    组合：
      - 深度范围校验
      - 天空检测与剔除
      - 极点密度归一化（sin(θ)）

    Args:
        depth_map: (H, W) float32，深度单位米
        erp_image: (H, W, 3) uint8 RGB（梯度天空检测需要）
        stride: 网格步长
        sky_mode: 天空像素处理方式
        pole_thinning: 是否启用 sin(θ) 密度归一化
        depth_min: 最小有效深度
        depth_max: 最大有效深度
        sky_detection: 天空检测算法
        min_density_ratio: 极点最小保留概率
        seed: 随机稀疏化种子

    Returns:
        keep_mask: (n_rows, n_cols) bool — True 表示在此生成高斯
        sky_mask_strided: (n_rows, n_cols) bool — True 表示该像素为天空
    """
    H, W = depth_map.shape

    # 网格位置上的深度
    depth_stride = depth_map[::stride, ::stride]

    # ── 深度范围 ──
    valid_depth = (depth_stride > depth_min) & (depth_stride < depth_max)

    # ── 天空检测 ──
    if sky_detection == "gradient":
        sky_mask_full = detect_sky_gradient(depth_map, erp_image, depth_max=depth_max)
    elif sky_detection == "depth":
        sky_mask_full = detect_sky_depth(depth_map, depth_max=depth_max)
    else:
        sky_mask_full = np.zeros((H, W), dtype=bool)

    # 天空掩码下采样到步长分辨率（在网格点采样）
    sky_mask_strided = sky_mask_full[::stride, ::stride]

    # ── 极点稀疏化 ──
    if pole_thinning:
        pole_mask = compute_pole_thinning_mask(H, W, stride, min_density_ratio, seed)
    else:
        n_rows = len(np.arange(0, H, stride))
        n_cols = len(np.arange(0, W, stride))
        pole_mask = np.ones((n_rows, n_cols), dtype=bool)

    # ── 合并 ──
    if sky_mode == SkyMode.SKIP:
        keep_mask = ~sky_mask_strided & valid_depth & pole_mask
    else:
        # 天空像素另作处理（背景球 / 低不透明通道）
        keep_mask = ~sky_mask_strided & valid_depth & pole_mask

    return keep_mask, sky_mask_strided


def apply_sky_mode_to_gaussians(
    gaussians: dict,
    sky_mask_strided: np.ndarray,
    depth_map_strided: np.ndarray,
    erp_image_strided: np.ndarray,
    sky_mode: SkyMode,
    depth_max: float = 100.0,
    sky_radius: float = 200.0,
    sky_opacity: float = 0.15,
    device: Optional[object] = None,
) -> dict:
    """
    可选：追加天空高斯（BACKGROUND_SPHERE 或 LOW_OPACITY 模式）。

    SKIP 模式请使用 sky_mode=SKIP 调用 filter_gaussian_candidates()，
    且不要调用本函数。

    Args:
        gaussians: 已有高斯字典（已过滤非天空像素）
        sky_mask_strided: (n_rows, n_cols) bool，步长分辨率天空掩码
        depth_map_strided: (n_rows, n_cols) 步长分辨率深度
        erp_image_strided: (n_rows, n_cols, 3) uint8 步长分辨率图像
        sky_mode: BACKGROUND_SPHERE 或 LOW_OPACITY
        depth_max: 场景深度上限（天空球半径默认相关）
        sky_radius: BACKGROUND_SPHERE 模式固定半径
        sky_opacity: 天空高斯不透明度

    Returns:
        追加天空高斯后的 gaussians（SKIP 模式则不变）
    """
    if sky_mode == SkyMode.SKIP:
        return gaussians

    if not sky_mask_strided.any():
        return gaussians

    # 步长分辨率下的天空像素位置
    sky_rows, sky_cols = np.where(sky_mask_strided)
    n_sky = len(sky_rows)

    if n_sky == 0:
        return gaussians

    H_s, W_s = sky_mask_strided.shape

    # 天空像素的球坐标
    theta = sky_rows / H_s * np.pi        # 余纬度 0…π
    phi   = sky_cols / W_s * 2 * np.pi    # 经度 0…2π

    if sky_mode == SkyMode.BACKGROUND_SPHERE:
        r = np.full(n_sky, sky_radius)
    else:  # LOW_OPACITY — 使用接近 depth_max 的距离
        r = np.full(n_sky, depth_max * 1.5)

    # 球面上三维位置（Y 向上，-Z 向前）
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = -r * np.sin(theta) * np.sin(phi)
    positions = np.stack([x, y, z], axis=-1).astype(np.float32)   # (N_sky, 3)

    # 颜色来自图像
    colors = erp_image_strided[sky_rows, sky_cols].astype(np.float32) / 255.0

    # 大而扁的公告牌作为天空
    scale_base = sky_radius * 0.04 * np.ones(n_sky, dtype=np.float32)
    scales = np.stack([scale_base, scale_base, scale_base * 0.01], axis=-1)

    # 单位四元数（各向同性朝向），XYZW
    quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_sky, 1))

    opacities = np.full((n_sky, 1), sky_opacity, dtype=np.float32)
    sh1 = np.zeros((n_sky, 9), dtype=np.float32)

    import torch
    if device is None:
        device = gaussians['means'].device

    sky_dict = {
        'means':     torch.from_numpy(positions).to(device),
        'scales':    torch.from_numpy(scales).to(device),
        'quats':     torch.from_numpy(quats).to(device),
        'colors':    torch.from_numpy(colors).to(device),
        'opacities': torch.from_numpy(opacities).to(device),
        'sh1':       torch.from_numpy(sh1).to(device),
    }

    # 与已有高斯拼接
    combined = {}
    common_keys = set(gaussians.keys()) & set(sky_dict.keys())
    for key in common_keys:
        combined[key] = torch.cat([gaussians[key], sky_dict[key]], dim=0)
    # 保留基高斯中多出的键
    for key in gaussians.keys():
        if key not in combined:
            combined[key] = gaussians[key]

    return combined

def prune_grazing_angle(
    gaussians: dict,
    depth_map: np.ndarray,
    stride: int = 2,
    max_angle_deg: float = 80.0,
) -> dict:
    """
    剔除掠射角极大的高斯：深度表面几乎与视线相切处。
    这类高斯会在岩石、物体后方产生「条纹」状伪影（深度在边缘缠绕）。

    通过局部深度梯度工作 — 相邻像素间深度剧变表示表面几乎平行于
    视线方向，这些泼溅会拉成长条。

    Args:
        gaussians: 高斯张量字典（means、scales 等）
        depth_map: (H, W) 原始深度图（numpy float32）
        stride: 转换时使用的像素步长
        max_angle_deg: 最大掠射角（度）（默认 80）。
            越小越激进剔除边缘泼溅。
            90 = 不过滤，60 = 激进。

    Returns:
        过滤后的 gaussians 字典
    """
    import torch
    if max_angle_deg >= 90.0 or gaussians['means'].shape[0] == 0:
        return gaussians

    H, W = depth_map.shape

    # 深度梯度幅值（类 Sobel）
    # θ（水平）与 φ（垂直）方向的梯度
    pad_depth = np.pad(depth_map, 1, mode='wrap')  # 水平方向环绕 360°
    grad_y = (pad_depth[2:, 1:-1] - pad_depth[:-2, 1:-1]) / 2.0
    grad_x = (pad_depth[1:-1, 2:] - pad_depth[1:-1, :-2]) / 2.0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 相对梯度：梯度/深度（无量纲）
    safe_depth = np.maximum(depth_map, 0.01)
    relative_grad = grad_mag / safe_depth

    # 将 max_angle 转为相对梯度阈值
    # tan(angle) ≈ depth_gradient / (depth * angular_spacing)
    # 与法线成 θ 角的表面：relative_grad ≈ tan(θ) * angular_spacing
    angular_spacing = np.pi / H * stride
    max_relative_grad = np.tan(np.radians(max_angle_deg)) * angular_spacing

    # 在高斯位置采样梯度
    means = gaussians['means'].detach().cpu().numpy()
    N = means.shape[0]

    # 位置反投影到像素坐标
    # positions = depth * rhat，rhat = [sin(phi)cos(theta), cos(phi), -sin(phi)sin(theta)]
    r = np.linalg.norm(means, axis=1)
    y = means[:, 1]
    phi = np.arccos(np.clip(y / np.maximum(r, 1e-8), -1, 1))  # [0, pi]
    theta = np.arctan2(-means[:, 2], means[:, 0])  # [-pi, pi]
    theta = theta % (2 * np.pi)  # [0, 2pi]

    # 映射到像素坐标
    px_row = np.clip((phi / np.pi * H).astype(int), 0, H - 1)
    px_col = np.clip((theta / (2 * np.pi) * W).astype(int), 0, W - 1)

    # 每个高斯像素处采样相对梯度
    sampled_grad = relative_grad[px_row, px_col]

    keep_mask_np = sampled_grad < max_relative_grad
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)

    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]

    removed = N - int(keep_mask_np.sum())
    if removed > 0:
        print(f"[Grazing Angle] Removed {removed:,} edge splats (max_angle={max_angle_deg}°)")

    return pruned


def prune_sparse_regions(
    gaussians: dict,
    min_neighbors: int = 3,
    radius_multiplier: float = 3.0,
    k: int = 8,
) -> dict:
    """
    剔除低密度区域的高斯：局部间距远大于预期密度处。
    与 SOR（用全局统计）不同，这里将每个泼溅的邻域距离与其自身尺度比较，
    剔除相对其预期尺寸孤立者。

    针对物体后方泼溅间距逐渐变大的「手指」伪影，同时保留远处背景等
    合理稀疏区域。

    Args:
        gaussians: 高斯张量字典
        min_neighbors: 自适应半径内最少邻居数。
            邻居更少则剔除。（默认 3）
        radius_multiplier: 搜索半径 = splat_scale * 该倍数。
            越大越宽松。（默认 3.0）
        k: 检查的最近邻数量。（默认 8）

    Returns:
        过滤后的 gaussians 字典
    """
    import torch

    N = gaussians['means'].shape[0]
    if N < k + 1:
        return gaussians

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        import warnings
        warnings.warn("scipy not installed. Skipping sparse region pruning.")
        return gaussians

    means_np = gaussians['means'].detach().cpu().numpy()
    scales_np = gaussians['scales'].detach().cpu().numpy()

    # 用最大尺度维作为预期泼溅半径
    splat_radius = np.max(scales_np, axis=1)  # [N]

    tree = cKDTree(means_np)

    # 对每个泼溅，统计半径 multiplier * scale 内的邻居数
    search_radius = splat_radius * radius_multiplier
    # 限制在合理范围，避免退化查询
    search_radius = np.clip(search_radius, 0.001, 100.0)

    # 查询 k 个最近邻
    distances, _ = tree.query(means_np, k=k + 1, workers=-1)
    neighbor_dists = distances[:, 1:]  # 排除自身

    # 统计在自适应半径内的邻居数
    within_radius = neighbor_dists < search_radius[:, np.newaxis]
    neighbor_count = np.sum(within_radius, axis=1)

    keep_mask_np = neighbor_count >= min_neighbors
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)

    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]

    removed = N - int(keep_mask_np.sum())
    if removed > 0:
        print(f"[Sparse Regions] Removed {removed:,} isolated splats (min_neighbors={min_neighbors})")

    return pruned


def prune_outliers(
    gaussians: dict,
    strength: float = 0.5,
    k: int = 16,
) -> dict:
    """
    用统计离群点剔除（SOR）移除孤立高斯（漂浮物）。

    Args:
        gaussians: 高斯张量字典（means、scales、opacities、colors、quats、sh1）
        strength: 剔除强度 0.0–1.0（越大越激进）
        k: 考虑的最近邻数量

    Returns:
        剔除后的 gaussians 字典
    """
    if strength <= 0.0 or gaussians['means'].shape[0] < k + 1:
        return gaussians

    import numpy as np
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        import warnings
        warnings.warn("scipy not installed. Skipping outlier pruning.")
        return gaussians

    means_np = gaussians['means'].detach().cpu().numpy()
    tree = cKDTree(means_np)

    # 查询 k 个最近邻距离（k+1 因包含距离为 0 的自身）
    # workers=-1 使用全部 CPU 核心
    distances, _ = tree.query(means_np, k=k + 1, workers=-1)

    # 到邻居的平均距离（排除自身）
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # 全局均值与标准差
    global_mean = np.mean(avg_distances)
    global_std = np.std(avg_distances)

    # 将 strength (0.0 - 1.0) 映射到 std_ratio (3.0 到 0.5)
    # std_ratio 定义距均值多少个标准差内可接受。
    # std_ratio 越低剔除越激进。
    # strength=0.0 -> std_ratio=3.0（温和）
    # strength=1.0 -> std_ratio=0.5（激进）
    std_ratio = 3.0 - (strength * 2.5)

    threshold = global_mean + (global_std * std_ratio)

    # 保留平均距离低于阈值的点
    keep_mask_np = avg_distances < threshold

    import torch
    keep_mask = torch.from_numpy(keep_mask_np).to(gaussians['means'].device)

    pruned = {}
    for key, tensor in gaussians.items():
        pruned[key] = tensor[keep_mask]

    removed_count = len(means_np) - int(keep_mask_np.sum())
    print(f"[Outlier Pruning] Removed {removed_count:,} floaters (strength={strength:.2f})")

    return pruned
