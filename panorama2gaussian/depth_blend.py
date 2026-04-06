# panorama2gaussian/depth_blend.py
"""
Panorama2Gaussian 的深度图融合与混合工具。

提供三种混合模式（按质量/成本排序）：
  1. FEATHERED  — 简单按到边缘距离加权平均（快，阶段 1 默认）
  2. LAPLACIAN  — 拉普拉斯金字塔频带融合（推荐）
  3. POISSON    — 梯度域泊松混合（质量最好、最慢）

主要用途：将全局一致、低频的 DAP 深度图与高频细节深度图（来自 Depth Pro、
DA3 等透视模型）融合，兼顾二者。

所有函数在 numpy float32 上工作，单位为 [0, ∞) 米。
需要：numpy、scipy、opencv-python。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import List, Optional, Tuple


class BlendMode(Enum):
    FEATHERED = "feathered"  # 简单、快速
    LAPLACIAN = "laplacian"  # 频域融合（推荐）
    POISSON   = "poisson"    # 梯度域，最慢，接缝消除最好


# ──────────────────────────────────────────────────────────────────────────────
# 高斯 / 拉普拉斯金字塔辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _build_gaussian_pyramid(image: np.ndarray, n_levels: int) -> List[np.ndarray]:
    """标准高斯金字塔：迭代模糊 + 下采样。"""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for pyramid blending. pip install opencv-python")

    pyramid = [image.astype(np.float32)]
    current = image.astype(np.float32)
    for _ in range(n_levels - 1):
        blurred    = cv2.GaussianBlur(current, (5, 5), 1.0)
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def _build_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """L[i] = G[i] - upsample(G[i+1])"""
    import cv2
    laplacian = []
    for i in range(len(gaussian_pyramid) - 1):
        g_fine   = gaussian_pyramid[i]
        g_coarse = gaussian_pyramid[i + 1]
        upsampled = cv2.resize(g_coarse, (g_fine.shape[1], g_fine.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        laplacian.append(g_fine - upsampled)
    laplacian.append(gaussian_pyramid[-1])   # 最粗层为高斯（残差）
    return laplacian


def _reconstruct_from_laplacian(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
    """由拉普拉斯金字塔重建图像。"""
    import cv2
    current = laplacian_pyramid[-1].copy()
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        finer = laplacian_pyramid[i]
        upsampled = cv2.resize(current, (finer.shape[1], finer.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        current = upsampled + finer
    return current


# ──────────────────────────────────────────────────────────────────────────────
# 对外融合函数
# ──────────────────────────────────────────────────────────────────────────────

def laplacian_depth_fusion(
    dap_depth: np.ndarray,
    dp_depth: np.ndarray,
    n_levels: int = 5,
    low_freq_cutoff: int = 2,
) -> np.ndarray:
    """
    使用拉普拉斯金字塔混合融合两幅深度图。

    低金字塔层（粗结构）来自 `dap_depth`，具有等距柱状全局一致性。
    高层（细边缘/细节）来自 `dp_depth`（Depth Pro、DA3 等透视深度）。

    Args:
        dap_depth:       (H, W) float32 — 全局一致深度（如 DAP）
        dp_depth:        (H, W) float32 — 高细节深度（如 Depth Pro）
        n_levels:        金字塔层数
        low_freq_cutoff: 层 0..cutoff 使用 dap_depth；cutoff+1..n 使用 dp_depth

    Returns:
        fused: (H, W) float32 融合深度
    """
    dap_lap = _build_laplacian_pyramid(_build_gaussian_pyramid(dap_depth, n_levels))
    dp_lap  = _build_laplacian_pyramid(_build_gaussian_pyramid(dp_depth,  n_levels))

    fused_lap = []
    for level in range(n_levels):
        # 层 0 最细，层 n-1 为最粗残差。
        if level >= n_levels - low_freq_cutoff:
            fused_lap.append(dap_lap[level])   # 粗：来自 DAP
        else:
            fused_lap.append(dp_lap[level])    # 细：来自 Depth Pro

    fused = _reconstruct_from_laplacian(fused_lap)
    return fused.clip(0.0, None).astype(np.float32)


def masked_laplacian_fusion(
    dap_depth: np.ndarray,
    dp_depth: np.ndarray,
    dp_confidence: np.ndarray,
    n_levels: int = 5,
) -> np.ndarray:
    """
    置信度加权的拉普拉斯金字塔融合。

    在每一层，两幅深度图之间的混合权重由该层下采样后的 `dp_confidence` 控制。
    细节模型置信高（锐利表面）处贡献更多；不确定处（天空、反射）由 DAP 深度主导。

    Args:
        dap_depth:      (H, W) float32 — 全局一致深度
        dp_depth:       (H, W) float32 — 高细节深度
        dp_confidence:  (H, W) float32，[0, 1] — Depth Pro / DA3 置信度
        n_levels:       金字塔层数

    Returns:
        fused: (H, W) float32
    """
    import cv2

    dap_lap  = _build_laplacian_pyramid(_build_gaussian_pyramid(dap_depth,      n_levels))
    dp_lap   = _build_laplacian_pyramid(_build_gaussian_pyramid(dp_depth,       n_levels))
    conf_pyr = _build_gaussian_pyramid(dp_confidence.astype(np.float32), n_levels)

    fused_lap = []
    for level in range(n_levels):
        conf = conf_pyr[min(level, len(conf_pyr) - 1)]
        dap_l = dap_lap[level]
        dp_l  = dp_lap[level]

        # 将置信度缩放到与本层尺寸一致
        if conf.shape != dap_l.shape:
            conf = cv2.resize(conf, (dap_l.shape[1], dap_l.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        # 反向逻辑：较低层（细细节）应对 DP 加权更高
        # 因此将置信度直接与 DP 细节相乘
        fused_l = conf * dp_l + (1.0 - conf) * dap_l
        fused_lap.append(fused_l)

    fused = _reconstruct_from_laplacian(fused_lap)
    return fused.clip(0.0, None).astype(np.float32)


def feathered_blend(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    weight_a: np.ndarray,
) -> np.ndarray:
    """
    两幅深度图的简单加权平均（alpha 混合）。

    Args:
        depth_a:  (H, W) float32
        depth_b:  (H, W) float32
        weight_a: (H, W) float32，[0, 1] — depth_a 的权重

    Returns:
        blended: (H, W) float32
    """
    w = np.clip(weight_a.astype(np.float32), 0.0, 1.0)
    return (w * depth_a + (1.0 - w) * depth_b).astype(np.float32)


def poisson_blend_faces(
    face_depths: dict,
    face_uv_maps: dict,
    target_shape: Tuple[int, int],
    dap_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将多幅深度面图以梯度域（泊松）方式混合到 ERP。

    将各面的深度梯度（按到边缘距离加权）合成，再解泊松方程得到与合成梯度
    最匹配的深度场。在数学上有利于消除接缝。

    若稀疏求解器不可用则回退为简单羽化合成。

    Args:
        face_depths:  dict[str, (H_f, W_f)] 每面对齐后的深度
        face_uv_maps: dict[str, (H_f, W_f, 2)] 每面对应的 ERP 像素坐标
        target_shape: (H, W) 目标 ERP 分辨率
        dap_depth:    (H, W) DAP 深度，用作狄利克雷边界 / 回退

    Returns:
        solved_depth: (H, W) float32
    """
    H, W = target_shape

    gradient_x  = np.zeros((H, W), dtype=np.float64)
    gradient_y  = np.zeros((H, W), dtype=np.float64)
    weight_map  = np.zeros((H, W), dtype=np.float64)
    composite   = np.zeros((H, W), dtype=np.float64)

    def _edge_weight(face_size: int) -> np.ndarray:
        """类余弦权重：中心为 1，边界为 0。"""
        r = np.arange(face_size)
        u = np.minimum(r, r[::-1]).astype(np.float64)
        half = face_size / 2.0
        w = np.clip(u / half, 0.0, 1.0)  # 线性 0→1→0
        uu, vv = np.meshgrid(w, w)
        return np.minimum(uu, vv)

    for face_dir, depth in face_depths.items():
        uv  = face_uv_maps[face_dir]           # (H_f, W_f, 2)
        H_f = depth.shape[0]
        w   = _edge_weight(H_f)                # (H_f, W_f)

        gx = np.gradient(depth.astype(np.float64), axis=1)
        gy = np.gradient(depth.astype(np.float64), axis=0)

        erp_x = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
        erp_y = np.round(uv[..., 1]).astype(int).clip(0, H - 1)

        np.add.at(gradient_x,  (erp_y, erp_x), gx * w)
        np.add.at(gradient_y,  (erp_y, erp_x), gy * w)
        np.add.at(weight_map,  (erp_y, erp_x), w)
        np.add.at(composite,   (erp_y, erp_x), depth * w)

    # 归一化累加梯度
    safe_w = np.where(weight_map > 1e-8, weight_map, 1.0)
    gradient_x /= safe_w
    gradient_y /= safe_w
    composite  /= safe_w

    # 尝试通过稀疏线性系统解泊松
    try:
        from scipy.sparse     import lil_matrix
        from scipy.sparse.linalg import spsolve

        N = H * W
        A = lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        div = (np.gradient(gradient_x, axis=1) + np.gradient(gradient_y, axis=0))

        for row in range(H):
            for col in range(W):
                idx = row * W + col
                # 边界条件：若可用则使用 dap_depth（或 composite）
                if row == 0 or row == H - 1 or col == 0 or col == W - 1:
                    A[idx, idx] = 1.0
                    if dap_depth is not None:
                        b[idx] = float(dap_depth[row, col])
                    else:
                        b[idx] = float(composite[row, col])
                    continue

                A[idx, idx]               =  4.0
                A[idx, idx - 1]           = -1.0
                A[idx, idx + 1]           = -1.0
                A[idx, idx - W]           = -1.0
                A[idx, idx + W]           = -1.0
                b[idx] = div[row, col]

        solved = spsolve(A.tocsr(), b)
        return solved.reshape(H, W).clip(0.0, None).astype(np.float32)

    except (ImportError, Exception):
        # 回退：简单合成
        if dap_depth is not None:
            gaps = weight_map < 0.01
            composite_f = composite.astype(np.float32)
            composite_f[gaps] = dap_depth[gaps].astype(np.float32)
            return composite_f
        return composite.clip(0.0, None).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 便捷封装类
# ──────────────────────────────────────────────────────────────────────────────

class DepthBlender:
    """
    所有深度融合方法的高层封装。

    用法::

        blender = DepthBlender(mode=BlendMode.LAPLACIAN)
        fused   = blender.fuse(dap_depth, detail_depth, confidence=conf_map)
    """

    def __init__(
        self,
        mode: BlendMode = BlendMode.LAPLACIAN,
        n_levels: int = 5,
        low_freq_cutoff: int = 2,
    ):
        self.mode            = mode
        self.n_levels        = n_levels
        self.low_freq_cutoff = low_freq_cutoff

    def fuse(
        self,
        dap_depth: np.ndarray,
        detail_depth: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        融合全局一致的 DAP 深度与高细节逐面深度。

        Args:
            dap_depth:    (H, W) float32 — DAP / PanDA 全局深度
            detail_depth: (H, W) float32 — 高细节深度（Depth Pro 等）
            confidence:   (H, W) float32 [0,1] — 可选逐像素置信度

        Returns:
            fused: (H, W) float32
        """
        if self.mode == BlendMode.LAPLACIAN:
            if confidence is not None:
                return masked_laplacian_fusion(
                    dap_depth, detail_depth, confidence, n_levels=self.n_levels
                )
            return laplacian_depth_fusion(
                dap_depth, detail_depth,
                n_levels=self.n_levels,
                low_freq_cutoff=self.low_freq_cutoff,
            )

        elif self.mode == BlendMode.FEATHERED:
            if confidence is not None:
                return feathered_blend(detail_depth, dap_depth, confidence)
            # 无置信度时等权 50/50
            return feathered_blend(detail_depth, dap_depth,
                                   np.full_like(dap_depth, 0.5))

        elif self.mode == BlendMode.POISSON:
            raise ValueError(
                "立方体贴图/切平面混合请直接使用 poisson_blend_faces()。"
                "DepthBlender.fuse() 不支持 POISSON 模式（需要面 UV 映射）。"
            )

        raise ValueError(f"Unknown blend mode: {self.mode}")

    def blend_cubemap_faces(
        self,
        face_depths: dict,
        face_uv_maps: dict,
        target_shape: Tuple[int, int],
        dap_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """将多幅立方体面深度合成回 ERP 空间。"""
        if self.mode == BlendMode.POISSON:
            return poisson_blend_faces(face_depths, face_uv_maps, target_shape, dap_depth)

        # LAPLACIAN 与 FEATHERED 模式下的羽化回退
        H, W = target_shape
        depth_sum  = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        for face_dir, depth in face_depths.items():
            uv   = face_uv_maps[face_dir]
            H_f  = depth.shape[0]
            half = H_f / 2.0
            r    = np.arange(H_f)
            u    = np.minimum(r, r[::-1]).astype(np.float32)
            uu, vv = np.meshgrid(u / half, u / half)
            w = np.minimum(uu, vv).clip(0.0, 1.0)

            erp_x = np.round(uv[..., 0]).astype(int).clip(0, W - 1)
            erp_y = np.round(uv[..., 1]).astype(int).clip(0, H - 1)

            np.add.at(depth_sum,  (erp_y, erp_x), depth.astype(np.float32) * w)
            np.add.at(weight_sum, (erp_y, erp_x), w)

        result = depth_sum / np.where(weight_sum > 1e-8, weight_sum, 1.0)

        if dap_depth is not None:
            gaps = weight_sum < 0.01
            result[gaps] = dap_depth.astype(np.float32)[gaps]

        return result
