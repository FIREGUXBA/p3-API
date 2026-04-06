"""根据深度图统计自动计算与场景相关的参数。"""

from __future__ import annotations

import numpy as np


def compute_scene_defaults(
    depth_map: np.ndarray,
    image_height: int | None = None,
) -> dict:
    """
    根据深度图计算与尺度相关的流水线默认值。

    适用于任意场景尺度 — 3m 房间与 100m 森林均可得到合适阈值而无需手调。

    Args:
        depth_map: [H, W] 深度，单位米（0 或负值表示无效）
        image_height: 图像高度，用于像素相关参数。
            默认 depth_map.shape[0]。

    Returns:
        含 sky_threshold、depth_min、depth_max、
        orbit_radius、confidence_decay_pixels 等键的字典
    """
    if image_height is None:
        image_height = depth_map.shape[0]

    # 掩掉无效深度（零、负、无穷）
    valid = (depth_map > 0.01) & np.isfinite(depth_map)
    if valid.sum() < 100:
        # 退化深度图的回退值
        return {
            "sky_threshold": 80.0,
            "depth_min": 0.1,
            "depth_max": 100.0,
            "orbit_radius": 0.5,
            "confidence_decay_pixels": max(10, int(image_height * 0.02)),
        }

    valid_depths = depth_map[valid]

    p1 = float(np.percentile(valid_depths, 1))
    p50 = float(np.percentile(valid_depths, 50))
    p95 = float(np.percentile(valid_depths, 95))
    p99 = float(np.percentile(valid_depths, 99))

    return {
        "sky_threshold": max(p95, p1 + 1.0),  # 至少 1m 范围
        "depth_min": max(0.01, p1 * 0.8),     # 在 1% 分位下方留 20% 余量
        "depth_max": p99 * 1.1,                # 在 99% 分位上方留 10% 余量
        "orbit_radius": max(0.05, p50 * 0.05), # 中位深度的 5%
        "confidence_decay_pixels": max(10, int(image_height * 0.02)),
    }
