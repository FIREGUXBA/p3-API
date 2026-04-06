# panorama2gaussian/depth_refiner.py
"""
RGB 引导的深度边缘细化。

使用引导滤波将 RGB 全景的锐利边缘迁移到深度图，
在保持全局深度结构的同时得到边界清晰的深度。

基于 "Guided Image Filtering"（He 等，2010/2013）。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GuidedDepthRefiner:
    """
    使用全景 RGB 作为引导来细化深度边缘。

    引导滤波利用 RGB 边缘来锐化对应深度边缘，而不改变全局深度结构。
    对 360° 深度图尤其有效，因为模型输出往往在物体边界处偏模糊。

    流程：
        1. 将深度与 RGB 转为 numpy
        2. 应用引导滤波（RGB 引导深度边缘位置）
        3. 返回锐化后的深度张量
    """

    def __init__(
        self,
        radius: int = 8,
        eps: float = 1e-4,
        use_opencv: bool = True,
    ):
        """
        Args:
            radius: 滤波半径（越大越平滑，但保留更多边缘）
            eps: 正则项（越小边缘越锐，易产生伪影）
            use_opencv: 是否优先尝试 OpenCV ximgproc（更快），否则回退纯 Python
        """
        self.radius = radius
        self.eps = eps
        self.use_opencv = use_opencv
        self._has_ximgproc = None  # 惰性检测

    def refine(
        self,
        depth: torch.Tensor,
        rgb_guide: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        使用 RGB 引导细化深度边缘。

        Args:
            depth: 深度图 [H, W]（任意尺度/范围）
            rgb_guide: RGB 图像 [H, W, 3]，float [0, 1]
            strength: 混合系数（0.0=原始，1.0=完全细化后）

        Returns:
            边缘更锐利的深度 [H, W]
        """
        device = depth.device

        # 校验
        strength = max(0.0, min(1.0, strength))
        if strength <= 0.001:
            return depth

        # 转为 numpy 以便滤波
        depth_np = depth.cpu().numpy().astype(np.float32)
        guide_np = rgb_guide.cpu().numpy().astype(np.float32)

        # 优先 OpenCV 路径（更快），否则回退纯 Python
        result_np = self._try_opencv_guided_filter(depth_np, guide_np)
        if result_np is None:
            result_np = self._guided_filter_python(depth_np, guide_np)

        # 转回张量
        result = torch.from_numpy(result_np).to(device)

        # 不要对结果做全局 min/max 重标定 — 那会抵消引导滤波带来的
        # 局部边缘锐化（锐化一处会迫使其它深度整体偏移）。
        # 下面的 strength 混合足以保持范围。

        # 按 strength 混合
        if strength < 1.0:
            result = result * strength + depth * (1.0 - strength)

        # 避免引导滤波振铃导致深度 < 0
        result = result.clamp(min=1e-4)

        return result


    def _try_opencv_guided_filter(
        self,
        depth_np: np.ndarray,
        guide_np: np.ndarray,
    ) -> Optional[np.ndarray]:
        """尝试 OpenCV ximgproc 引导滤波；不可用则返回 None。"""
        if not self.use_opencv:
            return None

        if self._has_ximgproc is None:
            try:
                import cv2
                # 检查 ximgproc 是否可用
                cv2.ximgproc.guidedFilter
                self._has_ximgproc = True
            except (ImportError, AttributeError):
                self._has_ximgproc = False
                print("[GuidedFilter] OpenCV ximgproc not available, using pure Python fallback")

        if not self._has_ximgproc:
            return None

        import cv2

        # OpenCV guidedFilter 期望引导图为 uint8 或 float32
        guide_uint8 = (guide_np * 255).clip(0, 255).astype(np.uint8)

        # 应用引导滤波
        result = cv2.ximgproc.guidedFilter(
            guide=guide_uint8,
            src=depth_np,
            radius=self.radius,
            eps=self.eps,
        )

        return result

    def _guided_filter_python(
        self,
        src: np.ndarray,
        guide: np.ndarray,
    ) -> np.ndarray:
        """
        纯 Python/NumPy 引导滤波实现。

        基于 He 等 2013 的 O(N) 盒式滤波形式。
        """
        r = self.radius
        eps = self.eps

        # 彩色引导则转灰度
        if guide.ndim == 3:
            guide_gray = np.mean(guide, axis=2).astype(np.float32)
        else:
            guide_gray = guide.astype(np.float32)

        src = src.astype(np.float32)

        # 用前缀和实现盒式滤波（每通道 O(N)）
        def box_filter(img, r):
            """用积分图实现快速盒式滤波。"""
            H, W = img.shape[:2]
            # 填充
            padded = np.pad(img, ((r, r), (r, r)), mode='reflect')
            # 积分图
            integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            # 盒式滤波结果
            result = (
                integral[2*r:, 2*r:]
                - integral[:H, 2*r:]
                - integral[2*r:, :W]
                + integral[:H, :W]
            )
            # 按窗口大小归一化
            count = (2 * r + 1) ** 2
            return result / count

        # 引导滤波核心
        mean_I = box_filter(guide_gray, r)
        mean_p = box_filter(src, r)
        mean_Ip = box_filter(guide_gray * src, r)
        mean_II = box_filter(guide_gray * guide_gray, r)

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)

        result = mean_a * guide_gray + mean_b

        return result
