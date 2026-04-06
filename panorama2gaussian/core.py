# panorama2gaussian/core.py
"""
编排转换流水线的主 Panorama2Gaussian 类。

流程：DAP/DA360 深度估计 -> Panorama2Gaussian 球面高斯转换 -> PLY 导出
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
import time

from PIL import Image, ImageOps


@dataclass
class ConversionResult:
    """Panorama2Gaussian 转换结果。"""
    output_path: str
    splat_count: int
    file_size: int
    processing_time: float
    depth_range: tuple
    depth_npy_path: Optional[str] = None
    panorama_size: Optional[tuple] = None


class Panorama2Gaussian:
    """Panorama2Gaussian：360° 全景图转高斯泼溅转换器。"""

    def __init__(
        self,
        device: str = "cuda",
        depth_model: str = "da360",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False,
    ):
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        self.default_depth_model = depth_model
        self.use_mock_dap = use_mock_dap
        self._depth_models = {}  # 惰性加载缓存：名称 -> 模型

        # 立即加载默认深度模型
        self._get_depth_model(depth_model)

    def _get_depth_model(self, name: str):
        """按名称获取或惰性加载深度模型。"""
        if name in self._depth_models:
            return self._depth_models[name]

        if self.use_mock_dap:
            from .dap_model import MockDAPModel
            model = MockDAPModel.load(device=self.device)
        elif name == "da360":
            from .da360_model import DA360Model
            model = DA360Model.load(device=self.device)
        else:
            from .dap_model import DAPModel
            model = DAPModel.load(device=self.device)

        self._depth_models[name] = model
        return model

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        depth_min: Optional[float] = None,
        depth_max: Optional[float] = None,
        sky_threshold: Optional[float] = None,
        stride: int = 2,
        outlier_pruning: float = 0.3,
        grazing_angle: float = 65.0,
        sparse_pruning: float = 0.3,
        global_scale: float = 1.0,
        force_erp: bool = False,
        depth_model: Optional[str] = None,
        grid_jitter: float = 0.0,
        depth_preview_path: Optional[Union[str, Path]] = None,
        depth_npy_path: Optional[Union[str, Path]] = None,
    ) -> ConversionResult:
        """
        将等距柱状（ERP）全景图转为高斯泼溅 PLY。

        Args:
            input_path: 输入 ERP 图像路径
            output_path: 输出 PLY 文件路径
            depth_min: 有效深度下限（米）
            depth_max: 有效深度上限（米）
            sky_threshold: 超过此深度视为天空（0 表示关闭）
            stride: Panorama2Gaussian 像素步长（1=全密度，2=四分之一，4=十六分之一）
            outlier_pruning: 统计离群点剔除强度（0=关，1=激进）
            grazing_angle: 掠射角过滤（度）
            sparse_pruning: 稀疏区域剔除强度
            global_scale: 深度手动缩放系数
            force_erp: 即使宽高比不是 2:1 也处理
            depth_model: 覆盖深度模型（"dap" 或 "da360"）
            grid_jitter: 网格抖动（预留）
            depth_preview_path: 可选，保存深度可视化图像的路径
            depth_npy_path: 可选，保存原始深度 .npy 的路径

        Returns:
            含输出详情的 ConversionResult
        """
        from .ply_writer import save_ply_gsplat

        start_time = time.time()

        # 加载并校验图像
        img = Image.open(input_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        W, H = img.size
        aspect = W / H

        if not (1.9 < aspect < 2.1) and not force_erp:
            raise ValueError(
                f"Image aspect ratio {aspect:.2f} is not 2:1. "
                "Use --force-erp to process anyway."
            )

        image_tensor = torch.from_numpy(np.array(img)).to(self.device)

        # 估计深度
        dm_name = depth_model or self.default_depth_model
        depth_engine = self._get_depth_model(dm_name)
        print(f"[Panorama2Gaussian] Running {dm_name.upper()} depth estimation...", flush=True)
        t_depth = time.time()
        with torch.inference_mode():
            depth_raw, _ = depth_engine.predict(image_tensor)
        depth = depth_raw * global_scale
        print(f"[Panorama2Gaussian] Depth estimation complete in {time.time() - t_depth:.1f}s")

        # 对任意为 None 的参数自动计算与场景相关的默认值
        depth_np = depth.cpu().numpy()
        from .scene_analysis import compute_scene_defaults
        scene_defaults = compute_scene_defaults(depth_np, image_height=H)

        if depth_min is None:
            depth_min = scene_defaults["depth_min"]
        if depth_max is None:
            depth_max = scene_defaults["depth_max"]
        if sky_threshold is None:
            sky_threshold = scene_defaults["sky_threshold"]

        print(f"[Panorama2Gaussian] Scene defaults: depth=[{depth_min:.1f}, {depth_max:.1f}]m, "
              f"sky={sky_threshold:.1f}m, orbit_r={scene_defaults['orbit_radius']:.2f}m")

        # 若指定则保存深度预览
        if depth_preview_path:
            self._save_depth_preview(depth, depth_preview_path)

        # 将原始深度保存为 numpy 数组，供下游细化使用
        if depth_npy_path:
            np.save(str(depth_npy_path), depth.cpu().numpy())

        # 运行 Panorama2Gaussian 流水线
        gaussians = self._run_panorama2gaussian_pipeline(
            image_tensor, depth,
            depth_min=depth_min, depth_max=depth_max,
            sky_threshold=sky_threshold, stride=stride,
        )
        colors_linear = False

        # 生成后滤波
        if outlier_pruning > 0.0 and gaussians['means'].shape[0] > 0:
            try:
                from .scene_filter import prune_outliers
                gaussians = prune_outliers(gaussians, strength=outlier_pruning)
            except Exception as e:
                import warnings
                warnings.warn(f"Outlier pruning failed: {e}")

        if grazing_angle < 90.0 and gaussians['means'].shape[0] > 0:
            try:
                from .scene_filter import prune_grazing_angle
                depth_np = depth.detach().cpu().numpy() if hasattr(depth, 'detach') else depth
                gaussians = prune_grazing_angle(
                    gaussians, depth_np, stride=stride, max_angle_deg=grazing_angle,
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Grazing angle filter failed: {e}")

        if sparse_pruning > 0.0 and gaussians['means'].shape[0] > 0:
            try:
                from .scene_filter import prune_sparse_regions
                # 将强度 0–1 映射为 min_neighbors 1–6
                min_n = max(1, int(1 + sparse_pruning * 5))
                gaussians = prune_sparse_regions(
                    gaussians, min_neighbors=min_n, radius_multiplier=3.0,
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Sparse region pruning failed: {e}")

        # 保存 PLY
        output_path = str(output_path)
        n_gaussians = gaussians['means'].shape[0]
        print(f"[Panorama2Gaussian] Saving PLY ({n_gaussians:,} Gaussians)...", flush=True)
        t_save = time.time()
        save_ply_gsplat(gaussians, output_path, sh_degree=0, colors_linear=colors_linear)
        print(f"[Panorama2Gaussian] Save complete in {time.time() - t_save:.1f}s")

        processing_time = time.time() - start_time
        file_size = Path(output_path).stat().st_size

        # 计算实际深度范围
        distances = gaussians['means'].norm(dim=-1)
        if distances.numel() > 0:
            depth_range = (distances.min().item(), distances.max().item())
        else:
            depth_range = (0.0, 0.0)

        return ConversionResult(
            output_path=output_path,
            splat_count=n_gaussians,
            file_size=file_size,
            processing_time=processing_time,
            depth_range=depth_range,
            depth_npy_path=str(depth_npy_path) if depth_npy_path else None,
            panorama_size=(W, H),
        )

    def _run_panorama2gaussian_pipeline(
        self,
        image_tensor: torch.Tensor,
        depth: torch.Tensor,
        depth_min: float,
        depth_max: float,
        sky_threshold: float,
        stride: int,
    ) -> dict:
        """Panorama2Gaussian 路径：由深度直接转为高斯。"""
        from .panorama2gaussian_converter import depth_to_gaussians, Panorama2GaussianParams

        print(f"[Panorama2Gaussian] Panorama2Gaussian conversion (stride={stride})...", flush=True)
        t = time.time()

        params = Panorama2GaussianParams(
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            sky_detection="depth",
        )

        gaussians = depth_to_gaussians(
            erp_image=image_tensor,
            depth_map=depth,
            params=params,
            device=self.device,
        )

        n = gaussians['means'].shape[0]
        print(f"[Panorama2Gaussian] Panorama2Gaussian generated {n:,} Gaussians in {time.time() - t:.1f}s")
        return gaussians

    @staticmethod
    def _save_depth_preview(depth: torch.Tensor, path: Union[str, Path]):
        """将深度图可视化保存为 JPEG。"""
        try:
            import cv2

            depth_np = depth.cpu().numpy()
            depth_log = np.log1p(depth_np)

            d_min = np.percentile(depth_log, 1)
            d_max = np.percentile(depth_log, 99)

            if d_max > d_min:
                depth_norm = np.clip((depth_log - d_min) / (d_max - d_min), 0, 1)
                depth_norm = (depth_norm * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_log, dtype=np.uint8)

            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(path), depth_color)
        except Exception as e:
            print(f"Failed to save depth preview: {e}")
