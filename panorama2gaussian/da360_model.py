# panorama2gaussian/da360_model.py
"""
DA360（Depth Anything 360）模型封装。

DA360 是在 Depth Anything V2（ViT-Large）基础上微调，并在 DPT 解码器中使用
循环填充（circular padding），以实现无缝 360° 深度估计。输出为尺度不变的
视差（非米制深度），需取逆并做尺度归一化。

关键架构要点：
  - DPT 头中所有 Conv2d 均替换为 ERPCircularConv2d
  - 水平方向：标准循环包裹（360° 连续）
  - 垂直/极点：按 W/2 滚动 + 翻转（正确极点几何）
  - ViT class token 上的 Shift MLP：仿射不变 -> 尺度不变视差
  - 输入：518×1036 ERP，ImageNet 归一化

参考：https://github.com/Insta360-Research-Team/DA360
论文："Depth Anything in 360: Towards Scale Invariance in the Wild"
许可：MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import hashlib


DA360_CONFIG = {
    # 权重在 Google Drive，通过 gdown 或手动下载
    "gdrive_folder": "https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2",
    "filename": "DA360_large.pth",
    "sha256": None,
    "size_mb": 500,
}
DA360_CACHE_DIR = Path.home() / ".cache" / "panorama2gaussian"

# DA360 固定输入分辨率（37*14=518 个 patch，ViT patch_size=14）
DA360_INPUT_H = 518
DA360_INPUT_W = 1036


class DA360Model:
    """
    DA360 深度估计模型封装。

    DA360 输出尺度不变视差，通过取逆并基于中位数的尺度归一化转为深度。
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        device: torch.device = torch.device('cuda')
    ) -> 'DA360Model':
        """
        从路径加载 DA360 模型或下载权重。

        Args:
            model_path: 可选，权重文件显式路径
            device: 加载到的 Torch 设备

        Returns:
            已加载的 DA360Model 实例
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()

        # 导入 DA360 架构
        try:
            from .da360_arch import build_da360_model
        except ImportError:
            raise ImportError(
                "DA360 architecture not found. Please set up DA360:\n"
                "1. Clone https://github.com/Insta360-Research-Team/DA360 "
                "into panorama2gaussian/da360_arch/DA360/\n"
                "2. Download DA360_large.pth from Google Drive\n"
            )

        model = build_da360_model()

        # 加载检查点 — DA360 将元数据键（'net'、'dinov2_encoder'、'height'、'width'）
        # 与模型 state_dict 条目混在同一扁平字典中
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        metadata_keys = {'net', 'dinov2_encoder', 'height', 'width'}
        if isinstance(checkpoint, dict):
            model_state = model.state_dict()
            # 过滤：跳过元数据键，仅保留与模型匹配的键
            filtered = {}
            for k, v in checkpoint.items():
                if k in metadata_keys:
                    continue
                # 若存在则去掉 'module.' 前缀（DataParallel）
                clean_k = k[7:] if k.startswith('module.') else k
                if clean_k in model_state:
                    filtered[clean_k] = v
            model.load_state_dict(filtered, strict=False)
            loaded = len(filtered)
            total = len(model_state)
            print(f"[DA360] Loaded {loaded}/{total} parameters")
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        return cls(model, device)

    @classmethod
    def _get_or_download_weights(cls) -> str:
        """下载权重或定位缓存文件。"""
        DA360_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DA360_CACHE_DIR / DA360_CONFIG["filename"]

        if cache_path.exists():
            if cls._verify_checksum(cache_path):
                print(f"Using cached DA360 weights: {cache_path}")
                return str(cache_path)
            else:
                print("Cached DA360 weights corrupted, re-downloading...")
                cache_path.unlink()

        # 尝试 gdown（Google Drive 下载器）
        print(f"Downloading DA360 weights (~{DA360_CONFIG['size_mb']}MB)...")
        try:
            import gdown
            # Google Drive 上的 DA360_large.pth 文件
            gdown.download(
                url=DA360_CONFIG["gdrive_folder"],
                output=str(DA360_CACHE_DIR),
                fuzzy=True,
                quiet=False,
            )
            if cache_path.exists():
                return str(cache_path)
        except ImportError:
            pass

        raise RuntimeError(
            f"DA360 weights not found at {cache_path}\n\n"
            "Download DA360_large.pth manually from:\n"
            f"  {DA360_CONFIG['gdrive_folder']}\n"
            f"Place it at: {cache_path}\n\n"
            "Or install gdown: pip install gdown"
        )

    @staticmethod
    def _verify_checksum(path: Path) -> bool:
        """校验文件 SHA256。"""
        if not DA360_CONFIG.get("sha256"):
            return True
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == DA360_CONFIG["sha256"]

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        由等距柱状图像预测深度。

        DA360 输出尺度不变视差；我们取逆得到深度，并通过中位数归一化得到
        近似的米制尺度。

        Args:
            image: RGB 张量 [H, W, 3] 或 [B, H, W, 3]，uint8 或 [0,1] float
            return_mask: 忽略（DA360 不产生 mask）

        Returns:
            (depth, None) 元组：
                - depth: [H, W] 或 [B, H, W]，近似米
                - mask: 恒为 None
        """
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)

        B, H, W, C = image.shape

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # DA360 期望 [B, C, H, W] 且经 ImageNet 归一化
        x = image.permute(0, 3, 1, 2).to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 缩放到 DA360 固定输入分辨率（518×1036）
        x = F.interpolate(x, size=(DA360_INPUT_H, DA360_INPUT_W),
                          mode='bilinear', align_corners=True)

        # 前向推理
        output = self.model(x)

        # 从输出中提取视差
        if isinstance(output, dict):
            disparity = output.get('pred_disp', None)
            if disparity is None:
                # 回退到任意可用键
                disparity = next(iter(output.values()))
        elif isinstance(output, (tuple, list)):
            disparity = output[0]
        else:
            disparity = output

        # 去掉通道维 [B, 1, H, W] -> [B, H, W]
        if disparity.dim() == 4:
            disparity = disparity.squeeze(1)

        # 视差 -> 深度：depth = 1 / disparity
        eps = 1e-6
        depth = 1.0 / (disparity.abs() + eps)

        # 基于中位数的尺度归一化，近似米制深度
        # 将中位深度设为约 5m（室内外合理中点）
        for i in range(B):
            median_depth = depth[i].median()
            if median_depth > eps:
                depth[i] = depth[i] * (5.0 / median_depth)

        # 上采样回原始分辨率
        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze(1)

        if not is_batched:
            depth = depth.squeeze(0)

        return depth, None
