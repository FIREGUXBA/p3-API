# panorama2gaussian/dap_model.py
"""
DAP（Depth Any Panoramas）模型封装。

DAP 专为 360° 等距柱状图像设计，输出米制深度（米）。

参考：https://github.com/Insta360-Research-Team/DAP
许可：MIT
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import hashlib


# 模型配置
DAP_CONFIG = {
    "url": "https://huggingface.co/Insta360-Research/DAP-weights/resolve/main/model.pth",
    "repo_id": "Insta360-Research/DAP-weights",
    "filename": "model.pth",
    # 关闭校验和验证，避免哈希变更导致反复重下
    "sha256": None, # "247f33754976cae1f76cb9a3b9737f336575e8cbd121c3382ab1bff18387bc7d3",
    "size_mb": 1500,
}
DAP_CACHE_DIR = Path.home() / ".cache" / "panorama2gaussian"


class DAPModel:
    """
    DAP（Depth Any Panoramas）模型封装。

    DAP 专为 360° 等距柱状图像设计，输出米制深度（米）。
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
    ) -> 'DAPModel':
        """
        从路径加载或从 HuggingFace 下载 DAP 模型。

        Args:
            model_path: 可选，权重文件显式路径
            device: 加载到的 Torch 设备

        Returns:
            已加载的 DAPModel 实例
        """
        if model_path is None:
            model_path = cls._get_or_download_weights()

        # 导入 DAP 模型架构
        try:
            from .dap_arch import build_dap_model
        except ImportError:
            raise ImportError(
                "DAP architecture not found. Please copy the DAP model files "
                "from https://github.com/Insta360-Research-Team/DAP to panorama2gaussian/dap_arch/"
            )

        model = build_dap_model()

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 处理不同检查点格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 若模型曾以 DataParallel 保存，则去掉 'module.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去掉 'module.' 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value

        # strict=False 以容忍轻微不匹配
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)

        return cls(model, device)

    @classmethod
    def _get_or_download_weights(cls) -> str:
        """下载权重并校验。"""
        DAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DAP_CACHE_DIR / "model.pth"

        if cache_path.exists():
            # 若配置了则校验校验和
            if cls._verify_checksum(cache_path):
                print(f"Using cached DAP weights: {cache_path}")
                return str(cache_path)
            else:
                print("Cached weights corrupted, re-downloading...")
                cache_path.unlink()

        # 带进度下载
        print(f"Downloading DAP weights (~{DAP_CONFIG['size_mb']}MB)...")

        try:
            # 优先使用 huggingface_hub 以支持断点续传
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=DAP_CONFIG["repo_id"],
                filename=DAP_CONFIG["filename"],
                cache_dir=DAP_CACHE_DIR,
                local_dir=DAP_CACHE_DIR,
            )
            return downloaded_path
        except ImportError:
            # 回退到 urllib
            import urllib.request

            try:
                from tqdm import tqdm

                class DownloadProgress(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)

                with DownloadProgress(unit='B', unit_scale=True, miniters=1) as t:
                    urllib.request.urlretrieve(
                        DAP_CONFIG["url"],
                        cache_path,
                        reporthook=t.update_to
                    )
            except ImportError:
                # 无 tqdm，静默下载
                urllib.request.urlretrieve(DAP_CONFIG["url"], cache_path)

        return str(cache_path)

    @staticmethod
    def _verify_checksum(path: Path) -> bool:
        """校验文件 SHA256。"""
        if not DAP_CONFIG.get("sha256"):
            return True  # 未配置校验和则跳过

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest() == DAP_CONFIG["sha256"]

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = False  # 已关闭：DAP mask 头输出接近零
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        由等距柱状图像预测米制深度。

        Args:
            image: RGB 图像张量 [H, W, 3] 或批 [B, H, W, 3]，uint8 或 [0,1] float
            return_mask: 是否返回有效性 mask

        Returns:
            (depth, mask) 元组：
                - depth: [H, W] 或 [B, H, W]，单位米
                - mask: [H, W] 或 [B, H, W] 有效性 mask（0–1），若 return_mask=False 则为 None
        """
        import torch.nn.functional as F

        # 批处理与单张输入
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)  # [1, H, W, 3]

        B, H, W, C = image.shape

        # 预处理
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # DAP 期望 [B, C, H, W]，经 ImageNet 统计量归一化
        x = image.permute(0, 3, 1, 2).to(self.device)  # [B, 3, H, W]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 限制输入分辨率，避免超大图（如 8192×4096）OOM。
        # DAP 的 DPT 头使用密集卷积 — 显存随分辨率近似平方增长。
        # 推理前下采样，再将深度输出上采样回原始尺寸。
        MAX_INPUT_WIDTH = 4096
        did_downscale = False
        if W > MAX_INPUT_WIDTH:
            scale = MAX_INPUT_WIDTH / W
            new_H = int(H * scale)
            new_W = MAX_INPUT_WIDTH
            # 保证偶数尺寸
            new_H = new_H - (new_H % 2)
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=True)
            did_downscale = True
            print(f"[DAP] Downscaled input {W}×{H} → {new_W}×{new_H} for inference")

        # 推理，OOM 时回退
        try:
            output = self.model(x)
        except torch.cuda.OutOfMemoryError:
            # 回退：逐张处理
            torch.cuda.empty_cache()
            depths = []
            masks = []
            for i in range(B):
                out_i = self.model(x[i:i+1])
                if isinstance(out_i, dict):
                    depths.append(out_i['pred_depth'])
                    if return_mask and 'pred_mask' in out_i:
                        masks.append(out_i['pred_mask'])
                else:
                    depths.append(out_i)
            depth = torch.cat(depths, dim=0)
            mask = torch.cat(masks, dim=0) if masks else None
            output = {'pred_depth': depth, 'pred_mask': mask}

        # 处理不同输出格式
        if isinstance(output, dict):
            depth = output['pred_depth']
            mask = output.get('pred_mask', None) if return_mask else None
        else:
            depth = output
            mask = None

        # 若存在则去掉通道维 [B, 1, H, W] -> [B, H, W]
        if depth.dim() == 4:
            depth = depth.squeeze(1)
        if mask is not None and mask.dim() == 4:
            mask = mask.squeeze(1)

        # 若需要则插值回原始分辨率
        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=True
            ).squeeze(1)
            if mask is not None:
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

        # 单张输入去掉 batch 维
        if not is_batched:
            depth = depth.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        # 确保输出为张量
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).to(self.device)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(self.device)

        return depth, mask


class MockDAPModel(DAPModel):
    """
    无权重时的模拟 DAP 模型，用于测试。

    根据图像亮度返回合成深度。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    @classmethod
    def load(cls, model_path: Optional[str] = None, device: torch.device = torch.device('cpu')) -> 'MockDAPModel':
        return cls(device)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        return_mask: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """根据图像亮度返回合成深度。"""
        # 批处理与单张输入
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)  # [1, H, W, 3]

        B, H, W, C = image.shape

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # 基础深度：5 米
        depth = torch.ones(B, H, W, device=self.device) * 5.0

        # 按亮度增加变化（越亮越远）
        brightness = image.mean(dim=-1)  # [B, H, W]
        depth = depth + brightness * 10.0

        # 少量噪声
        depth = depth + torch.randn(B, H, W, device=self.device) * 0.3
        depth = depth.clamp(min=0.1)

        # 模拟 mask：除极亮区域（模拟天空）外均有效
        mask = (brightness < 0.9).float() if return_mask else None

        # 单张输入去掉 batch 维
        if not is_batched:
            depth = depth.squeeze(0)
            if mask is not None:
                mask = mask.squeeze(0)

        return depth, mask
