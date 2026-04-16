"""阶段 2：GSFixer 模型加载、微调与推理。"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# 确保可导入 GSFix3D（子模块，非 pip 包）
_gsfix3d_path = str(Path(__file__).resolve().parents[2] / "third_party" / "GSFix3D")
if _gsfix3d_path not in sys.path:
    sys.path.insert(0, _gsfix3d_path)


class GSFixerAdapter:
    """封装 GSFix3D 的 MarigoldGSFixerPipeline，供 SPAG-4D 集成。

    生命周期：``__init__`` -> ``load()`` -> ``finetune()``（可选）
               -> ``infer()`` -> ``unload()``。
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.pipe = None

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self):
        """加载预训练的 GSFixer 扩散流水线。"""
        from marigold import MarigoldGSFixerPipeline

        logger.info("正在从 %s 加载 GSFixer …", self.checkpoint_path)
        self.pipe = MarigoldGSFixerPipeline.from_pretrained(self.checkpoint_path)
        self.pipe = self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("已启用 xformers 显存高效注意力")
        except Exception:
            logger.info("未安装 xformers，使用默认注意力")

        logger.info("GSFixer 加载成功")

    # ------------------------------------------------------------------
    # 微调
    # ------------------------------------------------------------------

    def finetune(
        self,
        gs_renders: List[np.ndarray],
        gt_images: List[np.ndarray],
        mesh,
        cameras,
        train_steps: int = 500,
        learning_rate: float = 1e-5,
    ):
        """在场景相关训练对上微调 GSFixer UNet。

        使用从全景提取的立方体贴图 GT 与对应 GS 渲染作为条件输入，
        让模型学习**当前场景**的视觉风格，以便后续补全一致。

        参数:
            gs_renders: (H, W, 3) float32 列表，取值 [0, 1]。
            gt_images:  (H, W, 3) float32 列表，取值 [0, 1]。
            mesh:       trimesh.Trimesh（单输入微调未用；预留双条件）。
            cameras:    CameraPose 列表（当前未用）。
            train_steps: 梯度步数。
            learning_rate: Adam 学习率。
        """
        if self.pipe is None:
            raise RuntimeError("请先调用 load() 再 finetune()")

        logger.info("正在微调 GSFixer UNet，共 %d 步 …", train_steps)

        # 仅 UNet 可训练
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        unet = self.pipe.unet
        unet.train()
        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

        # 预编码空文本嵌入（存为 self.pipe.empty_text_embed）
        self.pipe.encode_empty_text()

        n_pairs = len(gs_renders)
        if n_pairs == 0:
            logger.warning("未提供训练对，跳过微调")
            unet.eval()
            return

        # UNet 是否支持双条件（12 输入通道）或仅单条件（8 通道）
        unet_in_ch = unet.config.in_channels
        dual_input = unet_in_ch == 12
        logger.info(
            "UNet 输入通道：%d（%s 条件）",
            unet_in_ch,
            "双" if dual_input else "单",
        )

        for step in range(train_steps):
            idx = step % n_pairs

            # (H,W,3) float32 [0,1] -> (1,3,H,W) float32 [-1,1]
            gs_tensor = self._numpy_to_latent_input(gs_renders[idx])
            gt_tensor = self._numpy_to_latent_input(gt_images[idx])

            # 编码到潜空间（确定性：用后验均值）
            with torch.no_grad():
                gs_latent = self.pipe.encode_rgb(gs_tensor)
                gt_latent = self.pipe.encode_rgb(gt_tensor)

            # 随机扩散步
            timestep = torch.randint(
                0,
                self.pipe.scheduler.config.num_train_timesteps,
                (1,),
                device=self.device,
            )

            # 前向扩散：对 GT 潜变量加噪
            noise = torch.randn_like(gt_latent)
            noisy_latent = self.pipe.scheduler.add_noise(gt_latent, noise, timestep)

            # 构建 UNet 输入，与流水线 concat 顺序一致：
            #   双条件：[mesh_latent, gs_latent, noisy_target_latent]（12 通道）
            #   单条件：[gs_latent, noisy_target_latent]（8 通道）
            if dual_input:
                # 暂用 GS 潜变量复制为 mesh 通道；日后可换为真实 mesh 渲染
                unet_input = torch.cat([gs_latent, gs_latent, noisy_latent], dim=1)
            else:
                unet_input = torch.cat([gs_latent, noisy_latent], dim=1)

            # 空文本嵌入扩展到 batch 维
            batch_text = self.pipe.empty_text_embed.repeat(
                (gs_latent.shape[0], 1, 1)
            ).to(self.device)

            # 预测噪声残差
            noise_pred = unet(
                unet_input, timestep, encoder_hidden_states=batch_text
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info(
                    "  微调步 %d/%d，loss=%.4f", step, train_steps, loss.item()
                )

        unet.eval()
        # 恢复梯度标志，便于后续正常使用 VAE/text_encoder
        self.pipe.vae.requires_grad_(True)
        self.pipe.text_encoder.requires_grad_(True)
        logger.info("微调完成")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def infer(
        self,
        gs_renders: List[np.ndarray],
        hole_masks: List[np.ndarray],
        mesh,
        cameras,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> List[np.ndarray]:
        """对有空洞的 GS 渲染运行 GSFixer 推理。

        每个视角将 GS 渲染作为主条件（可选 mesh 渲染为次条件）。
        输出与原始图软合成：非空洞像素保留，仅空洞区域替换为模型输出。

        参数:
            gs_renders: (H, W, 3) float32 列表，[0, 1]。
            hole_masks: (H, W) float32 二值掩码列表（1=空洞）。
            mesh:       用于 mesh 条件渲染的 trimesh（可为 ``None``）。
            cameras:    CameraPose 列表。
            num_steps:  DDIM 去噪步数。
            guidance_scale: 未使用（流水线无文本引导）。

        返回:
            (H, W, 3) float32 修复图列表，[0, 1]。
        """
        if self.pipe is None:
            raise RuntimeError("请先调用 load() 再 infer()")

        logger.info("正在对 %d 个视角运行 GSFixer 推理 …", len(gs_renders))
        repaired: List[np.ndarray] = []

        from .mesh_extract import render_mesh

        for i, (gs_img, mask, cam) in enumerate(
            zip(gs_renders, hole_masks, cameras)
        ):
            hole_frac = float(mask.mean())
            logger.info(
                "  修复视角 %d/%d（空洞占比：%.1f%%）",
                i + 1,
                len(gs_renders),
                hole_frac * 100,
            )

            # numpy [0,1] float -> PIL uint8
            gs_pil = Image.fromarray(
                (gs_img * 255).clip(0, 255).astype(np.uint8)
            )

            # 双条件时渲染 mesh（mesh 为 None 时退回灰色）
            mesh_img = render_mesh(
                mesh, cam, resolution=(gs_img.shape[0], gs_img.shape[1])
            )
            mesh_pil = Image.fromarray(
                (mesh_img * 255).clip(0, 255).astype(np.uint8)
            )

            try:
                output = self.pipe(
                    gs_pil,
                    condition_image2=mesh_pil if mesh is not None else None,
                    denoising_steps=num_steps,
                    processing_res=0,  # 原生分辨率
                    match_input_res=True,
                    show_progress_bar=False,
                )
                result = (
                    np.array(output.fixed_rgb).astype(np.float32) / 255.0
                )
            except Exception as e:
                logger.warning(
                    "  GSFixer 推理失败（%s），保留原图", e
                )
                result = gs_img.copy()

            # 软合成：对掩码边界羽化，避免硬接缝
            # 对二值掩码做高斯模糊得到软过渡
            from scipy.ndimage import gaussian_filter
            soft_mask = gaussian_filter(mask.astype(np.float64), sigma=5.0)
            soft_mask = np.clip(soft_mask, 0, 1).astype(np.float32)
            mask_3d = soft_mask[..., None]
            composited = gs_img * (1.0 - mask_3d) + result * mask_3d
            repaired.append(composited.astype(np.float32))

        logger.info("GSFixer 推理完成")
        return repaired

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def unload(self):
        """释放流水线占用的 GPU 显存。"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("已卸载 GSFixer，GPU 显存已释放")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _numpy_to_latent_input(self, img_np: np.ndarray) -> torch.Tensor:
        """(H, W, 3) float32 [0,1] -> 设备上 (1, 3, H, W) [-1,1]。"""
        t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        t = t.to(dtype=self.pipe.dtype, device=self.device)
        t = t * 2.0 - 1.0
        return t
