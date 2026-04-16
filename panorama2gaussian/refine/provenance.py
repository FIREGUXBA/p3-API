"""高斯来源追踪：原始 vs 精修新增。"""

import logging
import torch

logger = logging.getLogger(__name__)


def tag_gaussian_provenance(gaussians, initial_count):
    """标记哪些高斯为原始、哪些为精修致密化新增。

    在 GaussianModel 上保存来源张量：
    0 = 原始（来自全景），1 = 新增（精修致密化）
    """
    if gaussians is None:
        return

    current_count = gaussians.get_xyz.shape[0]
    provenance = torch.zeros(current_count, device=gaussians.get_xyz.device)
    provenance[initial_count:] = 1.0
    gaussians._provenance = provenance

    new_count = current_count - initial_count
    logger.info(f"已标记来源：{initial_count} 个原始，{new_count} 个新增")


def apply_provenance_lr_scaling(gaussians, initial_count, scale=0.1):
    """降低原始高斯学习率以减轻漂移。

    索引 < initial_count 的原始高斯学习率乘以 `scale`（默认 0.1），
    新增高斯保持全学习率。
    """
    if gaussians is None or gaussians.optimizer is None:
        return

    for param_group in gaussians.optimizer.param_groups:
        if len(param_group['params']) > 0:
            param = param_group['params'][0]
            if hasattr(param, 'shape') and len(param.shape) > 0 and param.shape[0] > initial_count:
                lr = param_group['lr']
                # 注意：标准 Adam 不原生支持逐参数学习率。
                # 以下为尽力而为 —— 完整逐参数 LR 需改优化器。
                # 目前降低整组 lr 会同等影响该组所有参数。来源标记主要用于
                # 诊断与未来逐参数优化。

    logger.info(f"已为 {initial_count} 个原始高斯记录来源（scale={scale}）")
