"""精修流水线配置。"""

from dataclasses import dataclass, field


@dataclass
class RefineConfig:
    """GSFix3D 精修流水线的全部超参数。"""

    # --- 路径 ---
    gsfixer_checkpoint: str = "pretrained/gsfix3d"

    # --- 阶段 1：相机环 ---
    camera_fov: float = 60.0
    translation_fracs: tuple = (0.05, 0.15, 0.30)
    alpha_threshold: float = 0.1
    num_directions: int = 12
    render_resolution: int = 512

    # --- 阶段 1：相机筛选 ---
    min_hole_fraction: float = 0.03
    max_repair_cameras: int = 20

    # --- 阶段 2：GSFixer ---
    finetune_steps: int = 500
    finetune_lr: float = 1.0e-5
    inference_steps: int = 50
    guidance_scale: float = 7.5
    mesh_simplify_ratio: float = 0.1

    # --- 阶段 3：蒸馏 ---
    # 与 GSFix3D 参考实现一致（arguments.py OptimizationParams）
    distill_iterations: int = 3000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.005
    lr_position: float = 0.00032
    lr_feature: float = 0.0025
    lr_opacity: float = 0.025
    lr_scaling: float = 0.002
    lr_rotation: float = 0.001
    original_view_ratio: float = 0.3

    # --- 收敛 ---
    convergence_threshold: float = 0.02
    max_iterations: int = 3

    # --- 进度回调阶段名 ---
    STAGES: dict = field(default_factory=lambda: {
        "camera_rig": "生成相机",
        "mesh_extract": "提取网格",
        "finetune": "适配场景",
        "render_holes": "检测空洞",
        "gsfixer_inference": "修复空洞",
        "distill": "优化三维",
    })
