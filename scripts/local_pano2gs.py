#!/usr/bin/env python3
"""
本地文件夹：ERP 全景 -> PLY（SPAG-4D），可选 GSFix3D 精修。

在项目根目录执行（保证可 import spag4d）：
  python scripts/local_pano2gs.py -i ./in/pano.jpg -o ./out
  python scripts/local_pano2gs.py -i ./in/pano.jpg -o ./out --refine

精修依赖 requirements-refine.txt 与大显存环境，见 README。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main() -> int:
    repo_root = _ensure_repo_root_on_path()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    parser = argparse.ArgumentParser(
        description="本地全景转高斯 PLY，可选精修（refine_splat）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="输入 ERP 全景图路径（建议 2:1）",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="输出目录（将写入 PLY、深度 npy 等）",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="在转换完成后运行 GSFix3D 精修（需要已保存的 PLY + 全景 + depth.npy）",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="跳过转换，仅精修（需同时提供 --ply --panorama --depth-npy）",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        help="精修-only：粗 PLY 路径",
    )
    parser.add_argument(
        "--panorama",
        type=Path,
        help="精修-only：与转换时相同的全景图路径",
    )
    parser.add_argument(
        "--depth-npy",
        type=Path,
        help="精修-only：转换时保存的深度 .npy",
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        help="覆盖粗 PLY 输出路径（默认：<output-dir>/<stem>.ply）",
    )
    parser.add_argument(
        "--refined-ply",
        type=Path,
        help="精修 PLY 输出路径（默认：<output-dir>/<stem>_refined.ply）",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch 设备，例如 cuda 或 cpu",
    )
    parser.add_argument(
        "--depth-model",
        default="da360",
        choices=("da360", "dap"),
        help="深度估计后端",
    )
    # 默认与 API / 前端约定一致
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=100.0)
    parser.add_argument("--sky-threshold", type=float, default=80.0)
    parser.add_argument("--outlier-pruning", type=float, default=0.3)
    parser.add_argument("--grazing-angle", type=float, default=65.0)
    parser.add_argument("--sparse-pruning", type=float, default=0.3)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--force-erp", action="store_true")
    parser.add_argument("--grid-jitter", type=float, default=0.0)
    parser.add_argument(
        "--depth-preview",
        action="store_true",
        help="在输出目录保存深度可视化 JPEG",
    )
    # 精修超参（与 api /api/refine 对齐）
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--num-cameras", type=int, default=36)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="精修诊断图目录；默认 <output-dir>/refine_diagnostics",
    )

    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refine_only:
        if not all([args.ply, args.panorama, args.depth_npy]):
            parser.error("--refine-only 需要 --ply、--panorama、--depth-npy")
        ply_path = args.ply.resolve()
        pano_path = args.panorama.resolve()
        depth_npy = args.depth_npy.resolve()
        if not ply_path.is_file():
            print(f"错误：找不到 PLY：{ply_path}", file=sys.stderr)
            return 1
        if not pano_path.is_file():
            print(f"错误：找不到全景图：{pano_path}", file=sys.stderr)
            return 1
        if not depth_npy.is_file():
            print(f"错误：找不到深度文件：{depth_npy}", file=sys.stderr)
            return 1
        stem = pano_path.stem
    else:
        if not args.input:
            parser.error("转换需要 --input / -i")
        inp = args.input.resolve()
        if not inp.is_file():
            print(f"错误：找不到输入图像：{inp}", file=sys.stderr)
            return 1
        stem = inp.stem
        ply_path = (args.output_ply or (out_dir / f"{stem}.ply")).resolve()
        pano_path = inp
        depth_npy = (out_dir / f"{stem}_depth.npy").resolve()

    refined_path = (args.refined_ply or (out_dir / f"{stem}_refined.ply")).resolve()
    diag_dir = args.diagnostics_dir
    if args.refine or args.refine_only:
        diag_dir = (diag_dir or (out_dir / "refine_diagnostics")).resolve()

    if not args.refine_only:
        from spag4d import SPAG4D

        depth_preview = None
        if args.depth_preview:
            depth_preview = out_dir / f"{stem}_depth_preview.jpg"

        print(f"[convert] 输入: {pano_path}\n[convert] 输出 PLY: {ply_path}", flush=True)
        converter = SPAG4D(
            device=args.device,
            depth_model=args.depth_model,
        )
        result = converter.convert(
            input_path=pano_path,
            output_path=ply_path,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            sky_threshold=args.sky_threshold,
            stride=args.stride,
            outlier_pruning=args.outlier_pruning,
            grazing_angle=args.grazing_angle,
            sparse_pruning=args.sparse_pruning,
            global_scale=args.global_scale,
            force_erp=args.force_erp,
            depth_model=args.depth_model,
            grid_jitter=args.grid_jitter,
            depth_preview_path=depth_preview,
            depth_npy_path=depth_npy,
        )
        print(
            f"[convert] 完成 splats={result.splat_count:,} "
            f"time={result.processing_time:.1f}s -> {result.output_path}",
            flush=True,
        )

    if args.refine or args.refine_only:
        import numpy as np
        from spag4d.refine import refine_splat

        depth_map = np.load(str(depth_npy))
        print(
            f"[refine] PLY={ply_path}\n[refine] 全景={pano_path}\n"
            f"[refine] 深度 shape={depth_map.shape} -> {refined_path}",
            flush=True,
        )
        def _progress(iteration, stage, pct):
            print(f"[refine] iter={iteration} stage={stage} pct={pct}", flush=True)

        metrics = refine_splat(
            ply_path=str(ply_path),
            panorama_path=str(pano_path),
            depth_map=depth_map,
            max_iterations=args.max_rounds,
            num_cameras=args.num_cameras,
            finetune_steps=args.finetune_steps,
            output_path=str(refined_path),
            diagnostics_dir=str(diag_dir) if diag_dir else None,
            progress_callback=_progress,
        )
        print(
            f"[refine] 完成 iterations={metrics.get('iterations_used')} "
            f"gaussians={metrics.get('gaussians_count')} "
            f"time={metrics.get('total_time', 0):.1f}s -> {refined_path}",
            flush=True,
        )

    print(f"工作目录（仓库根）: {repo_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
