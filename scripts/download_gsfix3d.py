#!/usr/bin/env python3
"""
独立下载 GSFix3D 权重（与 cli.py 中 gsfix3d 使用同一 Hugging Face 仓库）。

尝试顺序：
  1) huggingface_hub.snapshot_download（推荐，支持断点与 HF_TOKEN）
  2) huggingface-cli download（若已安装 huggingface_hub 自带的 CLI）
  3) git + Git LFS 克隆（需本机已安装 git、git-lfs）

用法（在项目根目录）:
  python scripts/download_gsfix3d.py
  python scripts/download_gsfix3d.py --output pretrained/gsfix3d
  set HF_TOKEN=xxx && python scripts/download_gsfix3d.py   # 私有/需授权仓库
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "goldoak1421/gsfixer-full-replica-room1"
HF_URL = f"https://huggingface.co/{REPO_ID}"


def _repo_has_content(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if p.name in (".git", ".cache"):
            continue
        return True
    return bool((d / ".git").exists())


def download_via_hub(out: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        REPO_ID,
        local_dir=str(out),
        token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return Path(path)


def download_via_hf_cli(out: Path, token: str | None) -> None:
    """与当前 Python 环境内的 huggingface_hub 一致，避免 PATH 指到别的解释器。"""
    env = os.environ.copy()
    t = token or env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        env["HF_TOKEN"] = t

    candidates: list[list[str]] = []
    hf_cli = shutil.which("huggingface-cli")
    hf_bin = shutil.which("hf")
    if hf_cli:
        candidates.append([hf_cli, "download", REPO_ID, "--local-dir", str(out)])
    if hf_bin:
        candidates.append([hf_bin, "download", REPO_ID, "--local-dir", str(out)])
    candidates.append(
        [sys.executable, "-m", "huggingface_hub.cli", "download", REPO_ID, "--local-dir", str(out)]
    )
    candidates.append(
        [
            sys.executable,
            "-m",
            "huggingface_hub.cli.huggingface_cli",
            "download",
            REPO_ID,
            "--local-dir",
            str(out),
        ]
    )

    last_err = ""
    for cmd in candidates:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode == 0:
            return
        last_err = r.stderr or r.stdout or f"exit {r.returncode}"
    raise RuntimeError(last_err)


def download_via_git(out: Path) -> None:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("未找到 git，请安装 Git for Windows 并加入 PATH")

    out.parent.mkdir(parents=True, exist_ok=True)

    # 若目标已存在且非空，避免覆盖
    if out.exists() and _repo_has_content(out):
        raise RuntimeError(f"目标目录已存在内容: {out}，请换 --output 或清空后重试")

    if out.exists():
        shutil.rmtree(out)

    subprocess.run(
        [git, "lfs", "install"],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        [git, "clone", HF_URL, str(out)],
        check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "0"},
    )


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    default_out = root / "pretrained" / "gsfix3d"

    p = argparse.ArgumentParser(description=f"下载 GSFix3D 模型: {REPO_ID}")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_out,
        help=f"输出目录（默认: {default_out}）",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face token（也可设置环境变量 HF_TOKEN）",
    )
    p.add_argument(
        "--force-git",
        action="store_true",
        help="跳过 huggingface_hub，直接使用 git clone",
    )
    args = p.parse_args()
    out: Path = args.output.resolve()

    print(f"仓库: {REPO_ID}")
    print(f"输出: {out}")

    if not args.force_git:
        try:
            path = download_via_hub(out, args.token)
            print(f"完成（huggingface_hub）: {path}")
            return 0
        except ImportError:
            print(
                "当前环境未安装 huggingface_hub，将依次尝试 huggingface-cli、git",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"huggingface_hub 失败: {e}", file=sys.stderr)

        print("尝试 huggingface-cli / hf …", file=sys.stderr)
        try:
            download_via_hf_cli(out, args.token)
            print(f"完成（CLI）: {out}")
            return 0
        except Exception as e2:
            print(f"CLI 下载失败: {e2}", file=sys.stderr)
            print("尝试 git clone …", file=sys.stderr)

    try:
        download_via_git(out)
        print(f"完成（git）: {out}")
        return 0
    except Exception as e:
        print(f"git 克隆失败: {e}", file=sys.stderr)
        print(
            "\n可手动操作：\n"
            "  pip install -U huggingface-hub\n"
            f"  huggingface-cli download {REPO_ID} --local-dir \"{out}\"\n"
            f"  或: git clone {HF_URL} \"{out}\"\n"
            "若仓库需登录，请先: huggingface-cli login 或设置 HF_TOKEN",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
