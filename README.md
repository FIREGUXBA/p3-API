
# Panorama2Gaussian

Panorama2Gaussian 用于将单张等距柱状投影（ERP）全景图转换为 3D Gaussian Splatting（`.ply`）场景，并提供基于 **FastAPI** 的 Web UI 与 HTTP API。

系统默认优先使用 **DA360** 深度估计模型；若 DA360 不可用，则回退到 **DAP**；若两者均不可用，则回退到 mock 深度，方便无权重环境下调试流程。

本项目包含两类能力：

| 功能 | 说明 |
|---|---|
| 基础生成 | ERP 全景图 → 深度图 → 球面反投影 → 3DGS `.ply` |
| 可选精修 | 基于 GSFix3D 的空洞检测、生成式补全与 3DGS 蒸馏 |

---

## 1. 环境要求

### 基础环境

推荐环境：

- **Ubuntu 20.04 / 22.04**
- **Python 3.10 或 3.11**
- **NVIDIA GPU**
- **CUDA 11.8 / 12.1 / 12.4**
- **显存：**
  - 基础生成：建议 8GB+
  - 4K ERP 输入：建议 12GB+
  - GSFix3D 精修：建议 24GB+

不推荐使用 Python 3.12 部署精修流程，部分第三方库可能存在兼容问题。

---

## 2. 克隆项目

```bash
git clone <你的仓库地址> Panorama2Gaussian
cd Panorama2Gaussian
````

若仓库包含 submodule：

```bash
git submodule update --init --recursive
```

---

## 3. 创建虚拟环境

推荐使用 Python 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

检查 Python 版本：

```bash
python --version
```

应输出类似：

```text
Python 3.10.x
```

---

## 4. 安装 PyTorch

请根据机器 CUDA 版本选择对应 PyTorch。

### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU / macOS 调试环境

```bash
pip install torch torchvision
```

注意：无 NVIDIA GPU 时只能进行有限调试，不推荐用于完整生成或精修。

检查 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

---

## 5. 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 6. 安装上游深度模型代码

深度推理依赖 Insta360 的 DA360 / DAP 开源代码。需要将仓库克隆到固定路径。

在项目根目录执行：

```bash
# DA360：Depth Anything 360，默认优先使用
git clone https://github.com/Insta360-Research-Team/DA360 panorama2gaussian/da360_arch/DA360

# DAP：Depth Any Panoramas，作为回退方案
git clone https://github.com/Insta360-Research-Team/DAP panorama2gaussian/dap_arch/DAP
```

目录结构应类似：

```text
Panorama2Gaussian/
├── panorama2gaussian/
│   ├── da360_arch/
│   │   └── DA360/
│   ├── dap_arch/
│   │   └── DAP/
│   └── ...
├── requirements.txt
└── README.md
```

---

## 7. 下载模型权重

下载全部模型：

```bash
python -m panorama2gaussian download-models
```

只下载 DA360：

```bash
python -m panorama2gaussian download-models --model da360
```

只下载 DAP：

```bash
python -m panorama2gaussian download-models --model dap
```

校验权重：

```bash
python -m panorama2gaussian download-models --verify
```

模型权重通常会缓存在：

```text
~/.cache/panorama2gaussian/
```

或项目运行日志中指定的 `pretrained/` 路径。

---

## 8. 启动 Web UI / API

启动服务：

```bash
python -m panorama2gaussian serve --host 0.0.0.0 --port 7860
```

本地访问：

```text
http://127.0.0.1:7860
```

远程服务器访问：

```text
http://服务器IP:7860
```

交互式 API 文档：

```text
http://服务器IP:7860/docs
```

常用参数：

| 参数         | 说明                                        |
| ---------- | ----------------------------------------- |
| `--host`   | 绑定地址，本地调试可用 `127.0.0.1`，服务器部署建议 `0.0.0.0` |
| `--port`   | 服务端口，默认可使用 `7860`                         |
| `--reload` | 开发模式自动重载                                  |

开发模式示例：

```bash
python -m panorama2gaussian serve --host 127.0.0.1 --port 7860 --reload
```

---

## 9. 命令行使用

查看主命令帮助：

```bash
python -m panorama2gaussian --help
```

查看模型下载命令：

```bash
python -m panorama2gaussian download-models --help
```

查看服务命令：

```bash
python -m panorama2gaussian serve --help
```

---

## 10. 可选：GSFix3D 精修部署

GSFix3D 精修用于对初始 3DGS 场景进行空洞检测、生成式补全和蒸馏优化。该功能依赖更多第三方库和较大显存。

### 10.1 安装精修依赖

```bash
pip install -r requirements-refine.txt
```

建议额外安装：

```bash
pip install open3d trimesh "pyglet<2" pillow scipy fast-simplification
```

### 10.2 下载 GSFix3D 权重

```bash
python -m panorama2gaussian download-models --model gsfix3d
```

权重目录通常为：

```text
pretrained/gsfix3d/
```

### 10.3 检查 GSFix3D 目录

若项目通过 submodule 引入 GSFix3D，请确认目录存在：

```text
third_party/GSFix3D/
```

若不存在，执行：

```bash
git submodule update --init --recursive
```

---

## 11. AutoDL / Linux 服务器部署注意事项

在无显示器的服务器环境中，GSFix3D 精修的 mesh condition 渲染可能依赖 OpenGL、GLU、Mesa、Xvfb 等组件。缺失这些组件会导致 mesh 渲染失败，进而让 GSFixer 使用灰色占位图作为条件输入，最终造成空洞无法被正确修复。

### 11.1 安装系统 OpenGL 依赖

Ubuntu / Debian 系统执行：

```bash
apt-get update
apt-get install -y \
  libglu1-mesa \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  libegl1-mesa \
  libosmesa6 \
  mesa-utils \
  xvfb
```

检查 GL / GLU / EGL / OSMesa 是否可用：

```bash
python - <<'PY'
import ctypes.util
print("GL :", ctypes.util.find_library("GL"))
print("GLU:", ctypes.util.find_library("GLU"))
print("EGL:", ctypes.util.find_library("EGL"))
print("OSMesa:", ctypes.util.find_library("OSMesa"))
PY
```

正常应看到类似：

```text
GL : libGL.so.1
GLU: libGLU.so.1
EGL: libEGL.so.1
OSMesa: libOSMesa.so.8
```

---

### 11.2 检查 Mesa 软件渲染

```bash
find /usr -name "swrast_dri.so" 2>/dev/null
```

正常应存在：

```text
/usr/lib/x86_64-linux-gnu/dri/swrast_dri.so
```

若 `/usr/lib/dri/swrast_dri.so` 不存在，可建立软链接：

```bash
mkdir -p /usr/lib/dri
ln -sf /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
```

测试 Xvfb + Mesa：

```bash
LIBGL_ALWAYS_SOFTWARE=1 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
glxinfo | grep -E "OpenGL vendor|OpenGL renderer|OpenGL version"
```

正常输出应类似：

```text
OpenGL vendor string: Mesa
OpenGL renderer string: llvmpipe
OpenGL version string: 4.5 ...
```

---

### 11.3 解决 `GLIBCXX_3.4.30 not found`

在某些 AutoDL / Conda 环境中，Python 进程可能优先加载 `/root/miniconda3/lib/libstdc++.so.6`，导致 Mesa 的 `libLLVM` 加载失败，报错类似：

```text
/root/miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
required by /usr/lib/x86_64-linux-gnu/libLLVM-15.so.1
```

此时可强制使用系统 `libstdc++`：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

检查系统库是否包含 `GLIBCXX_3.4.30`：

```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

---

### 11.4 推荐的精修启动方式

在服务器或 AutoDL 环境中，建议使用如下方式启动带精修的任务：

```bash
cd /root/p3-API
source .venv/bin/activate

unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
unset DISPLAY

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export MESA_LOADER_DRIVER_OVERRIDE=swrast

xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python -m panorama2gaussian serve --host 0.0.0.0 --port 7860
```

也可以将以上内容保存为脚本：

```bash
cat > scripts/run_server_xvfb.sh <<'SH'
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
unset DISPLAY

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export MESA_LOADER_DRIVER_OVERRIDE=swrast

xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python -m panorama2gaussian serve --host 0.0.0.0 --port 7860
SH

chmod +x scripts/run_server_xvfb.sh
```

之后启动：

```bash
bash scripts/run_server_xvfb.sh
```

---

## 12. 本地测试 mesh condition 渲染

精修失败或空洞修复图变灰时，建议先单独测试 mesh 渲染。

创建测试脚本：

```bash
cat > scripts/test_mesh_render.py <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from panorama2gaussian.refine.mesh_extract import extract_conditioning_mesh, render_mesh

depth_path = "/root/p3-API/output/erp_4k_da360_stride1/erp_4k_depth.npy"
pano_path = "/root/p3-API/test-data/erp_4k.png"
out_path = "/root/p3-API/output/mesh_render_test.png"

depth = np.load(depth_path)
pano = Image.open(pano_path).convert("RGB")
pano = np.asarray(pano).astype(np.float32) / 255.0

print("[test] depth:", depth.shape, depth.min(), depth.max())
print("[test] pano :", pano.shape, pano.min(), pano.max())

mesh = extract_conditioning_mesh(
    depth_map=depth,
    panorama=pano,
    simplify_ratio=0.1,
)

print("[test] mesh:", mesh)

img = render_mesh(
    mesh=mesh,
    camera=None,
    resolution=(512, 512),
)

print("[test] render:", img.shape, img.min(), img.max(), img.mean(), img.std())

Image.fromarray((img * 255).clip(0, 255).astype(np.uint8)).save(out_path)
print("[test] saved:", out_path)
PY
```

运行：

```bash
cd /root/p3-API
source .venv/bin/activate

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export MESA_LOADER_DRIVER_OVERRIDE=swrast

xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python scripts/test_mesh_render.py
```

若输出类似：

```text
[test] render: (512, 512, 3) 0.5 0.5 0.5 0.0
```

说明渲染仍然失败并返回了灰色占位图。

正常情况下，`std` 应明显大于 0，例如：

```text
[test] render: (512, 512, 3) 0.0 1.0 0.xxx 0.xxx
```

---

## 13. 重要建议：禁止灰色 mesh fallback

调试 GSFix3D 精修时，不建议在 mesh 渲染失败后继续返回灰色图。灰色 mesh condition 会导致 GSFixer 输出灰色空洞伪真值，从而使后续蒸馏无法真正补洞。

建议将 `panorama2gaussian/refine/mesh_extract.py` 中的 fallback：

```python
except Exception as e:
    logger.warning(f"网格渲染失败（{e}），使用灰色占位图")
    return np.ones((*resolution, 3), dtype=np.float32) * 0.5
```

改为：

```python
except Exception as e:
    logger.exception(f"网格渲染失败：{e}")
    raise RuntimeError(
        "mesh condition 渲染失败，已中止。不要继续使用灰色占位图，"
        "否则 GSFixer 会生成灰色空洞伪真值。"
    ) from e
```

这样可以避免流程表面成功、结果实际错误的问题。

---

## 14. 输出与缓存

默认输出目录：

```text
output/
```

常见内容：

```text
output/
├── jobs/
├── xxx_depth.npy
├── xxx.ply
├── xxx_refined.ply
└── diagnostics/
```

模型缓存可能位于：

```text
~/.cache/panorama2gaussian/
pretrained/
```

输出文件和缓存文件通常较大，默认不应提交到 Git。

---

## 15. 常见问题

### Q1：启动后提示找不到 DA360 / DAP

确认是否已经克隆：

```bash
ls panorama2gaussian/da360_arch/DA360
ls panorama2gaussian/dap_arch/DAP
```

缺失时重新执行：

```bash
git clone https://github.com/Insta360-Research-Team/DA360 panorama2gaussian/da360_arch/DA360
git clone https://github.com/Insta360-Research-Team/DAP panorama2gaussian/dap_arch/DAP
```

---

### Q2：CUDA 不可用

检查：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

若输出 `False`，检查：

```bash
nvidia-smi
```

并确认安装的是 CUDA 版本的 PyTorch，而不是 CPU 版本。

---

### Q3：精修时出现 `Library "GLU" not found`

安装系统依赖：

```bash
apt-get update
apt-get install -y libglu1-mesa libgl1-mesa-glx libgl1-mesa-dri
```

---

### Q4：精修时出现 `Cannot connect to "None"`

说明当前环境没有显示器。使用 `xvfb-run` 启动：

```bash
xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python -m panorama2gaussian serve --host 0.0.0.0 --port 7860
```

---

### Q5：精修时出现 `Could not create GL context`

先测试 Mesa：

```bash
LIBGL_ALWAYS_SOFTWARE=1 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
xvfb-run -a -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
glxinfo | grep -E "OpenGL vendor|OpenGL renderer|OpenGL version"
```

若 Python 中仍失败，并出现 `GLIBCXX_3.4.30 not found`，加入：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

---

### Q6：精修输出的修复伪真值图中空洞仍是灰色

优先检查 mesh condition 是否渲染失败。
若日志中出现：

```text
网格渲染失败，使用灰色占位图
```

说明 GSFixer 实际拿到的是灰色 mesh 条件图，需要先解决 OpenGL / Xvfb / libstdc++ 问题，再重新运行 refine。

---

## 16. 许可证与致谢

* DA360、DAP、GSFix3D 等第三方模型与代码版权归原作者及对应仓库所有。
* 使用前请遵守各依赖、模型权重和数据集的许可证要求。
* 本项目仅整合全景深度估计、3D Gaussian Splatting 初始化与可选精修流程，相关算法基础来自对应开源项目与论文。

