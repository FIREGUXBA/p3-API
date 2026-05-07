# Panorama2Gaussian

将等距柱状（ERP）全景图转换为 3D Gaussian Splatting（`.ply`），并提供基于 **FastAPI** 的 Web UI 与 HTTP API。默认优先加载 **DA360** 深度模型，不可用时回退 **DAP**，再回退为 mock 深度（便于无权重调试）。

## 环境要求

- **Python** 3.10+（建议 3.10 或 3.11）
- **NVIDIA GPU** + **CUDA**：推理与训练相关流程依赖 PyTorch CUDA 构建；无独显时仅能有限运行或走 mock，不推荐生产使用
- 磁盘空间：依赖项 + 权重（DAP/DA360 等量级约数 GB，精修与缓存另计）

## 快速开始

### 1. 虚拟环境与 PyTorch

在项目根目录执行：

```bash
python -m venv .venv
```

激活环境：

- **Linux / macOS：** `source .venv/bin/activate`
- **Windows（cmd）：** `.venv\Scripts\activate.bat`
- **Windows（PowerShell）：** `.venv\Scripts\Activate.ps1`

安装 PyTorch（若本机 CUDA 版本与下述不一致，请到 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 自选）：

- **Windows / Linux（NVIDIA GPU，CUDA 12.1 / `cu121` wheel）：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- **macOS：** 无 NVIDIA CUDA 官方 wheel，请用默认索引安装（Apple Silicon 上可用 **MPS** 加速；Intel Mac 多为 CPU）：

```bash
pip install torch torchvision
```

再安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 上游模型代码（DA360 / DAP）

深度推理需要把 **Insta360** 开源仓库放到固定路径（与 `panorama2gaussian` 包内导入一致）。在项目根目录执行：

```bash
# DA360（Depth Anything 360，服务端默认优先使用）
git clone https://github.com/Insta360-Research-Team/DA360 panorama2gaussian/da360_arch/DA360

# DAP（Depth Any Panoramas，作为回退）
git clone https://github.com/Insta360-Research-Team/DAP panorama2gaussian/dap_arch/DAP
```

若仓库中配置了其它 **git submodule**（例如第三方精修相关），可在克隆本仓库后执行：

```bash
git submodule update --init --recursive
```

（具体以当前 `.gitmodules` 为准；**DAP/DA360 仍以手动克隆到上述目录为主**。）

### 3. 下载权重

下载 DAP、DA360 等预训练权重（缓存目录通常在用户目录下的 `.cache/panorama2gaussian`）：

```bash
python -m panorama2gaussian download-models
```

只下载某一项：

```bash
python -m panorama2gaussian download-models --model da360
python -m panorama2gaussian download-models --model dap
```

校验已下载文件（若支持）：

```bash
python -m panorama2gaussian download-models --verify
```

### 4. 启动 Web UI / API

```bash
python -m panorama2gaussian serve --port 7860
```

默认监听 `http://127.0.0.1:7860`。常用参数：

- `--host`：绑定地址，默认 `127.0.0.1`
- `--reload`：开发时自动重载

浏览器访问根路径即可使用前端；交互式 API 文档一般为 **`/docs`**（Swagger）。

## 可选：GSFix3D 精修

精修管线（Web UI 中的 refinement / `/api/refine`，或下文的本地脚本）依赖更多包与大显存，请见 `requirements-refine.txt` 内说明（约 **24GB VRAM** 等提示）。

```bash
# 1) 拉取 GSFix3D 源码（提供 marigold / MarigoldGSFixerPipeline 与 gs 渲染器）
git clone https://github.com/GSFix3D/GSFix3D.git third_party/GSFix3D

# 2) 安装精修管线依赖
pip install -r requirements-refine.txt

# 3) 下载 GSFixer 权重
python -m panorama2gaussian download-models --model gsfix3d
```

说明：

- 仓库 `.gitmodules` 中登记了 `third_party/GSFix3D` 这个子模块，但目前索引并未提交对应的 gitlink，所以 `git submodule update` 不会生效。**请直接按上面 `git clone` 到 `third_party/GSFix3D`**。
- GSFix3D 仓库自带 `diff-gaussian-rasterization`、`gs/`、`marigold/` 等，需要按其 `requirements.txt` 再补装对应依赖（如 `gsplat`、`diffusers`、`transformers` 等，`requirements-refine.txt` 已涵盖大部分）。
- 若仍需旧文档中的额外工具库，可按需安装，例如 `open3d`、`trimesh`、`scipy`。

## 本地文件夹脚本：`scripts/local_pano2gs.py`

如果只是本地跑一张全景，不想起 Web 服务，可以直接用脚本，输入输出都在本地目录。支持两种模式：

```bash
# 仅转换：ERP → PLY，同时保存 <stem>_depth.npy 供精修复用
python scripts/local_pano2gs.py -i /path/to/pano.jpg -o /path/to/out_dir

# 可选：保存深度可视化 JPEG
python scripts/local_pano2gs.py -i /path/to/pano.jpg -o /path/to/out_dir --depth-preview

# 转换 + 精修（需完成上一节 GSFix3D 安装）
python scripts/local_pano2gs.py -i /path/to/pano.jpg -o /path/to/out_dir --refine

# 仅精修：复用上一次生成的 PLY + 深度 npy
python scripts/local_pano2gs.py -o /path/to/out_dir --refine-only \
  --ply /path/to/out_dir/pano.ply \
  --panorama /path/to/pano.jpg \
  --depth-npy /path/to/out_dir/pano_depth.npy
```

常用参数与 `panorama2gaussian.core.convert` 对齐：`--stride`、`--depth-min`、`--depth-max`、`--sky-threshold`、`--outlier-pruning`、`--grazing-angle`、`--sparse-pruning`、`--global-scale`、`--force-erp`、`--depth-model {da360,dap}` 等；精修侧与 `/api/refine` 对齐：`--max-rounds`、`--num-cameras`、`--finetune-steps`、`--diagnostics-dir`。

脚本会把仓库根目录加入 `sys.path` 并自动配置 `logging`，所以请在 **仓库根目录** 下运行；`--refine` 期间全部进度会以 `[refine] iter=... stage=... pct=...` 形式实时打印，便于观察具体卡在哪一阶段。

## 输出与缓存

- 任务输出默认在仓库根目录下的 **`output/`**（含 `jobs/` 等），`.gitignore` 中通常已忽略，避免将大文件提交到 Git。
- 模型权重缓存、Hugging Face 下载等可能在用户目录或项目下的 `pretrained/` 等路径，以运行日志为准。

## 命令行帮助

```bash
python -m panorama2gaussian --help
python -m panorama2gaussian download-models --help
python -m panorama2gaussian serve --help
```

## 许可与致谢

- 深度模型与上游代码版权归各自仓库（如 [DA360](https://github.com/Insta360-Research-Team/DA360)、[DAP](https://github.com/Insta360-Research-Team/DAP)）及对应许可协议所有。
- 使用前请遵守各依赖与权重的许可证要求。
