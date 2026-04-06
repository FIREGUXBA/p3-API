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

安装 **CUDA 12.1** 对应的 PyTorch（与 `cu121`  wheel 一致；若你本机 CUDA 版本不同，请到 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择对应命令）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
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

精修管线（Web UI 中的 refinement / `/api/refine`）依赖更多包与大显存，请见 `requirements-refine.txt` 内说明（约 **24GB VRAM** 等提示）。

```bash
pip install -r requirements-refine.txt
python -m panorama2gaussian download-models --model gsfix3d
```

若仍需旧文档中的额外工具库，可按需安装，例如：`open3d`、`trimesh`、`scipy`（部分功能或脚本可能用到）。

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
