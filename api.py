# api.py
"""
Panorama2Gaussian 的 FastAPI HTTP API。
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
import shutil
import subprocess
import urllib.parse
import urllib.request
from typing import Optional
from contextlib import asynccontextmanager

# 配置日志，使精修流水线消息可见
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
)
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from panorama2gaussian import Panorama2Gaussian, ConversionResult

from minio_service import MinIOService


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────
# 输出保存在项目目录（临时清理与重启后仍可保留）
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
TEMP_DIR = OUTPUT_ROOT / "jobs"
JOB_TTL_SECONDS = 30 * 60  # 30 分钟
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
GPU_SEMAPHORE_LIMIT = 1


def _normalize_minio_object_key(key: str) -> str:
    k = key.strip().replace("\\", "/").lstrip("/")
    if not k or ".." in k:
        raise HTTPException(400, "非法的 MinIO 对象键")
    return k


def _suffix_from_name(name: str) -> str:
    s = Path(name).suffix.lower()
    if s in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        return s
    return ".jpg"


async def _read_upload_chunks(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"文件过大。最大：{MAX_UPLOAD_SIZE // 1024 // 1024}MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _download_http_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Panorama2Gaussian/1.0"},
        method="GET",
    )
    chunks = []
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                raise HTTPException(400, f"文件过大。最大：{MAX_UPLOAD_SIZE // 1024 // 1024}MB")
            chunks.append(chunk)
    return b"".join(chunks)


def _load_bytes_from_minio_object(object_name: str) -> bytes:
    data = MinIOService.get_file_bytes(object_name)
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"文件过大。最大：{MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    return data


# ─────────────────────────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────────────────────────
processor: Optional[Panorama2Gaussian] = None
gpu_semaphore: Optional[asyncio.Semaphore] = None
jobs: dict = {}  # job_id -> JobInfo


class JobInfo:
    """跟踪一次转换任务。"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "queued"
        self.created_at = time.time()
        self.last_updated = time.time()
        self.input_path: Optional[Path] = None
        self.output_ply_path: Optional[Path] = None
        self.depth_preview_path: Optional[Path] = None
        self.depth_npy_path: Optional[Path] = None
        self.result: Optional[ConversionResult] = None
        self.error: Optional[str] = None
        self.params: dict = {}
        self.minio_object_name: Optional[str] = None
        self.minio_upload_error: Optional[str] = None
        self.minio_depth_preview_object: Optional[str] = None
        self.minio_depth_npy_object: Optional[str] = None
        self.minio_depth_upload_error: Optional[str] = None


class RefineJobInfo:
    """跟踪一次精修任务。"""

    def __init__(self, refine_id: str, source_job_id: str):
        self.refine_id = refine_id
        self.source_job_id = source_job_id
        self.status = "queued"
        self.created_at = time.time()
        self.last_updated = time.time()
        self.round_number = 0
        self.stage = ""
        self.progress_pct = 0
        self.output_ply_path: Optional[Path] = None
        self.diagnostics_dir: Optional[Path] = None
        self.metrics: dict = {}
        self.error: Optional[str] = None
        self.params: dict = {}


refine_jobs: dict = {}  # refine_id -> RefineJobInfo


# ─────────────────────────────────────────────────────────────────
# 生命周期管理
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭。"""

    global processor, gpu_semaphore

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 优先初始化 DA360（默认），失败则尝试 DAP，再失败则使用 mock
    try:
        processor = Panorama2Gaussian(device="cuda", depth_model="da360")
        print("已加载 DA360 深度模型")
    except Exception as e:
        print(f"DA360 不可用（{e}），正在尝试 DAP...")
        try:
            processor = Panorama2Gaussian(device="cuda", depth_model="dap")
            print("已加载 DAP 深度模型")
        except Exception as e2:
            print(f"DAP 不可用（{e2}），使用 mock 深度")
            processor = Panorama2Gaussian(device="cuda", use_mock_dap=True)

    gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_LIMIT)
    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    await run_cleanup()


async def cleanup_loop():
    """定期清理过期任务与临时文件。"""

    while True:
        try:
            await asyncio.sleep(60)
            await run_cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"清理出错：{e}")
            await asyncio.sleep(60)


async def run_cleanup():
    """移除过期任务及其文件。"""

    now = time.time()

    expired_jobs = [
        job_id for job_id, job in jobs.items()
        if job.status in ("complete", "error")
        and now - job.last_updated > JOB_TTL_SECONDS
    ]

    for job_id in expired_jobs:
        job = jobs.pop(job_id, None)
        if job:
            for path in [job.input_path, job.output_ply_path, job.depth_preview_path, job.depth_npy_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

    # 清理临时目录中的孤立文件
    active_paths = set()
    for j in jobs.values():
        if j.status in ("queued", "processing"):
            for p in [j.input_path, j.output_ply_path, j.depth_preview_path, j.depth_npy_path]:
                if p:
                    active_paths.add(str(p))

    try:
        for f in TEMP_DIR.iterdir():
            if str(f) in active_paths:
                continue
            if now - f.stat().st_mtime > JOB_TTL_SECONDS:
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink()
    except Exception:
        pass


app = FastAPI(title="Panorama2Gaussian", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────
# COOP/COEP 中间件（SharedArrayBuffer 所需）
# ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_coop_coep(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


# ─────────────────────────────────────────────────────────────────
# API 端点
# ─────────────────────────────────────────────────────────────────
@app.api_route("/api/convert", methods=["GET", "POST"])
async def convert_panorama(
    file: Optional[UploadFile] = File(None),
    minio_object: Optional[str] = Query(
        None,
        description="MinIO 桶内对象键（如 input/pano.jpg），与上传文件、source_url 三选一",
    ),
    source_url: Optional[str] = Query(
        None,
        description="图片的 http(s) URL（含 MinIO 预签名链接），与上传文件、minio_object 三选一",
    ),
    depth_model: str = Query("da360", pattern="^(dap|da360)$"),
    stride: int = Query(2, ge=1, le=8),
    depth_min: Optional[float] = Query(None, ge=0.01),
    depth_max: Optional[float] = Query(None, le=1000.0),
    sky_threshold: Optional[float] = Query(None),
    outlier_pruning: float = Query(0.0, ge=0.0, le=1.0),
    grazing_angle: float = Query(90.0, ge=30.0, le=90.0),
    sparse_pruning: float = Query(0.0, ge=0.0, le=1.0),
    global_scale: float = Query(1.0, ge=0.1, le=10.0),
):
    """将上传的全景图，或 MinIO/URL 指向的图片，转换为高斯泼溅 PLY。

    GET 与 POST 均支持；GET 仅适用于查询参数 minio_object 或 source_url（不能 multipart 上传）。"""

    if depth_min is not None and depth_max is not None and depth_min >= depth_max:
        raise HTTPException(400, f"depth_min（{depth_min}）必须小于 depth_max（{depth_max}）。")

    mo = (minio_object or "").strip()
    su = (source_url or "").strip()
    has_file = file is not None

    if mo and (su or has_file):
        raise HTTPException(400, "minio_object 不能与 source_url 或 file 同时使用")
    if su and has_file:
        raise HTTPException(400, "source_url 不能与 file 同时使用")

    if mo:
        key = _normalize_minio_object_key(mo)
        try:
            content = await run_in_threadpool(_load_bytes_from_minio_object, key)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("从 MinIO 读取输入图失败")
            raise HTTPException(502, f"从 MinIO 读取失败：{e}") from e
        suffix = _suffix_from_name(key)
        input_source = "minio"
        input_ref = key
    elif su:
        parsed = urllib.parse.urlparse(su)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(400, "source_url 须为 http 或 https")
        try:
            content = await run_in_threadpool(_download_http_bytes, su)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("从 URL 下载输入图失败")
            raise HTTPException(502, f"从 URL 下载失败：{e}") from e
        suffix = _suffix_from_name(urllib.parse.unquote(parsed.path) or "")
        input_source = "url"
        input_ref = su
    elif has_file:
        content = await _read_upload_chunks(file)
        suffix = _suffix_from_name(file.filename or "image.jpg")
        input_source = "upload"
        input_ref = file.filename or ""
    else:
        raise HTTPException(
            400,
            "请提供以下之一：multipart 上传 file，或查询参数 minio_object（桶内路径），或 source_url（图片 URL）",
        )

    job_id = str(uuid.uuid4())
    job = JobInfo(job_id)
    jobs[job_id] = job

    job.params = {
        "depth_model": depth_model,
        "stride": stride,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "sky_threshold": sky_threshold,
        "outlier_pruning": outlier_pruning,
        "grazing_angle": grazing_angle,
        "sparse_pruning": sparse_pruning,
        "global_scale": global_scale,
        "input_source": input_source,
        "input_ref": input_ref,
    }

    # suffix 已在上面分支赋值；统一保证带点的扩展名
    if not suffix.startswith("."):
        suffix = "." + suffix
    job.input_path = TEMP_DIR / f"{job_id}_input{suffix}"
    job.output_ply_path = TEMP_DIR / f"{job_id}_output.ply"
    job.depth_preview_path = TEMP_DIR / f"{job_id}_depth.jpg"
    job.depth_npy_path = TEMP_DIR / f"{job_id}_depth.npy"

    with open(job.input_path, "wb") as f:
        f.write(content)

    asyncio.create_task(process_job(
        job,
        depth_model=depth_model,
        stride=stride,
        depth_min=depth_min,
        depth_max=depth_max,
        sky_threshold=sky_threshold,
        outlier_pruning=outlier_pruning,
        grazing_angle=grazing_angle,
        sparse_pruning=sparse_pruning,
        global_scale=global_scale,
    ))

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "queue_position": sum(1 for j in jobs.values() if j.status == "queued"),
    })


async def process_job(
    job: JobInfo,
    depth_model: str = "dap",
    stride: int = 2,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    sky_threshold: float = 80.0,
    outlier_pruning: float = 0.0,
    grazing_angle: float = 90.0,
    sparse_pruning: float = 0.0,
    global_scale: float = 1.0,
):
    """在 GPU 信号量控制下处理转换任务。"""

    try:
        job.status = "queued"
        async with gpu_semaphore:
            job.status = "processing"
            job.last_updated = time.time()

            result = await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.output_ply_path),
                depth_min=depth_min,
                depth_max=depth_max,
                sky_threshold=sky_threshold,
                stride=stride,
                outlier_pruning=outlier_pruning,
                grazing_angle=grazing_angle,
                sparse_pruning=sparse_pruning,
                global_scale=global_scale,
                depth_model=depth_model,
                depth_preview_path=str(job.depth_preview_path),
                depth_npy_path=str(job.depth_npy_path),
            )

            job.result = result

            # 须在标记 complete 之前完成上传，否则轮询会在 minio_object 写入前看到 complete
            if job.output_ply_path and job.output_ply_path.exists():
                object_name = f"output/{job.job_id}.ply"
                try:
                    await run_in_threadpool(
                        MinIOService.upload_file,
                        str(job.output_ply_path),
                        object_name,
                    )
                    job.minio_object_name = object_name
                    try:
                        job.output_ply_path.unlink()
                    except OSError as oe:
                        logger.warning("删除本地 PLY 失败：%s", oe)
                except Exception as me:
                    job.minio_upload_error = str(me)
                    logger.warning("MinIO 上传 PLY 失败：%s", me)

            # 深度预览与深度数组同步写入 MinIO（保留本地文件供 /api/refine 使用）
            depth_err_parts: list = []
            if job.depth_preview_path and job.depth_preview_path.exists():
                preview_key = f"output/{job.job_id}_depth.jpg"
                try:
                    await run_in_threadpool(
                        MinIOService.upload_file,
                        str(job.depth_preview_path),
                        preview_key,
                    )
                    job.minio_depth_preview_object = preview_key
                except Exception as me:
                    depth_err_parts.append(f"depth_preview:{me}")
                    logger.warning("MinIO 上传深度预览失败：%s", me)
            if job.depth_npy_path and job.depth_npy_path.exists():
                npy_key = f"output/{job.job_id}_depth.npy"
                try:
                    await run_in_threadpool(
                        MinIOService.upload_file,
                        str(job.depth_npy_path),
                        npy_key,
                    )
                    job.minio_depth_npy_object = npy_key
                except Exception as me:
                    depth_err_parts.append(f"depth_npy:{me}")
                    logger.warning("MinIO 上传深度 npy 失败：%s", me)
            if depth_err_parts:
                job.minio_depth_upload_error = "; ".join(depth_err_parts)

            job.status = "complete"
            job.last_updated = time.time()

            # 保留输入全景图供后续精修（随任务过期一并清理）

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = "error"
        job.error = str(e)
        print(f"任务失败，错误：{e}")
        job.last_updated = time.time()


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """获取任务状态与结果。"""

    if job_id not in jobs:
        raise HTTPException(404, "未找到任务")

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job.status,
    }

    if job.status == "queued":
        response["queue_position"] = sum(
            1 for j in jobs.values()
            if j.status == "queued" and j.created_at < job.created_at
        ) + 1

    if job.status == "complete" and job.result:
        response["splat_count"] = job.result.splat_count
        response["file_size_mb"] = round(job.result.file_size / 1024 / 1024, 2)
        response["processing_time"] = round(job.result.processing_time, 2)
        if job.output_ply_path and job.output_ply_path.exists():
            response["ply_url"] = f"/api/download/{job_id}"
        if job.depth_preview_path and job.depth_preview_path.exists():
            response["depth_preview_url"] = f"/api/depth_preview/{job_id}"
        # 精修就绪条件：PLY（本地或已在 MinIO）、全景图与深度
        has_ply = (job.output_ply_path and job.output_ply_path.exists()) or bool(
            job.minio_object_name
        )
        has_pano = job.input_path and job.input_path.exists()
        has_depth = job.depth_npy_path and job.depth_npy_path.exists()
        response["refineable"] = bool(has_ply and has_pano and has_depth)
        if job.minio_object_name:
            response["minio_object"] = job.minio_object_name
            try:
                response["minio_presigned_url"] = MinIOService.get_presigned_url(
                    job.minio_object_name
                )
            except Exception:
                pass
        if job.minio_upload_error:
            response["minio_upload_error"] = job.minio_upload_error
        if job.minio_depth_preview_object:
            response["minio_depth_preview_object"] = job.minio_depth_preview_object
            try:
                response["minio_depth_preview_presigned_url"] = (
                    MinIOService.get_presigned_url(job.minio_depth_preview_object)
                )
            except Exception:
                pass
        if job.minio_depth_npy_object:
            response["minio_depth_npy_object"] = job.minio_depth_npy_object
            try:
                response["minio_depth_npy_presigned_url"] = MinIOService.get_presigned_url(
                    job.minio_depth_npy_object
                )
            except Exception:
                pass
        if job.minio_depth_upload_error:
            response["minio_depth_upload_error"] = job.minio_depth_upload_error

    if job.status == "error":
        response["error"] = job.error

    if job.params:
        response["params"] = job.params

    return JSONResponse(response)


@app.get("/api/depth_preview/{job_id}")
async def get_depth_preview(job_id: str):
    """获取深度图预览图像。"""

    if job_id not in jobs:
        raise HTTPException(404, "未找到任务")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "任务未完成")

    if not job.depth_preview_path or not job.depth_preview_path.exists():
        raise HTTPException(404, "暂无深度预览")

    return FileResponse(
        job.depth_preview_path,
        media_type="image/jpeg",
        filename=f"depth_{job_id[:8]}.jpg"
    )


@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """下载生成的 PLY 文件。"""

    if job_id not in jobs:
        raise HTTPException(404, "未找到任务")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "任务未完成")

    if not job.output_ply_path or not job.output_ply_path.exists():
        raise HTTPException(404, "未找到文件")

    return FileResponse(
        job.output_ply_path,
        media_type="application/octet-stream",
        filename=f"panorama2gaussian_{job_id[:8]}.ply"
    )


# ─────────────────────────────────────────────────────────────────
# 精修端点
# ─────────────────────────────────────────────────────────────────
@app.post("/api/refine")
async def start_refinement(
    job_id: str = Query(..., description="源转换任务 ID"),
    max_rounds: int = Query(3, ge=1, le=5),
    num_cameras: int = Query(36, ge=6, le=72),
    finetune_steps: int = Query(500, ge=100, le=2000),
):
    """在已有转换任务上启动 GSFix3D 精修。"""

    if job_id not in jobs:
        raise HTTPException(404, "未找到源任务")

    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "源任务未完成")

    has_local_ply = job.output_ply_path and job.output_ply_path.exists()
    if not has_local_ply and not job.minio_object_name:
        raise HTTPException(400, "未找到 PLY 文件（本地不存在且无 MinIO 对象）")
    if not (job.input_path and job.input_path.exists()):
        raise HTTPException(400, "未找到输入全景图（可能已被清理）")
    if not (job.depth_npy_path and job.depth_npy_path.exists()):
        raise HTTPException(400, "未找到深度图")

    refine_id = str(uuid.uuid4())
    refine_job = RefineJobInfo(refine_id, job_id)
    refine_job.params = {
        "max_rounds": max_rounds,
        "num_cameras": num_cameras,
        "finetune_steps": finetune_steps,
    }
    refine_job.output_ply_path = TEMP_DIR / f"{refine_id}_refined.ply"
    refine_job.diagnostics_dir = TEMP_DIR / f"{refine_id}_diagnostics"
    refine_jobs[refine_id] = refine_job

    asyncio.create_task(process_refinement(refine_job, job))

    return JSONResponse({
        "refine_job_id": refine_id,
        "status": "queued",
    })


async def process_refinement(refine_job: RefineJobInfo, source_job: JobInfo):
    """在后台运行精修流水线。"""

    try:
        async with gpu_semaphore:
            refine_job.status = "processing"
            refine_job.last_updated = time.time()

            result = await run_in_threadpool(
                _run_refinement,
                source_job=source_job,
                refine_job=refine_job,
            )

            refine_job.metrics = result
            refine_job.status = "complete"
            refine_job.last_updated = time.time()

    except Exception as e:
        import traceback
        traceback.print_exc()
        refine_job.status = "error"
        refine_job.error = str(e)
        refine_job.last_updated = time.time()


def _run_refinement(source_job: JobInfo, refine_job: RefineJobInfo) -> dict:
    """执行 GSFix3D 精修流水线（阻塞，在线程中运行）。"""

    import numpy as np
    from panorama2gaussian.refine import refine_splat

    params = refine_job.params
    output_dir = refine_job.diagnostics_dir or TEMP_DIR / f"{refine_job.refine_id}_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_local_download: Optional[Path] = None
    try:
        if source_job.output_ply_path and source_job.output_ply_path.exists():
            ply_path_str = str(source_job.output_ply_path)
        elif source_job.minio_object_name:
            ply_local_download = TEMP_DIR / f"{refine_job.refine_id}_source.ply"
            MinIOService.download_file(
                source_job.minio_object_name,
                str(ply_local_download),
            )
            ply_path_str = str(ply_local_download)
        else:
            raise FileNotFoundError(
                "无可用 PLY：本地不存在且任务未关联 MinIO 对象"
            )

        depth_map = np.load(str(source_job.depth_npy_path))

        def update_progress(round_num, stage, pct):
            refine_job.round_number = round_num
            refine_job.stage = stage
            refine_job.progress_pct = pct
            refine_job.last_updated = time.time()

        result = refine_splat(
            ply_path=ply_path_str,
            panorama_path=str(source_job.input_path),
            depth_map=depth_map,
            max_iterations=params.get("max_rounds", 3),
            num_cameras=params.get("num_cameras", 36),
            finetune_steps=params.get("finetune_steps", 500),
            output_path=str(refine_job.output_ply_path),
            progress_callback=update_progress,
            diagnostics_dir=str(output_dir / "diagnostics"),
        )

        return {
            "initial_hole_fraction": result["initial_hole_fraction"],
            "final_hole_fraction": result["final_hole_fraction"],
            "final_count": result["gaussians_count"],
            "iterations_used": result["iterations_used"],
            "total_time": result["total_time"],
        }
    finally:
        if ply_local_download is not None and ply_local_download.exists():
            try:
                ply_local_download.unlink()
            except OSError:
                pass


@app.get("/api/refine/status/{refine_id}")
async def get_refine_status(refine_id: str):
    """获取精修任务状态。"""

    if refine_id not in refine_jobs:
        raise HTTPException(404, "未找到精修任务")

    rj = refine_jobs[refine_id]
    response = {
        "refine_job_id": refine_id,
        "status": rj.status,
        "round": rj.round_number,
        "stage": rj.stage,
        "progress_pct": rj.progress_pct,
    }

    if rj.diagnostics_dir and rj.diagnostics_dir.exists():
        response["diagnostics_url"] = f"/api/refine/diagnostics/{refine_id}"

    if rj.status == "complete":
        response["metrics"] = rj.metrics
        if rj.output_ply_path and rj.output_ply_path.exists():
            response["ply_url"] = f"/api/refine/download/{refine_id}"

    if rj.status == "error":
        response["error"] = rj.error

    return JSONResponse(response)


@app.get("/api/refine/download/{refine_id}")
async def download_refined_ply(refine_id: str):
    """下载精修后的 PLY 文件。"""

    if refine_id not in refine_jobs:
        raise HTTPException(404, "未找到精修任务")

    rj = refine_jobs[refine_id]
    if rj.status != "complete":
        raise HTTPException(400, "精修未完成")

    if not rj.output_ply_path or not rj.output_ply_path.exists():
        raise HTTPException(404, "未找到精修 PLY")

    return FileResponse(
        rj.output_ply_path,
        media_type="application/octet-stream",
        filename=f"refined_{refine_id[:8]}.ply",
    )


@app.get("/api/refine/diagnostics/{refine_id}")
async def get_refine_diagnostics(refine_id: str):
    """列出某次精修任务的诊断图像。"""

    if refine_id not in refine_jobs:
        raise HTTPException(404, "未找到精修任务")

    rj = refine_jobs[refine_id]
    # 诊断结果保存在 output_dir/diagnostics/ 子目录
    diag_dir = rj.diagnostics_dir / "diagnostics" if rj.diagnostics_dir else None
    if not diag_dir or not diag_dir.exists():
        # 回退到诊断根目录
        diag_dir = rj.diagnostics_dir
    if not diag_dir or not diag_dir.exists():
        raise HTTPException(404, "暂无诊断数据")

    images = sorted(
        f.name for f in diag_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    # 按相机分组供预览画廊使用
    cameras = {}
    for name in images:
        # 解析 r{轮次}_cam{索引}_{类型}.png
        parts = name.replace(".png", "").replace(".jpg", "").split("_")
        cam_key = "_".join(parts[:2]) if len(parts) >= 3 else name
        img_type = parts[2] if len(parts) >= 3 else "combined"
        if cam_key not in cameras:
            cameras[cam_key] = {}
        cameras[cam_key][img_type] = f"/api/refine/diagnostics/{refine_id}/{name}"

    return JSONResponse({
        "refine_job_id": refine_id,
        "cameras": cameras,
        "images": [
            {"name": name, "url": f"/api/refine/diagnostics/{refine_id}/{name}"}
            for name in images
        ],
    })


@app.get("/api/refine/diagnostics/{refine_id}/{filename}")
async def get_refine_diagnostic_image(refine_id: str, filename: str):
    """提供单张诊断图像。"""

    if refine_id not in refine_jobs:
        raise HTTPException(404, "未找到精修任务")

    rj = refine_jobs[refine_id]
    if not rj.diagnostics_dir:
        raise HTTPException(404, "暂无诊断数据")

    # 规范化文件名，防止路径穿越
    safe_name = Path(filename).name
    # 先在 diagnostics/ 子目录查找，再在根目录查找
    image_path = rj.diagnostics_dir / "diagnostics" / safe_name
    if not image_path.exists():
        image_path = rj.diagnostics_dir / safe_name
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(404, "未找到诊断图像")

    media_type = "image/png" if safe_name.endswith(".png") else "image/jpeg"
    return FileResponse(image_path, media_type=media_type)


@app.get("/api/refine/metrics/{refine_id}")
async def get_refine_metrics(refine_id: str):
    """获取精修指标。"""

    if refine_id not in refine_jobs:
        raise HTTPException(404, "未找到精修任务")

    rj = refine_jobs[refine_id]
    if rj.status != "complete":
        raise HTTPException(400, "精修未完成")

    return JSONResponse(rj.metrics)


@app.post("/api/shutdown")
async def shutdown_server(request: Request):
    """关闭 Panorama2Gaussian 服务（仅本机）。"""

    import os

    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "仅允许从本机关闭服务")

    async def _exit():
        await asyncio.sleep(0.5)
        os._exit(0)

    asyncio.create_task(_exit())
    return JSONResponse({"status": "shutting_down"})


@app.get("/api/health")
async def health_check():
    """健康检查端点。"""

    return {
        "status": "ok",
        "gpu_available": gpu_semaphore._value > 0 if gpu_semaphore else False,
        "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
        "queued_jobs": sum(1 for j in jobs.values() if j.status == "queued"),
    }


@app.get("/")
async def root():
    """纯 API 服务：根路径返回 OpenAPI 文档与健康检查入口。"""

    return {
        "service": "Panorama2Gaussian",
        "openapi": "/openapi.json",
        "docs": "/docs",
        "health": "/api/health",
    }


# ─────────────────────────────────────────────────────────────────
# 启动：在绑定端口前结束占用该端口的旧进程
# ─────────────────────────────────────────────────────────────────
def kill_existing_server(port: int):
    """在绑定前结束占用 *port* 的 Panorama2Gaussian 服务（若存在）。"""

    import urllib.request

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown", method="POST"
        )
        urllib.request.urlopen(req, timeout=2)
        time.sleep(1)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "LISTENING" in line and len(parts) >= 2:
                local_addr = parts[1]
                if local_addr.endswith(f":{port}"):
                    pid = parts[-1]
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                    )
    except Exception:
        pass


DEFAULT_PORT = 7860


if __name__ == "__main__":
    import argparse, uvicorn

    parser = argparse.ArgumentParser(description="Panorama2Gaussian 服务")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 表示监听本机所有网卡")
    args = parser.parse_args()

    kill_existing_server(args.port)
    uvicorn.run("api:app", host=args.host, port=args.port)
