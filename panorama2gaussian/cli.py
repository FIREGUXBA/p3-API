# panorama2gaussian/cli.py
"""
Panorama2Gaussian 命令行接口：模型下载与 Web UI。
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version="3.0.0")
def main():
    """Panorama2Gaussian：下载模型权重或启动 Web UI。"""
    pass


@main.command("download-models")
@click.option(
    "--model",
    type=click.Choice(["dap", "da360", "gsfix3d", "all"]),
    default="all",
    help="要下载的模型权重",
)
@click.option("--verify", is_flag=True, help="校验已下载的权重")
def download_models(model: str, verify: bool):
    """下载并缓存模型权重。"""
    if model in ("dap", "all"):
        from .dap_model import DAPModel

        click.echo("Downloading DAP model weights...")
        try:
            path = DAPModel._get_or_download_weights()
            click.echo(f"DAP weights cached at: {path}")
            if verify:
                if DAPModel._verify_checksum(Path(path)):
                    click.echo("Checksum verified")
                else:
                    click.echo("Checksum verification skipped (no reference hash)")
        except Exception as e:
            click.echo(f"DAP download failed: {e}", err=True)
            if model == "dap":
                raise click.Abort()

    if model in ("da360", "all"):
        try:
            from .da360_model import DA360Model

            click.echo("Downloading DA360 model weights...")
            path = DA360Model._get_or_download_weights()
            click.echo(f"DA360 weights cached at: {path}")
        except ImportError:
            click.echo(
                "DA360 model not yet available (architecture files needed)", err=True
            )
        except Exception as e:
            click.echo(f"DA360 download failed: {e}", err=True)
            if model == "da360":
                raise click.Abort()

    if model in ("gsfix3d", "all"):
        click.echo("Downloading GSFix3D checkpoint...")
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                "goldoak1421/gsfixer-full-replica-room1",
                local_dir="pretrained/gsfix3d",
            )
            click.echo(f"GSFix3D checkpoint cached at: {path}")
        except ImportError:
            click.echo(
                "huggingface_hub not installed. Install with: pip install huggingface-hub",
                err=True,
            )
        except Exception as e:
            click.echo(f"GSFix3D download failed: {e}", err=True)
            if model == "gsfix3d":
                raise click.Abort()


@main.command()
@click.option("--port", default=7860, help="服务端口")
@click.option(
    "--host",
    default="0.0.0.0",
    help="绑定地址；0.0.0.0 表示本机所有网卡（局域网可访问）",
)
@click.option("--reload", is_flag=True, help="开发时启用自动重载")
def serve(port: int, host: str, reload: bool):
    """启动 Web UI 服务。"""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install uvicorn"
        )

    import copy
    import logging

    from uvicorn.config import LOGGING_CONFIG

    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("GET /api/status") == -1

    log_config = copy.deepcopy(LOGGING_CONFIG)

    if "filters" not in log_config:
        log_config["filters"] = {}

    log_config["filters"]["endpoint_filter"] = {"()": EndpointFilter}

    if "uvicorn.access" in log_config["loggers"]:
        if "filters" not in log_config["loggers"]["uvicorn.access"]:
            log_config["loggers"]["uvicorn.access"]["filters"] = []
        log_config["loggers"]["uvicorn.access"]["filters"].append("endpoint_filter")

    from api import kill_existing_server

    kill_existing_server(port)

    if host in ("0.0.0.0", "::"):
        click.echo(
            f"Starting Panorama2Gaussian web UI — open http://127.0.0.1:{port} "
            f"(listening on {host}:{port})"
        )
    else:
        click.echo(f"Starting Panorama2Gaussian web UI at http://{host}:{port}")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
