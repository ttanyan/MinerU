import sys
import uuid
import os
import re
import tempfile
import asyncio
import uvicorn
import click
import zipfile
from pathlib import Path
import glob
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from typing import List, Optional
from loguru import logger
from base64 import b64encode

# MinerU 内部导入
from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes, word_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

# --- 日志配置 ---
log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level)

# --- 全局变量与辅助类 ---
_request_semaphore: Optional[asyncio.Semaphore] = None


class ProgressTracker:
    def __init__(self):
        self.progress = 0
        self.status = "初始化"

    def update(self, progress: float, status: str):
        self.progress = progress
        self.status = status

    def get_progress(self):
        return {"progress": self.progress, "status": self.status}


async def limit_concurrency():
    if _request_semaphore is not None:
        if _request_semaphore.locked():
            raise HTTPException(
                status_code=503,
                detail=f"Server is at maximum capacity: {os.getenv('MINERU_API_MAX_CONCURRENT_REQUESTS', '0')}."
            )
        async with _request_semaphore:
            yield
    else:
        yield


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'


def cleanup_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


# --- 创建 API 路由组 ---
api_router = APIRouter(prefix="/api")


@api_router.post(path="/file_parse", dependencies=[Depends(limit_concurrency)])
async def parse_pdf(
        files: List[UploadFile] = File(..., description="Upload pdf, image, or Word files for parsing"),
        output_dir: str = Form("./output", description="Output local directory"),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("hybrid-auto-engine"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        response_format_zip: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
):
    # 从 app 实例状态中获取配置 (FastAPI 实例会在下方创建)
    from fastapi import Request
    config = getattr(app.state, "config", {})

    try:
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)
            temp_path = Path(unique_dir) / file_path.name
            with open(temp_path, "wb") as f:
                f.write(content)

            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes + word_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"Failed to load file: {str(e)}"})
            else:
                return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {file_suffix}"})

        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        await aio_do_parse(
            output_dir=unique_dir, pdf_file_names=pdf_file_names, pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list, backend=backend, parse_method=parse_method,
            formula_enable=formula_enable, table_enable=table_enable, server_url=server_url,
            f_draw_layout_bbox=False, f_draw_span_bbox=False, f_dump_md=return_md,
            f_dump_middle_json=return_middle_json, f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False, f_dump_content_list=return_content_list,
            start_page_id=start_page_id, end_page_id=end_page_id, **config
        )

        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
            os.close(zip_fd)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_name in pdf_file_names:
                    safe_pdf_name = sanitize_filename(pdf_name)
                    # 路径匹配逻辑
                    if backend.startswith("pipeline"):
                        p_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    elif backend.startswith("vlm"):
                        p_dir = os.path.join(unique_dir, pdf_name, "vlm")
                    else:
                        p_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")

                    if not os.path.exists(p_dir): continue
                    if return_md:
                        path = os.path.join(p_dir, f"{pdf_name}.md")
                        if os.path.exists(path): zf.write(path,
                                                          arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))
                    if return_images:
                        images_dir = os.path.join(p_dir, "images")
                        for img in glob.glob(os.path.join(glob.escape(images_dir), "*.jpg")):
                            zf.write(img, arcname=os.path.join(safe_pdf_name, "images", os.path.basename(img)))

            return FileResponse(path=zip_path, media_type="application/zip", filename="results.zip",
                                background=BackgroundTask(cleanup_file, zip_path))
        else:
            result_dict = {}
            for pdf_name in pdf_file_names:
                result_dict[pdf_name] = {}
                data = result_dict[pdf_name]
                if backend.startswith("pipeline"):
                    p_dir = os.path.join(unique_dir, pdf_name, parse_method)
                elif backend.startswith("vlm"):
                    p_dir = os.path.join(unique_dir, pdf_name, "vlm")
                else:
                    p_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")

                if os.path.exists(p_dir):
                    if return_md: data["md_content"] = get_infer_result(".md", pdf_name, p_dir)
                    if return_images:
                        img_dir = os.path.join(p_dir, "images")
                        data["images"] = {os.path.basename(p): f"data:image/jpeg;base64,{encode_image(p)}"
                                          for p in glob.glob(os.path.join(glob.escape(img_dir), "*.jpg"))}

            return JSONResponse(status_code=200,
                                content={"backend": backend, "version": __version__, "results": result_dict})
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": f"Internal Error: {str(e)}"})


# --- FastAPI 核心应用 ---
def create_app():
    enable_docs = str(os.getenv("MINERU_API_ENABLE_FASTAPI_DOCS", "1")).lower() in ("1", "true", "yes")
    app = FastAPI(
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )

    global _request_semaphore
    try:
        mcr = int(os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "0"))
        if mcr > 0:
            _request_semaphore = asyncio.Semaphore(mcr)
            logger.info(f"Concurrency limited to {mcr}")
    except:
        pass

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                       allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 1. 首先挂载 API 路由组 (处理所有 /api/* 请求)
    app.include_router(api_router)

    # 2. 根路径重定向或特定处理 (可选)
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # 3. 最后挂载静态文件服务 (处理剩下的所有请求，如 /, /index.html, /assets/*)
    static_dir = Path(__file__).parent / "static" / "web"
    if static_dir.exists():
        logger.info(f"Mounting static files from {static_dir}")
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    else:
        logger.warning("Static directory not found, web UI will not be available.")

    return app


app = create_app()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8000, type=int)
@click.option('--reload', is_flag=True)
def main(ctx, host, port, reload, **kwargs):
    kwargs.update(arg_parse(ctx))
    app.state.config = kwargs

    mcr = str(kwargs.get("mineru_api_max_concurrent_requests", "0") or "0")
    os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = mcr

    uvicorn.run("mineru.cli.fast_api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()