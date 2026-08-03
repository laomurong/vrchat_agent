"""CosyVoice2 克隆语音 HTTP 服务（轻量 FastAPI，任务 1）。

端点：
- ``GET  /health``      服务状态（模型是否已加载、模型文件检查结果）；
- ``POST /tts``         JSON：``text`` / ``ref_audio_path``（或 ``ref_audio`` base64）/
                        ``ref_text`` / ``lang``（zh/en/ja/ko）/ ``speed`` / ``stream``
                        → ``audio/wav`` 字节；
- ``POST /tts/upload``  multipart：``text`` / ``ref_text`` / ``lang`` + 参考音频文件上传。

字段归一化：兼容 Agent ``LocalCloneTTSEngine`` 发送的 GPT-SoVITS 命名
（``refer_wav_path`` / ``prompt_text`` / ``text_language`` / ``prompt_language``），
服务端统一转 CosyVoice 协议 → 引擎侧几乎零改动（见 model_loader.normalize_tts_params）。

- 模型只加载一次（全局单例，惰性 + 双重检查锁）；
- 端点均为同步 def → Starlette 线程池执行，模型加载/推理不阻塞事件循环；
- 重依赖（torch/cosyvoice）只在首次合成时惰性导入，未装时返回清晰中文 500。

启动：``python -m vrchat_agent.cosyvoice.server --port 8000``
一键：双击 ``scripts/start_cosyvoice.bat``（或运行 ``scripts/start_cosyvoice.ps1``）。
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .model_loader import (
    CosyVoiceModelContext,
    CosyVoiceServerError,
    TTSParams,
    check_model_files,
    normalize_tts_params,
)

logger = logging.getLogger("vrchat_agent.cosyvoice.server")

# 默认模型目录：<项目根>/model（本文件位于 src/vrchat_agent/cosyvoice/）
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "model"

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


class TTSRequest(BaseModel):
    """/tts JSON 请求体：同时接受 CosyVoice 与 GPT-SoVITS（Agent）命名。"""

    text: str = ""
    ref_audio_path: str = ""
    refer_wav_path: str = ""  # GPT-SoVITS 命名兼容
    ref_audio: str = ""       # base64 编码的参考音频字节（上传方式）
    ref_text: str = ""
    prompt_text: str = ""     # GPT-SoVITS 命名兼容
    lang: str = "zh"
    text_language: str = ""   # GPT-SoVITS 命名兼容
    prompt_language: str = "" # GPT-SoVITS 命名兼容
    speed: float = 1.0
    stream: bool = False


def create_app(
    context_factory: Callable[[], CosyVoiceModelContext] | None = None,
    model_dir: str | os.PathLike[str] | None = None,
) -> FastAPI:
    """创建 FastAPI 应用（context_factory 可注入替身，供测试离线使用）。"""
    base_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    state: dict[str, Any] = {
        "model_dir": str(base_dir),
        "context_factory": context_factory
        or (lambda: CosyVoiceModelContext(str(base_dir))),
        "context": None,
    }

    def _get_context() -> CosyVoiceModelContext:
        if state["context"] is None:
            state["context"] = state["context_factory"]()
        return state["context"]

    app = FastAPI(
        title="CosyVoice 克隆语音服务",
        version="0.1.0",
        description=(
            "CosyVoice 2.0-0.5B 克隆语音推理服务。POST /tts 兼容 Agent "
            "LocalCloneTTSEngine 的 GPT-SoVITS 命名字段（refer_wav_path/prompt_text/"
            "text_language/prompt_language），返回 audio/wav。"
        ),
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """服务健康检查：返回模型加载状态与模型文件检查结果。"""
        ctx = state["context"]
        check = check_model_files(state["model_dir"])
        return {
            "status": "ok",
            "service": "cosyvoice2",
            "model_loaded": bool(ctx is not None and ctx.loaded),
            "model_dir": state["model_dir"],
            "files_ok": check.ok,
            "missing": check.missing,
            "hints": check.hints,
        }

    def _synthesize_response(params: TTSParams) -> Response:
        """调用模型合成并返回 audio/wav；失败返回 JSON error（带状态码）。"""
        ctx = _get_context()
        try:
            wav = ctx.synthesize(
                params.text,
                params.ref_audio_path,
                params.ref_text,
                lang=params.lang,
                speed=params.speed,
            )
        except CosyVoiceServerError as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})
        except Exception as exc:  # 兜底转 500，保证响应可读
            logger.exception("CosyVoice 推理失败", extra={"event": "cosyvoice_error"})
            return JSONResponse(
                status_code=500, content={"error": f"CosyVoice 推理失败: {exc}"}
            )
        return Response(content=wav, media_type="audio/wav")

    @app.post("/tts")
    def tts(req: TTSRequest) -> Response:
        """JSON 合成：text + 参考音频路径（或 base64）→ audio/wav。"""
        params = normalize_tts_params(req.model_dump())
        # 参数校验必须先于模型加载：缺参直接 400，
        # 避免模型依赖未装/文件缺失时把缺参请求误报成 500（load() 抛错）。
        if not params.text:
            return JSONResponse(status_code=400, content={"error": "text 为空"})
        if not params.ref_audio_path and not req.ref_audio:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "ref_audio_path 为空且无 ref_audio base64："
                        "CosyVoice2 需要参考音频（3~10s 干净人声）进行声音克隆"
                    )
                },
            )
        tmp_path: str | None = None
        if req.ref_audio:
            # base64 上传方式：解码后写临时 wav，合成结束清理
            try:
                raw = base64.b64decode(req.ref_audio)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    status_code=400, content={"error": f"ref_audio base64 解码失败: {exc}"}
                )
            if not raw:
                return JSONResponse(status_code=400, content={"error": "ref_audio base64 为空"})
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="cosyvoice_ref_")
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            params.ref_audio_path = tmp_path
        try:
            return _synthesize_response(params)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _register_upload_route(fastapi_app: FastAPI) -> None:
        """注册 multipart 上传端点（需 python-multipart；未安装时跳过）。"""

        @fastapi_app.post("/tts/upload")
        def tts_upload(
            file: UploadFile = File(...),  # noqa: B008 - FastAPI 惯用 File/Form 默认值
            text: str = Form(""),
            ref_text: str = Form(""),
            lang: str = Form("zh"),
            speed: float = Form(1.0),
            stream: bool = Form(False),
        ) -> Response:
            """multipart 上传参考音频合成（curl/Postman 友好）。"""
            if not text or not text.strip():
                return JSONResponse(status_code=400, content={"error": "text 为空"})
            try:
                raw = file.file.read()
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    status_code=400, content={"error": f"参考音频读取失败: {exc}"}
                )
            if not raw:
                return JSONResponse(status_code=400, content={"error": "参考音频为空"})
            suffix = Path(file.filename or "ref.wav").suffix or ".wav"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="cosyvoice_ref_")
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            params = TTSParams(
                text=text,
                ref_audio_path=tmp_path,
                ref_text=ref_text,
                lang=lang,
                speed=speed,
                stream=stream,
            )
            try:
                return _synthesize_response(params)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # multipart 端点依赖 python-multipart；未安装时优雅跳过（JSON /tts 不受影响）
    try:
        import python_multipart  # noqa: F401
    except ImportError:
        logger.warning(
            "python-multipart 未安装，/tts/upload（multipart 上传）端点不注册；"
            "JSON /tts（含 ref_audio base64 上传）不受影响"
        )
    else:
        _register_upload_route(app)

    return app


def main() -> None:
    """CLI 入口：python -m vrchat_agent.cosyvoice.server [--model-dir ...] [--port 8000]"""
    parser = argparse.ArgumentParser(description="CosyVoice2 克隆语音 HTTP 服务")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("COSYVOICE_MODEL_DIR") or str(DEFAULT_MODEL_DIR),
        help=f"模型目录（默认 {DEFAULT_MODEL_DIR}）",
    )
    parser.add_argument("--host", default=os.environ.get("COSYVOICE_HOST") or _DEFAULT_HOST)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COSYVOICE_PORT") or _DEFAULT_PORT),
    )
    parser.add_argument("--device", default=None, help="cuda/cpu；默认自动检测")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 fp16（默认 GPU 上启用）")
    parser.add_argument(
        "--load-jit", action="store_true", help="启用 flow encoder JIT（需 GPU + flow.encoder.*.zip）"
    )
    parser.add_argument("--load-trt", action="store_true", help="启用 TensorRT（需 tensorrt）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # 启动前先做模型文件检查：尽早失败，避免空转加载报错
    check = check_model_files(args.model_dir)
    if not check.ok:
        print("[错误] 模型文件不完整，缺少：", file=sys.stderr)
        for item in check.missing:
            print(f"    - {item}", file=sys.stderr)
        for hint in check.hints:
            print(f"[提示] {hint}", file=sys.stderr)
        print("请先补全模型文件（详见 README「CosyVoice 克隆语音」节），再启动服务。", file=sys.stderr)
        raise SystemExit(1)
    for hint in check.hints:
        logger.warning("%s", hint)
    for item in check.optional_missing:
        logger.info("可选文件缺失（不影响基本推理）: %s", item)

    ctx = CosyVoiceModelContext(
        args.model_dir,
        device=args.device,
        fp16=not args.no_fp16,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
    )
    app = create_app(context_factory=lambda: ctx, model_dir=args.model_dir)

    import uvicorn

    print(f"CosyVoice 服务启动中（模型目录: {args.model_dir}）…")
    print(f"健康检查: http://{args.host}:{args.port}/health")
    print(f"合成接口: POST http://{args.host}:{args.port}/tts（返回 audio/wav）")
    print("首次 /tts 会加载模型（llm.pt 2GB，约 1~3 分钟），请耐心等待。")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
