"""A 语音输入：SpeechToText（faster-whisper 本地转写，A-02）。

- faster-whisper 基于 CTranslate2：GPU（cuda/float16）或 CPU（int8）均可。
- 模型惰性加载（首次 transcribe 时），避免无 GPU/未安装依赖时 import 失败。
- 支持本地模型目录（model_path）：传入目录路径时 faster-whisper 直接加载本地
  模型，无需联网下载；未配置/目录不可用时回退 model_size 在线下载（HF 镜像兜底）。
- 输入：16k mono float32 np.ndarray；输出：识别文本（str）。
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

# 国内网络优先走 HuggingFace 镜像（仅当未设置时注入，便于用户用环境变量覆盖）。
# 与参考项目 vrc_chat_translator/voice.py 保持一致，保证在线下载模型能连上。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger("vrchat_agent.audio.asr")


def _ensure_nvidia_dll_on_path() -> None:
    """将 pip 安装的 nvidia CUDA 运行库 bin 目录注入 PATH（幂等）。

    faster-whisper（CTranslate2）在推理期按 PATH 搜索 cublas64_12.dll 等运行库；
    pip 安装的 nvidia-cublas-cu12 / nvidia-cudnn-cu12 等包把 DLL 放在
    site-packages/nvidia/<pkg>/bin，默认不在 PATH，导致 GPU 推理报
    "Library cublas64_12.dll is not found or cannot be loaded"。
    此处扫描并注入，保证 GPU 可用；若注入失败仅记 debug，后续由降级兜底接管。
    """
    try:
        import glob
        import site

        injected: list[str] = []
        for sp in site.getsitepackages():
            for bin_dir in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
                if bin_dir and bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    injected.append(bin_dir)
        if injected:
            logger.debug("已注入 nvidia DLL 目录: %s", injected)
    except Exception:
        logger.debug(
            "注入 nvidia DLL 目录失败（忽略，GPU 不可用时会自动降级 CPU）",
            exc_info=True,
        )


_ensure_nvidia_dll_on_path()

# GPU 加载/推理失败的典型错误关键字（ctranslate2 / CUDA 运行库缺失或驱动异常）。
# 命中任一关键字即视为"GPU 不可用"，触发 device=cpu + compute_type=int8 自动降级。
_GPU_ERROR_KEYWORDS = ("cublas", "cudnn", "cuda", "driver")


def _is_gpu_error(exc: Exception) -> bool:
    """判断异常是否属于 GPU 运行库/驱动类错误（大小写不敏感）。"""
    msg = str(exc).lower()
    return any(kw in msg for kw in _GPU_ERROR_KEYWORDS)


class ASRError(RuntimeError):
    """ASR 转写错误。"""


class SpeechToText:
    """faster-whisper 转写器。

    model_path: 本地 faster-whisper 模型目录（含 model.bin）。配置后优先于
        model_size 加载；为 None 或目录不可用时回退 model_size（在线下载）。
    """

    def __init__(
        self,
        model_size: str = "medium",
        model_path: str | None = None,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "zh",
    ) -> None:
        self.model_size = model_size
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._load_lock = asyncio.Lock()

    def _resolve_model_arg(self) -> str:
        """决定传给 WhisperModel 的加载参数：本地目录优先，否则模型尺寸名。

        1) model_path 配置为含 model.bin 的本地目录 → 直接返回该目录路径
           （faster-whisper 传入本地目录路径时直接加载，无需联网下载）；
        2) model_path 是目录但缺 model.bin → 仍按该路径尝试，让 faster-whisper
           给出明确报错（避免静默回退掩盖用户配置错误）；
        3) 其他情况（未配置 / 路径不存在）→ 回退 model_size 在线下载。
        """
        if self.model_path:
            path = os.path.expanduser(self.model_path)
            if os.path.isdir(path):
                if not os.path.isfile(os.path.join(path, "model.bin")):
                    logger.warning("model_path 目录缺少 model.bin: %s", path)
                return path
            logger.warning(
                "model_path 不存在（%s），回退 model_size=%s 在线下载",
                self.model_path, self.model_size,
            )
        return self.model_size

    def _load_with_fallback(
        self, load_arg: str, *, allow_fallback: bool = True
    ):
        """按配置 device/compute_type 加载 Whisper 模型；GPU 库缺失时降级 CPU 重试一次。

        降级契约（符合"GPU 缺库不阻断 Agent"）：
        - 若当前 device != cpu 且加载异常命中 GPU 错误关键字（cublas/cudnn/cuda/driver），
          记录 WARNING 后以 device=cpu + compute_type=int8 重试一次，并同步更新实例状态，
          保证后续转写不再反复尝试 GPU；
        - 降级后仍失败 → 抛 ASRError（由调用方透传）；
        - 非 GPU 错误 → 原样抛出，交由 _ensure_model 包装为 ASRError。
        """
        try:
            from faster_whisper import WhisperModel

            logger.info(
                "加载 faster-whisper 模型: %s (%s/%s)",
                load_arg, self.device, self.compute_type,
            )
            return WhisperModel(
                load_arg, device=self.device, compute_type=self.compute_type
            )
        except Exception as exc:
            if allow_fallback and self.device != "cpu" and _is_gpu_error(exc):
                logger.warning(
                    "GPU 加载失败（%s/%s），自动降级 device=cpu compute_type=int8 重试: %s",
                    self.device, self.compute_type, exc,
                )
                self.device = "cpu"
                self.compute_type = "int8"
                try:
                    model = WhisperModel(load_arg, device="cpu", compute_type="int8")
                    logger.info("降级成功：faster-whisper 使用 CPU/int8")
                    return model
                except Exception as exc2:
                    raise ASRError(
                        f"加载 faster-whisper 模型失败（GPU 降级 CPU 后仍失败）: {exc2}"
                    ) from exc2
            raise

    async def _ensure_model(self):
        """惰性加载 Whisper 模型（线程安全，A-09：加载在 executor 线程执行）。"""
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                load_arg = self._resolve_model_arg()
                # A-09：模型加载为 CPU/GPU 密集阻塞（数秒~分钟），必须移出事件循环，
                # 否则首次 transcribe 会卡死 Web 控制台/急停。双重检查锁语义保留：
                # 锁内 await，其它协程无法并发进入。
                self._model = await asyncio.to_thread(self._load_with_fallback, load_arg)
            except ImportError as exc:
                raise ASRError("faster-whisper 未安装（pip install faster-whisper）") from exc
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError(
                    f"加载 faster-whisper 模型失败（{self.model_path or self.model_size}/{self.device}/{self.compute_type}）: {exc}"
                ) from exc

    async def transcribe(self, audio: np.ndarray) -> str:
        """转写一段 16k mono float32 音频，返回识别文本。"""
        if audio is None or len(audio) == 0:
            return ""
        await self._ensure_model()
        loop = asyncio.get_running_loop()
        language = self.language if self.language and self.language != "auto" else None

        def _run() -> str:
            segments, _info = self._model.transcribe(
                audio, language=language, beam_size=5, vad_filter=False
            )
            return "".join(seg.text for seg in segments).strip()

        try:
            text = await loop.run_in_executor(None, _run)
            logger.debug("ASR 结果: %r", text)
            return text
        except ASRError:
            raise
        except Exception as exc:
            if self.device != "cpu" and _is_gpu_error(exc):
                # 模型已在 GPU 上但推理期抛 GPU 运行库错误（如驱动/显存异常）：
                # 重置模型并降级 CPU/int8 重试一次，保证不阻断 Agent。
                logger.warning(
                    "ASR 转写 GPU 错误（%s），重置模型并降级 CPU/int8 重试: %s",
                    self.device, exc,
                )
                self._model = None
                self.device = "cpu"
                self.compute_type = "int8"
                await self._ensure_model()
                try:
                    text = await loop.run_in_executor(None, _run)
                    logger.debug("ASR 降级转写结果: %r", text)
                    return text
                except Exception as exc2:
                    raise ASRError(
                        f"ASR 转写失败（GPU 降级 CPU 后仍失败）: {exc2}"
                    ) from exc2
            raise ASRError(f"ASR 转写失败: {exc}") from exc


def create_asr(config):
    """按 config.asr.engine 装配 ASR 引擎（whisper | dolphin）。

    - engine=whisper（默认）：原路径不变，SpeechToText 参数逐字透传；
    - engine=dolphin：DolphinSpeechToText（sherpa-onnx Dolphin small CTC），
      model_path 指向含 model.onnx + tokens.txt 的模型目录。
    """
    engine = getattr(config.asr, "engine", "whisper")
    if engine == "dolphin":
        from .asr_dolphin import DolphinSpeechToText

        return DolphinSpeechToText(
            model_path=config.asr.model_path,
            decoding_method=getattr(config.asr, "decoding_method", "greedy_search"),
        )
    return SpeechToText(
        model_size=config.asr.model_size,
        model_path=config.asr.model_path,
        device=config.asr.device,
        compute_type=config.asr.compute_type,
        language=config.asr.language,
    )
