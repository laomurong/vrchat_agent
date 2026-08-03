"""A 语音输入：DolphinSpeechToText（sherpa-onnx Dolphin small CTC 转写）。

- sherpa-onnx OfflineRecognizer（ONNX Runtime，CPU 推理），与 faster-whisper 引擎
  通过 config.asr.engine 切换（whisper | dolphin），互不影响。
- 模型目录 config.asr.model_path 指向含 model.onnx + tokens.txt 的 Dolphin 模型目录。
- 输入任意采样率 float32 np.ndarray，sherpa 内部自动重采样到 16k，无需外部转换。
- 惰性加载（首次 transcribe 时）+ asyncio 线程池执行（sherpa 推理为 CPU 阻塞，
  参考 SpeechToText 用 loop.run_in_executor，不阻塞事件循环）。
- 缺 sherpa-onnx / 模型文件不存在 → ASRError（提示 pip install sherpa-onnx）。
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from .asr import ASRError

logger = logging.getLogger("vrchat_agent.audio.asr_dolphin")

# Dolphin 模型目录内默认文件名（config.asr.model_path 指向目录时自动拼接）。
_MODEL_ONNX = "model.onnx"
_TOKENS_TXT = "tokens.txt"


class DolphinSpeechToText:
    """sherpa-onnx Dolphin small CTC 转写器（接口与 SpeechToText 对齐）。

    model_path: 指向 Dolphin 模型目录（含 model.onnx + tokens.txt）；也可直接指向
        model.onnx 文件（此时 tokens 需显式传入）。None → 抛 ASRError。
    tokens: tokens.txt 路径，默认在 model_path 目录内查找。
    """

    def __init__(
        self,
        model_path: str | None = None,
        tokens: str | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
        decoding_method: str = "greedy_search",
    ) -> None:
        self.model_path = model_path
        self.tokens = tokens
        self.num_threads = num_threads
        self.provider = provider
        self.decoding_method = decoding_method
        self._recognizer = None
        self._load_lock = asyncio.Lock()

    def _resolve_files(self) -> tuple[str, str]:
        """解析 model.onnx 与 tokens.txt 的绝对路径（兼容目录/文件两种写法）。

        1) model_path 为目录 → 拼接目录内 model.onnx / tokens.txt；
        2) model_path 为文件（model.onnx）→ 直接使用，tokens 从显式配置或同目录取。
        """
        if not self.model_path:
            raise ASRError("Dolphin ASR 未配置 model_path（指向含 model.onnx 的目录）")
        path = os.path.expanduser(self.model_path)
        if os.path.isdir(path):
            onnx = os.path.join(path, _MODEL_ONNX)
            tokens = self.tokens or os.path.join(path, _TOKENS_TXT)
        else:
            onnx = path
            tokens = self.tokens or (
                os.path.join(os.path.dirname(path), _TOKENS_TXT) if os.path.dirname(path) else None
            )
        if not tokens:
            raise ASRError(f"Dolphin ASR 缺少 tokens.txt（{self.tokens or path}）")
        return onnx, tokens

    async def _ensure_model(self):
        """惰性加载 sherpa-onnx recognizer（线程安全，缺依赖/缺文件抛 ASRError）。"""
        if self._recognizer is not None:
            return
        async with self._load_lock:
            if self._recognizer is not None:
                return
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise ASRError("sherpa-onnx 未安装（pip install sherpa-onnx）") from exc
            try:
                onnx_path, tokens_path = self._resolve_files()
                for f in (onnx_path, tokens_path):
                    if not os.path.isfile(f):
                        raise ASRError(f"Dolphin ASR 模型文件缺失: {f}")
                logger.info(
                    "加载 sherpa-onnx Dolphin 模型: %s（%s/%s/%s）",
                    onnx_path, self.provider, self.num_threads, self.decoding_method,
                )
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
                    model=onnx_path,
                    tokens=tokens_path,
                    num_threads=self.num_threads,
                    provider=self.provider,
                    decoding_method=self.decoding_method,
                )
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError(
                    f"加载 Dolphin 模型失败（{self.model_path}）: {exc}"
                ) from exc

    async def transcribe(self, audio: np.ndarray) -> str:
        """转写一段 float32 音频（任意采样率，sherpa 自动重采样），返回识别文本。"""
        if audio is None or len(audio) == 0:
            return ""
        await self._ensure_model()
        loop = asyncio.get_running_loop()

        def _run() -> str:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self._recognizer.decode_stream(stream)
            return stream.result.text if stream.result else ""

        try:
            text = await loop.run_in_executor(None, _run)
            logger.debug("Dolphin ASR 结果: %r", text)
            return text
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError(f"Dolphin ASR 转写失败: {exc}") from exc
