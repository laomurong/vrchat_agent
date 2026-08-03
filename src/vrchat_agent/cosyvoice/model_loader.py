"""CosyVoice2 克隆语音：模型文件检查 / 字段归一化 / WAV 编码 / 惰性模型加载（任务 1）。

设计约束（离线可测）：
- 文件检查、字段归一化、WAV 编码**不依赖 torch/cosyvoice**，QA 可在无 GPU/无模型环境跑；
- torch/cosyvoice 等重依赖只在 :class:`CosyVoiceModelContext.load` 内**惰性导入**，
  未安装时抛出带清晰中文信息的 :class:`CosyVoiceServerError`（HTTP 500 响应体），
  Agent 侧 FallbackTTSEngine 会自动回退 Edge，不阻断说话。

fp16 兼容性结论（核对官方 FunAudioLLM/CosyVoice `cosyvoice/cli/cosyvoice.py`）：
- `flow.encoder.fp16.zip` 是 **TorchScript（JIT）** 打包的 flow encoder，仅当
  `load_jit=True` 且 `fp16=True` 且设备为 CUDA 时被 `torch.jit.load` 加载；
- 官方默认推理（`load_jit=False`）走**纯 PyTorch flow encoder**（权重来自
  flow.pt / flow.cache.pt），**完全不使用** flow.encoder.*.zip —— 因此
  「用户只有 fp16.zip、没有 fp32.zip」**不影响基本推理**，无需换 tp16；
- 若要在 GPU 上启用 JIT 加速，fp16.zip 与 `fp16=True` 是配套的，可直接用；
- 官方模型卡（HuggingFace `FunAudioLLM/CosyVoice2-0.5B`）实际提供的是 `flow.pt`，
  部分镜像命名为 `flow.cache.pt`。官方 `CosyVoice2.__init__` 固定加载
  `{model_dir}/flow.pt`，因此本模块在只有 `flow.cache.pt` 时自动补一个硬链接
  `flow.pt`（NTFS 免管理员），保证官方加载逻辑零改动可跑。
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vrchat_agent.cosyvoice.model_loader")

# 推理必需文件（name → 中文说明）
REQUIRED_FILES: list[tuple[str, str]] = [
    ("llm.pt", "LLM 语言模型权重（约 2GB）"),
    ("hift.pt", "HiFT 声码器权重"),
    ("campplus.onnx", "说话人编码器（onnxruntime）"),
    ("speech_tokenizer_v2.onnx", "语音 tokenizer（onnxruntime）"),
    ("cosyvoice2.yaml", "模型结构配置"),
]

# flow 流模型权重：官方 HF 卡=flow.pt；部分镜像=flow.cache.pt（任一即可）
FLOW_WEIGHTS: tuple[str, ...] = ("flow.pt", "flow.cache.pt")

# 可选文件（缺失不阻断，仅提示）
OPTIONAL_FILES: dict[str, str] = {
    "flow.encoder.fp16.zip": "flow encoder JIT（GPU+fp16 加速，可选）",
    "flow.encoder.fp32.zip": "flow encoder JIT（GPU+fp32 加速，可选）",
    "flow.decoder.estimator.fp32.onnx": "flow decoder ONNX（TensorRT 加速用，可选）",
    "speech_tokenizer_v2.batch.onnx": "批量语音 tokenizer（可选）",
    "CosyVoice-BlankEN": "英文等多语种 tokenizer 增强（中文克隆不需要）",
}

_DEFAULT_SAMPLE_RATE = 24000  # CosyVoice2 cosyvoice2.yaml 固定 sample_rate


class CosyVoiceServerError(RuntimeError):
    """CosyVoice 服务错误；status_code 决定 HTTP 响应状态（默认 500）。"""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ModelFileCheck:
    """模型目录完整性检查结果。"""

    ok: bool
    missing: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)


def check_model_files(model_dir: str | os.PathLike[str]) -> ModelFileCheck:
    """检查模型目录关键文件；重点识别 flow.*.crdownload 未完成下载。"""
    d = Path(model_dir)
    missing: list[str] = []
    hints: list[str] = []
    optional_missing: list[str] = []

    for name, desc in REQUIRED_FILES:
        if not (d / name).exists():
            missing.append(f"{name}（{desc}）")

    flow_found = any((d / name).exists() for name in FLOW_WEIGHTS)
    if not flow_found:
        missing.append("flow.pt / flow.cache.pt（flow 流模型权重，必需）")
        for name in FLOW_WEIGHTS:
            crd = d / f"{name}.crdownload"
            if crd.exists():
                size = crd.stat().st_size if crd.is_file() else 0
                hints.append(
                    f"检测到 {name}.crdownload（{size} 字节，未完成下载）→ "
                    f"请重新下载 {name}（之前下载中断），完整文件约 1GB+"
                )
    else:
        # 完整文件与 0 字节 crdownload 并存时也提醒（避免误判）
        for name in FLOW_WEIGHTS:
            crd = d / f"{name}.crdownload"
            if crd.exists() and crd.is_file() and crd.stat().st_size == 0:
                hints.append(f"检测到 {name}.crdownload（0 字节）→ 可删除该残留文件，避免混淆")

    for name, desc in OPTIONAL_FILES.items():
        if not (d / name).exists():
            optional_missing.append(f"{name}（{desc}）")

    return ModelFileCheck(
        ok=not missing,
        missing=missing,
        hints=hints,
        optional_missing=optional_missing,
    )


def resolve_flow_weights(model_dir: str | os.PathLike[str]) -> Path | None:
    """返回 flow 权重路径：优先 flow.pt（官方命名），其次 flow.cache.pt。"""
    d = Path(model_dir)
    for name in FLOW_WEIGHTS:
        p = d / name
        if p.exists():
            return p
    return None


def _ensure_flow_pt(model_dir: Path) -> None:
    """官方 CosyVoice2 固定加载 flow.pt；镜像卡只有 flow.cache.pt 时补映射。"""
    flow_pt = model_dir / "flow.pt"
    if flow_pt.exists():
        return
    flow_cache = model_dir / "flow.cache.pt"
    if not flow_cache.exists():
        return  # 缺文件由 check_model_files 报错
    try:
        os.link(flow_cache, flow_pt)  # NTFS 硬链接（免管理员，同卷）
        logger.info("已用硬链接把 flow.cache.pt 映射为 flow.pt（官方加载约定）")
    except OSError:
        shutil.copyfile(flow_cache, flow_pt)
        logger.info("已复制 flow.cache.pt 为 flow.pt（硬链接失败，改用复制）")


# ---------------------------------------------------------------------------
# 字段归一化（Agent GPT-SoVITS 命名 → CosyVoice 协议）
# ---------------------------------------------------------------------------
@dataclass
class TTSParams:
    """归一化后的 /tts 请求参数。"""

    text: str = ""
    ref_audio_path: str = ""
    ref_text: str = ""
    lang: str = "zh"
    speed: float = 1.0
    stream: bool = False


def normalize_tts_params(raw: dict[str, Any]) -> TTSParams:
    """把 Agent 发送的字段（GPT-SoVITS 命名）统一为 CosyVoice 协议。

    兼容别名：
    - 参考音频：ref_audio_path / refer_wav_path；
    - 参考文本：ref_text / prompt_text；
    - 语言：lang / text_language / prompt_language（优先级从高到低）。
    """
    text = str(raw.get("text") or "").strip()
    ref_audio = str(
        raw.get("ref_audio_path") or raw.get("refer_wav_path") or ""
    ).strip()
    ref_text = str(raw.get("ref_text") or raw.get("prompt_text") or "").strip()
    lang = str(
        raw.get("lang")
        or raw.get("text_language")
        or raw.get("prompt_language")
        or "zh"
    ).strip() or "zh"
    try:
        speed = float(raw.get("speed", 1.0) or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    if speed <= 0:
        speed = 1.0
    stream = bool(raw.get("stream", False))
    return TTSParams(
        text=text,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        lang=lang,
        speed=speed,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# WAV 编码（纯 Python，零依赖）
# ---------------------------------------------------------------------------
def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int = _DEFAULT_SAMPLE_RATE) -> bytes:
    """float32 音频 → 16-bit PCM WAV 字节（44 字节 RIFF 头 + PCM 数据）。

    CosyVoice2 输出 24k mono float32；用纯 Python 写 WAV 避免依赖
    torchaudio/soundfile 的保存接口差异。
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        if arr.shape[0] == 1:
            # CosyVoice2 inference_zero_shot 的 tts_speech 是 batch-first (1, T)
            # （batch=1 在前）：这是单声道 batch 形式，**不是立体声**！
            # 取 batch 0 保留全部采样；绝不能沿时间轴（axis=1）平均，
            # 否则会把整段音频压成 1 个采样 → 输出近空 WAV（静音）。
            arr = arr[0]
        elif arr.shape[1] == 2:
            # 帧交错立体声 (N, 2)（N>1 帧 × 2 通道）：mean(axis=1) 混为单声道
            arr = arr.mean(axis=1)
        else:
            # 兜底：通道-first 立体声 (2, T) 沿 axis=0 混音；其余非常规 2D 取第一行。
            # 统一原则：绝不沿时间轴（axis=1）压缩，保证采样数不丢失。
            arr = arr.mean(axis=0) if arr.shape[0] == 2 else arr[0]
    pcm = np.clip(arr, -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    n = len(data)
    byte_rate = int(sample_rate) * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + n,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        int(sample_rate),
        byte_rate,
        2,  # block align
        16,  # bits per sample
        b"data",
        n,
    )
    return header + data


# ---------------------------------------------------------------------------
# 模型上下文（惰性加载，线程安全）
# ---------------------------------------------------------------------------
class CosyVoiceModelContext:
    """CosyVoice2 模型单例上下文：加载一次，推理由 FastAPI 线程池调用。

    - 构造只记录参数，不 import torch/cosyvoice；
    - :meth:`load` 首次调用时执行完整加载（约 1~3 分钟，llm.pt 2GB），
      双重检查锁保证并发安全；
    - :meth:`synthesize` 返回 24k mono 16-bit PCM WAV 字节。
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        device: str | None = None,
        fp16: bool = True,
        load_jit: bool = False,
        load_trt: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16
        self.load_jit = load_jit
        self.load_trt = load_trt
        self._model: Any = None
        self._sample_rate: int = _DEFAULT_SAMPLE_RATE
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """模型是否已加载完成。"""
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _ensure_imports(self) -> None:
        """惰性导入重依赖，缺失时给出可操作的中文错误。"""
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError as exc:
            raise CosyVoiceServerError(
                "缺少 torch/torchaudio：请在 CosyVoice 推理环境安装 "
                "（推荐 conda create -n cosyvoice python=3.11 后 pip install torch torchaudio），"
                "或用 CosyVoice 官方环境运行 scripts/start_cosyvoice.ps1"
            ) from exc
        try:
            import cosyvoice  # noqa: F401
        except ImportError as exc:
            raise CosyVoiceServerError(
                "缺少 cosyvoice 包：请 git clone https://github.com/FunAudioLLM/CosyVoice "
                "并 pip install -e CosyVoice 后重试（部署指引见 README「CosyVoice 克隆语音」节）"
            ) from exc

    def load(self) -> CosyVoiceModelContext:
        """加载模型（幂等）。未满足前置条件时抛 CosyVoiceServerError。"""
        if self._model is not None:
            return self
        with self._lock:
            if self._model is not None:
                return self
            check = check_model_files(self.model_dir)
            if not check.ok:
                detail = "、".join(check.missing)
                if check.hints:
                    detail += "；" + "；".join(check.hints)
                raise CosyVoiceServerError(f"模型文件不完整: {detail}")
            self._ensure_imports()

            import torch
            from cosyvoice.cli.cosyvoice import CosyVoice2

            device = (self.device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
            fp16 = bool(self.fp16) and device == "cuda"
            # JIT 仅当 GPU 且对应 encoder zip 存在才启用；缺 JIT 文件回退纯 PyTorch
            load_jit = bool(self.load_jit) and device == "cuda"
            if load_jit:
                zip_name = "flow.encoder.fp16.zip" if fp16 else "flow.encoder.fp32.zip"
                if not (self.model_dir / zip_name).exists():
                    logger.warning(
                        "未找到 %s，回退纯 PyTorch flow encoder（不影响推理）", zip_name
                    )
                    load_jit = False
            # 官方固定加载 flow.pt；镜像卡只有 flow.cache.pt 时补硬链接
            _ensure_flow_pt(self.model_dir)
            logger.info(
                "CosyVoice2 模型加载中（首次约 1~3 分钟，llm.pt 2GB）… device=%s fp16=%s jit=%s",
                device,
                fp16,
                load_jit,
            )
            model = CosyVoice2(
                str(self.model_dir),
                load_jit=load_jit,
                load_trt=False,
                fp16=fp16,
            )
            self._model = model
            self._sample_rate = int(getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE))
            logger.info("CosyVoice2 模型加载完成（sample_rate=%s）", self._sample_rate)
            return self

    def synthesize(
        self,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        lang: str = "zh",
        speed: float = 1.0,
    ) -> bytes:
        """zero-shot 声音克隆合成，返回 24k mono 16-bit PCM WAV 字节。"""
        self.load()
        if not text or not str(text).strip():
            raise CosyVoiceServerError("text 为空", status_code=400)
        ref_path = str(ref_audio_path or "").strip()
        if not ref_path:
            raise CosyVoiceServerError(
                "ref_audio_path 为空：CosyVoice2 需要参考音频（3~10s 干净人声）进行声音克隆",
                status_code=400,
            )
        wav_path = Path(ref_path)
        if not wav_path.exists():
            raise CosyVoiceServerError(f"参考音频不存在: {wav_path}", status_code=400)

        model = self._model
        try:
            import torchaudio

            prompt_wav, prompt_sr = torchaudio.load(str(wav_path))
            if prompt_wav.shape[0] > 1:
                prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 立体声混为单声道
            if int(prompt_sr) != self._sample_rate:
                prompt_wav = torchaudio.functional.resample(
                    prompt_wav, int(prompt_sr), self._sample_rate
                )
            # 兼容带 lang 参数的 cosyvoice 版本（CosyVoice3 等），旧版自动忽略
            import inspect

            params = inspect.signature(model.inference_zero_shot).parameters
            kwargs: dict[str, Any] = {
                "tts_text": str(text),
                "prompt_text": str(ref_text or ""),
                "prompt_wav": prompt_wav,
                "stream": False,
                "speed": float(speed) if float(speed) > 0 else 1.0,
            }
            if "lang" in params:
                kwargs["lang"] = str(lang or "zh")
            outputs = model.inference_zero_shot(**kwargs)
            speech = None
            for out in outputs:
                speech = out.get("tts_speech")
                break
            if speech is None:
                raise CosyVoiceServerError("CosyVoice2 未返回音频")
            samples = np.asarray(speech.detach().cpu().numpy(), dtype=np.float32)
            return float32_to_wav_bytes(samples, self._sample_rate)
        except CosyVoiceServerError:
            raise
        except Exception as exc:  # 统一转服务错误，附带可读信息
            raise CosyVoiceServerError(f"CosyVoice2 推理失败: {exc}") from exc
