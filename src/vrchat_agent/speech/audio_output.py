"""C 语音输出：AudioOutput（sounddevice 播放到 VB-Audio 虚拟声卡，C-02）。

平台依赖：[Win] VB-Audio Virtual Cable（TTS 输出 → VRChat 麦克风）。
- play(clip)：在 executor 线程中播放（不阻塞事件循环），阻塞到真正播完
- stop()：立即静音（置停止标记 + abort 当前流 + sd.stop()），供打断/急停使用

播放链路修复（BugFix：TTS 无声）：
旧实现用 `with sd.OutputStream(...) as stream: stream.write(chunk)` 分块写入，
但实测 sounddevice 的 write() 是**缓冲语义**——数据写入 PortAudio 内部环形缓冲后
立即返回（1.49s 音频 0.12s 写完），`with` 退出时 stop()/close() 把仍在缓冲中
未播出的音频直接丢弃 → 用户听不见声音且无任何报错。
本实现改为**回调式 OutputStream**（callback 拉取样本 + finished_callback 设完成
事件）：播放线程阻塞到 finished_callback 触发（= 音频真正播完）才返回；急停用
stream.abort() 立即丢弃缓冲（实测可被另一线程安全调用，不崩溃）。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

from ..audio.device_utils import DeviceResolveError, resolve_output_device_index
from .tts import AudioClip

logger = logging.getLogger("vrchat_agent.speech.output")


class AudioOutputError(RuntimeError):
    """音频输出错误。"""


class AudioOutput:
    """把 AudioClip 播放到指定输出设备（默认 VB-Audio 虚拟声卡）。"""

    def __init__(self, device: str | None = None, sample_rate: int = 16000, chunk_frames: int = 4096) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_frames = chunk_frames
        self._stop_flag = threading.Event()
        # 当前播放中的 OutputStream（供 stop() 从另一线程 abort 急停）。
        self._active_stream: Any = None
        # A-06：播放互斥锁——保证任意时刻只有一个活动播放流。
        # play() 串行化：并发播放请求（如 TTS 与降级 beep）不会互相覆盖
        # _active_stream，避免打断/急停只 abort 后一条流而原语音继续播。
        self._play_lock = asyncio.Lock()

    async def play(self, clip: AudioClip) -> None:
        """播放一段音频（阻塞直到播完或被 stop() 中断）。

        A-06：互斥保护——同一时刻只允许一个活动播放流；若已有流在播
        （理论上被锁挡住，防御性兜底），先 stop() 停掉旧流再播新流。
        """
        if clip is None or len(clip.samples) == 0:
            return
        async with self._play_lock:
            # A-06：已有活动流 → 先停止（等待旧播放线程退出）再播新流
            if self._active_stream is not None:
                await self.stop()
            self._stop_flag = threading.Event()
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._play_blocking, clip)
            except AudioOutputError:
                raise
            except Exception as exc:
                raise AudioOutputError(f"播放失败: {exc}") from exc

    @staticmethod
    def _resolve_output_device(device: str | None) -> int | str | None:
        """把输出侧配置里的设备标识解析为 sounddevice 可用参数（P0 修复）。

        与输入侧 _resolve_mic_device 对称（共享 audio.device_utils 公共解析）：
        - None / 空串    → None（系统默认设备）；
        - "15"（数字串） → int(15)（Web 下拉框存 index 的契约，不再当设备名搜索）；
        - 设备名         → 精确匹配唯一 index；重名/找不到 → AudioOutputError
          列出候选（找不到时含 VB-Audio 安装提示）。
        """
        try:
            return resolve_output_device_index(device)
        except DeviceResolveError as exc:
            raise AudioOutputError(str(exc)) from exc

    def _play_blocking(self, clip: AudioClip) -> None:
        """阻塞播放：直到音频真正播完（finished_callback）或被 stop() 急停。

        修复说明（BugFix：TTS 无声）：
        - 旧实现 `with OutputStream + stream.write(chunk)` 的 write() 只把数据塞进
          PortAudio 内部缓冲就返回（不等待播出）；`with` 退出时 stop()/close()
          丢弃未播出的缓冲音频 → 用户听不见。实测 1.49s 音频 0.12s 写完即证。
        - 本实现用**回调式 OutputStream**：callback 从 samples 拉取数据，播完后
          抛 CallbackStop 优雅停止，finished_callback 置完成事件；播放线程阻塞
          等待该事件 → play() 耗时 ≈ 音频时长（真实播出）。
        - 急停：stop() 置停止标记并从另一线程 stream.abort()（立即丢弃缓冲），
          播放线程在 50ms 内感知标记并中止，返回耗时 ≈ 打断时刻。
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioOutputError("sounddevice 未安装（pip install sounddevice）") from exc

        samples = clip.samples
        rate = clip.sample_rate
        if rate != self._sample_rate:
            samples = self._resample(samples, rate, self._sample_rate)
            rate = self._sample_rate

        # 设备语义归一化（BugFix P0）：device=None = 系统默认设备；
        # 配置层空串（Web 下拉框默认设备项）须转 None，否则 sounddevice 会把 ""
        # 当名称模糊搜索（多设备时 ValueError 崩溃）。
        # P0 修复（QA 遗留隐患）：数字字符串 index（如 "15"）必须先解析为 int，
        # 否则 sounddevice 会把它当设备名子串搜索（No output device matching '15'）。
        resolved = self._resolve_output_device(self._device)

        duration = len(samples) / float(rate) if rate > 0 else 0.0
        logger.info(
            "播放开始: device=%s rate=%d samples=%d (%.2fs)",
            resolved, rate, len(samples), duration,
            extra={"event": "audio_output_play_start"},
        )
        start_ts = time.monotonic()

        state = {"pos": 0}
        done_event = threading.Event()

        def _callback(outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            """从 samples 拉取下一段数据；播完（或已急停）优雅停止。"""
            if self._stop_flag.is_set():
                outdata.fill(0.0)
                return
            pos = state["pos"]
            remaining = len(samples) - pos
            n = min(frames, remaining)
            if n > 0:
                outdata[:n] = samples[pos : pos + n].reshape(-1, 1)
                state["pos"] = pos + n
            if n < frames:
                outdata[n:] = 0.0
            if remaining <= frames:
                # 全部样本已交给声卡：优雅停止（播完剩余缓冲，不截断）。
                raise sd.CallbackStop

        def _finished_callback() -> None:
            done_event.set()

        try:
            stream = sd.OutputStream(
                device=resolved,
                samplerate=rate,
                channels=1,
                dtype="float32",
                callback=_callback,
                finished_callback=_finished_callback,
            )
        except sd.PortAudioError as exc:
            logger.warning(
                "打开音频输出流失败: %s", exc, extra={"event": "audio_output_open_failed"}
            )
            raise AudioOutputError(
                f"打开音频输出设备失败: {exc}。"
                "提示：若该设备仅支持特定采样率（如 WASAPI 只认 48k），"
                "请在 Web 控制台换一个支持 16k 的输出设备（MME/DirectSound，index）"
            ) from exc

        self._active_stream = stream
        try:
            stream.start()
        except sd.PortAudioError as exc:
            logger.warning(
                "启动音频输出流失败: %s", exc, extra={"event": "audio_output_start_failed"}
            )
            try:
                stream.close()
            except Exception:
                pass
            self._active_stream = None
            raise AudioOutputError(f"启动音频输出设备失败: {exc}") from exc

        interrupted = False
        try:
            # 阻塞直到真正播完（finished_callback）或被 stop() 急停。
            # deadline 是安全兜底（设备异常挂死时强制中止，避免永久阻塞）。
            deadline = start_ts + max(2.0, duration) + 2.0
            while not done_event.is_set():
                if self._stop_flag.is_set():
                    interrupted = True
                    self._abort_stream_safe(stream, sd, "急停")
                    break
                if time.monotonic() >= deadline:
                    logger.warning(
                        "播放超过安全时限（%.2fs）仍未完成，强制中止", duration + 2.0,
                        extra={"event": "audio_output_timeout"},
                    )
                    self._abort_stream_safe(stream, sd, "超时")
                    break
                done_event.wait(timeout=0.05)
        finally:
            try:
                stream.close()
            except sd.PortAudioError as exc:
                logger.warning(
                    "关闭音频输出流失败: %s", exc, extra={"event": "audio_output_close_failed"}
                )
            self._active_stream = None

        elapsed = time.monotonic() - start_ts
        logger.info(
            "播放完成（用时 %.2fs）%s", elapsed, "（被中断）" if interrupted else "",
            extra={"event": "audio_output_play_done", "elapsed": elapsed, "interrupted": interrupted},
        )

    @staticmethod
    def _abort_stream_safe(stream: Any, sd: Any, reason: str) -> None:
        """安全中止播放流（急停/超时）：失败仅记 WARNING，不打断主流程。"""
        try:
            stream.abort()
        except Exception as exc:
            logger.warning(
                "%s中止播放失败: %s", reason, exc, extra={"event": "audio_output_abort_failed"}
            )

    async def stop(self) -> None:
        """立即静音：置停止标记、abort 当前播放流并调用 sd.stop()。

        供打断/急停使用：播放线程在 50ms 内感知停止标记并中止；
        abort() 从另一线程调用立即丢弃声卡缓冲（实测无崩溃）。
        """
        self._stop_flag.set()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._sd_stop)
        except Exception:
            pass

    def _sd_stop(self) -> None:
        """同步急停：abort 当前播放流（若有）+ sd.stop() 兜底。"""
        stream = self._active_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """线性插值重采样（简单实现，保证 16k 全链路约定）。"""
        if src_rate == dst_rate or len(samples) == 0:
            return samples
        n = max(1, int(len(samples) * dst_rate / src_rate))
        x_old = np.linspace(0.0, len(samples), num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, len(samples), num=n, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)
