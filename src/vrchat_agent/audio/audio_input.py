"""A 语音输入：麦克风（sounddevice）/ 电脑内录（WASAPI 回环）→ 16kHz mono float32 块。

平台依赖：[Win] 电脑内录（WASAPI 回环）用 PyAudioWPatch（pyaudiowpatch）——
sounddevice 官方**不支持** WasapiSettings(loopback=True)（0.3.7~0.5.5 均无
loopback 参数），旧实现每次回环都 TypeError 降级；PyAudioWPatch 把 WASAPI
输出设备的 [Loopback] 变体作为输入设备枚举，实机验证可稳定回环。

说明：
- VRChat 运行期间方可枚举到回环设备（A-01）。
- 输出全链路 16kHz mono float32 音频块（共享知识约定第 5 条）。
- set_muted(True) 期间丢弃音频块（自抑制静音窗口）。

输入来源（问题 2：麦克风 vs 电脑内录）：
- 普通麦克风：device 为麦克风名，_open_stream **不再硬编码 loopback**（旧实现
  对普通麦克风开回环会失败）；
- 电脑内录（WASAPI 回环）：device 为 `<输出设备名> [内录]`（Web 下拉框枚举条目
  的 name），运行时自动识别后缀 → 用 pyaudiowpatch 打开该输出设备的 [Loopback]
  变体（WASAPI 回环只能以设备默认采样率打开，回调内重采样到 16k），捕获系统
  输出（如 VRChat 里其他人的声音）。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import numpy as np

logger = logging.getLogger("vrchat_agent.audio.audio_input")

# 回环（电脑内录）设备名的统一后缀：Web 下拉框枚举条目 name 以它结尾，
# _open_stream 据此识别"该配置项是回环"并自动剥掉后缀还原真实输出设备名。
LOOPBACK_SUFFIX = " [内录]"
# PyAudioWPatch 为 WASAPI 输出设备生成的 loopback 变体后缀（枚举匹配用）。
_PYAUDIO_LOOPBACK_MARK = "[Loopback]"

# 全链路输出采样率（共享知识约定第 5 条）：回环设备默认采样率重采样到此。
_OUTPUT_RATE = 16000


class AudioInputError(RuntimeError):
    """音频输入设备错误。"""


class AudioInput:
    """音频捕获器：持续产出 16k mono float32 块并交给 sink 协程。

    支持两种输入来源（问题 2）：
    - 普通麦克风（默认）：device 为麦克风名/默认设备，sounddevice InputStream；
    - 电脑内录（WASAPI 回环）：loopback=True（或 device 以 LOOPBACK_SUFFIX 结尾），
      device 为输出设备名，用 pyaudiowpatch 打开其 [Loopback] 变体
      （sounddevice 不支持 loopback；保留 WasapiSettings 兼容路径作为降级）。
    """

    def __init__(
        self,
        device: str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = 30,
        channels: int = 1,
        loopback: bool | None = None,
        silence_threshold: float = 0.005,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.channels = channels
        # loopback 模式：True=强制回环 / False=强制普通 / None=自动识别（默认）。
        self.loopback = loopback
        # RMS 静音预筛阈值（归一化 0~1）：仅作日志/统计参考；实际丢弃在
        # SpeechSegmenter.feed()（RMS 命中 → 不进 VAD 但保留静音切分计数）。
        self.silence_threshold = float(silence_threshold)
        self._chunk_frames = int(sample_rate * chunk_ms / 1000)
        self._muted = False
        self._mute_lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._stream = None  # sounddevice.InputStream / pyaudiowpatch.Stream（惰性创建）
        self._pa = None  # pyaudiowpatch.PyAudio 实例（必须存活，防 GC 流失效）
        self._pa_module = None  # pyaudiowpatch 模块（回调用 paContinue）
        self._loopback_rate = 48000  # 回环设备默认采样率（回调重采样用）
        self._loopback_channels = 2  # 回环设备输入声道数（回调取第一声道）

    # ---- 自抑制 ----
    def set_muted(self, muted: bool) -> None:
        """设置静音窗口：为 True 时丢弃捕获到的音频块（自抑制）。"""
        with self._mute_lock:
            self._muted = bool(muted)

    def is_muted(self) -> bool:
        with self._mute_lock:
            return self._muted

    # ---- 主循环 ----
    async def run(self, sink) -> None:
        """持续捕获并调用 `sink(chunk: np.ndarray)`（16k mono float32，30ms 块）。"""
        loop = asyncio.get_running_loop()
        stream = self._open_stream()
        try:
            while True:
                try:
                    chunk = await loop.run_in_executor(None, lambda: self._queue.get(True, 0.5))
                except queue.Empty:
                    continue
                if chunk is None:  # stop() 结束标记
                    break
                if self.is_muted():
                    continue  # 自抑制：丢弃音频块
                await sink(chunk)
        finally:
            self._close_stream(stream)

    def _close_stream(self, stream) -> None:
        """关闭流并释放 pyaudiowpatch 实例（sounddevice 与 PyAudio 方法名不同，兼容）。"""
        try:
            if stream is not None:
                stop = getattr(stream, "stop", None) or getattr(stream, "stop_stream", None)
                if stop is not None:
                    stop()
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
        except Exception:
            pass
        pa = self._pa
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
            self._pa = None
            self._pa_module = None

    # ---- 设备 ----
    def _resolve_loopback(self) -> tuple[str | None, bool]:
        """解析设备配置 → (真实设备标识, 是否回环)。

        - loopback=True  → 强制回环：device 视为输出设备名（剥掉 [内录] 后缀）；
        - loopback=False → 强制普通：device 原样（麦克风名）；
        - loopback=None  → 自动识别：device 以 LOOPBACK_SUFFIX 结尾即为回环。
        """
        if self.loopback is True:
            name = self.device or ""
            if name.endswith(LOOPBACK_SUFFIX):
                name = name[: -len(LOOPBACK_SUFFIX)]
            return name or None, True
        if self.loopback is False:
            return self.device or None, False
        name = self.device or ""
        if name.endswith(LOOPBACK_SUFFIX):
            return name[: -len(LOOPBACK_SUFFIX)] or None, True
        return name or None, False

    def _open_stream(self):
        """打开输入流：按解析结果分派普通麦克风 / 电脑内录（问题 2）。"""
        device, is_loopback = self._resolve_loopback()
        if is_loopback:
            return self._open_loopback_stream(device)
        return self._open_mic_stream(device)

    @staticmethod
    def _resolve_mic_device(device: str | None):
        """把配置里的设备标识解析为 sounddevice 可用的 index 或 None（P0 重名修复）。

        兼容两种语义（config 层 device 保持 str）：
        - None / 空串        → None（系统默认设备）；
        - 数字字符串（"26"）→ int(index)（Web 下拉框新契约：value 用 index）；
        - 非数字（旧配置名） → sounddevice.query_devices() **精确匹配** name（完全
          相等，不子串模糊），取唯一 index；重名/找不到 → 抛 AudioInputError
          列出候选（含 index/hostapi），让用户去 Web 下拉框选，而不是崩溃。
        """
        if device is None:
            return None
        s = str(device).strip()
        if s == "":
            return None
        # 仅 ASCII 数字才按 index 解析：unicode 数字（如 '²'）isdigit()=True 但
        # int() 会抛未包装 ValueError，改为落入下方设备名精确匹配分支（包装 AudioInputError）。
        if s.isascii() and s.isdigit():
            return int(s)
        # 旧配置：设备名 → 精确匹配（区分 hostapi 取唯一 index）。
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioInputError("sounddevice 未安装，无法解析音频设备（pip install -r requirements.txt）") from exc
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as exc:
            raise AudioInputError(f"无法枚举音频设备以解析配置的设备名 {s!r}: {exc}") from exc
        matches: list[tuple[int, str, str]] = []
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "") or "")
            if name != s:
                continue
            ha_name = ""
            try:
                ha_idx = dev.get("hostapi")
                if ha_idx is not None and 0 <= int(ha_idx) < len(hostapis):
                    ha_name = str(hostapis[int(ha_idx)].get("name", "") or "")
            except (TypeError, ValueError):
                ha_name = ""
            matches.append((int(idx), name, ha_name))
        if len(matches) == 1:
            return matches[0][0]
        if len(matches) > 1:
            candidates = "；".join(f"index={i}（{name}，{ha or '未知 hostapi'}）" for i, name, ha in matches)
            raise AudioInputError(
                f"设备名 {s!r} 在多个 hostapi 下重名（{len(matches)} 个），无法自动选择：{candidates}。"
                "请在 Web 控制台「参数 → 音频输入 → 输入设备」下拉框中按具体设备（index）重新选择"
            )
        raise AudioInputError(
            f"找不到音频输入设备 {s!r}。请确认设备已连接，或到 Web 控制台重新选择设备（index）"
        )

    def _open_mic_stream(self, device: str | None):
        """普通麦克风：sounddevice InputStream，**不再硬编码 loopback**（问题 2 修复）。

        P0 修复：device 先经 _resolve_mic_device 解析为数字 index 再传给 sounddevice，
        避免"多 hostapi 同名设备名"被按名称模糊匹配导致 Multiple devices 崩溃。
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioInputError("sounddevice 未安装，无法捕获音频（pip install -r requirements.txt）") from exc

        # 设备语义归一化（BugFix P0）：device=None = 系统默认设备，device="" = 按名称
        # 模糊搜索（空串会匹配到多个设备 → ValueError 崩溃）。配置层 device 保持 str
        # （空串 = 默认设备，Web 下拉框契约），此处统一解析为 index / None。
        resolved = self._resolve_mic_device(device)
        try:
            stream = sd.InputStream(
                device=resolved,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self._chunk_frames,
                extra_settings=None,
                callback=self._callback,
            )
        except Exception as exc:
            raise AudioInputError(
                f"无法打开音频输入设备（{self.device!r}，index={resolved}）: {exc}。"
                "提示：若该设备仅支持特定采样率（如 WASAPI 虚拟麦克风仅 48k），"
                "请在 Web 控制台换一个支持 16k 的麦克风（index）"
            ) from exc
        stream.start()
        logger.info(
            "音频输入已启动[普通输入]: device=%s resolved=%s sr=%d chunk_ms=%d",
            self.device, resolved, self.sample_rate, self.chunk_ms,
        )
        return stream

    def _open_loopback_stream(self, device_name: str | None):
        """电脑内录（WASAPI 回环）。

        优先 pyaudiowpatch（真实支持 WASAPI loopback，实机验证可用）；
        pyaudiowpatch 明确失败（未安装/未找到设备）→ 直接报错并给出安装提示，
        不再静默降级成麦克风（用户明确选了"内录"，静默降级会录错源）；
        保留 sounddevice WasapiSettings(loopback=True) 兼容路径（未来支持的环境）。
        """
        try:
            stream = self._open_loopback_pyaudio(device_name)
            if stream is not None:
                return stream
        except AudioInputError:
            raise
        except Exception as exc:
            logger.warning("pyaudiowpatch 回环打开失败，尝试 sounddevice 兼容路径: %s", exc)
        # 降级：sounddevice WasapiSettings(loopback=True)（任务要求保留的写法）。
        try:
            return self._open_loopback_sounddevice(device_name)
        except AudioInputError:
            raise
        except Exception as exc:
            raise AudioInputError(
                f"无法打开 WASAPI 回环（{device_name or '默认输出'}）: {exc}。"
                "请确认已安装 pyaudiowpatch（pip install pyaudiowpatch）且 VRChat/系统正在播放音频"
            ) from exc

    def _open_loopback_pyaudio(self, device_name: str | None):
        """用 pyaudiowpatch 打开 WASAPI 回环（[Loopback] 变体）。

        未安装 pyaudiowpatch → 返回 None（交给 sounddevice 兼容路径）；
        已安装但找不到匹配设备 → 抛 AudioInputError。
        WASAPI 回环只能以设备默认采样率打开（16000 会 Invalid sample rate），
        回调内重采样到 self.sample_rate（16k）。
        """
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            return None
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            target = (device_name or "").strip()
            lb_index = None
            dev_rate = 48000
            dev_channels = 2
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = str(info.get("name", "") or "")
                if _PYAUDIO_LOOPBACK_MARK in name:
                    if not target or name.replace(_PYAUDIO_LOOPBACK_MARK, "").strip() == target:
                        lb_index = int(i)
                        dev_rate = int(info.get("defaultSampleRate", 0) or 48000)
                        dev_channels = int(info.get("maxInputChannels", 0) or 2)
                        break
            if lb_index is None:
                raise AudioInputError(f"未找到 WASAPI 回环设备（{target or '默认输出'}）")
            self._loopback_rate = dev_rate
            self._loopback_channels = dev_channels
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=dev_channels,
                rate=dev_rate,
                input=True,
                input_device_index=lb_index,
                frames_per_buffer=int(dev_rate * self.chunk_ms / 1000),
                stream_callback=self._pyaudio_callback,
            )
            stream.start_stream()
            self._pa = pa
            self._pa_module = pyaudio
            logger.info(
                "音频输入已启动[电脑内录]: device=%s rate=%d(回调重采样→%d) chunk_ms=%d",
                self.device,
                dev_rate,
                self.sample_rate,
                self.chunk_ms,
            )
            return stream
        except AudioInputError:
            raise
        except Exception as exc:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
            raise AudioInputError(f"pyaudiowpatch 回环打开失败: {exc}") from exc

    def _open_loopback_sounddevice(self, device_name: str | None):
        """sounddevice WasapiSettings(loopback=True) 兼容路径（任务要求写法）。

        当前 sounddevice 官方版不支持 loopback 参数（会 TypeError 被上层转成
        AudioInputError）；保留此路径以兼容未来支持 loopback 的环境/社区 fork。
        """
        import sounddevice as sd
        from sounddevice import WasapiSettings

        extra = WasapiSettings(loopback=True)
        stream = sd.InputStream(
            device=device_name,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self._chunk_frames,
            extra_settings=extra,
            callback=self._callback,
        )
        stream.start()
        logger.info("音频输入已启动[电脑内录-sounddevice]: device=%s", self.device)
        return stream

    def _callback(self, indata, frames, time_info, status) -> None:
        """sounddevice 回调（音频线程）：压入线程安全队列。"""
        if status:
            logger.debug("音频状态: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 and indata.shape[1] > 1 else indata.ravel()
        # P0 数据完整性：.copy() 强制拷贝——np.ascontiguousarray 对已连续数组只返回
        # 视图不拷贝；PortAudio 回调返回后会复用 indata 缓冲区，若入队的是视图，队列
        # 内所有块共享同一内存 → 延迟读取得到被覆盖的垃圾值（RMS=inf 绕过静音预筛）。
        chunk = np.ascontiguousarray(mono, dtype=np.float32).copy()
        self._put_chunk(chunk)

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """pyaudiowpatch 回调：int16 bytes → 取第一声道 → 重采样到 16k → 入队。"""
        if status:
            logger.debug("音频状态: %s", status)
        arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        dev_channels = max(1, int(self._loopback_channels))
        mono = arr[0::dev_channels] if dev_channels > 1 else arr
        mono = self._resample_to_16k(mono, int(self._loopback_rate))
        self._put_chunk(np.ascontiguousarray(mono, dtype=np.float32))
        return (None, self._pa_module.paContinue) if self._pa_module is not None else None

    def _put_chunk(self, chunk: np.ndarray) -> None:
        """压入队列；积压时丢弃最旧一帧，保证实时性。"""
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    @staticmethod
    def _resample_to_16k(samples: np.ndarray, in_rate: int) -> np.ndarray:
        """把回环音频重采样到 16k（WASAPI loopback 只能以设备默认采样率打开）。

        - 整数比（如 48k→16k，3:1）：分块均值（低通 + 抽取），简单且对语音足够；
        - 非整数比（如 44.1k→16k）：np.interp 线性插值。
        """
        if in_rate == _OUTPUT_RATE:
            return samples
        if in_rate % _OUTPUT_RATE == 0:
            factor = in_rate // _OUTPUT_RATE
            n = len(samples) - (len(samples) % factor)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            return samples[:n].reshape(-1, factor).mean(axis=1).astype(np.float32)
        n_out = int(round(len(samples) * _OUTPUT_RATE / in_rate))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, len(samples), len(samples), endpoint=False)
        x_new = np.linspace(0.0, len(samples), n_out, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    def stop(self) -> None:
        """请求停止（向队列投递结束标记）。"""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    @staticmethod
    def list_loopback_devices() -> list[dict]:
        """列出可作为 WASAPI 回环（电脑内录）的 [Loopback] 变体设备（调试用，[Win]）。

        用 pyaudiowpatch 枚举（sounddevice 官方不支持 loopback）。
        返回 [{index, name, default_samplerate, max_input_channels}]——这些是输入
        设备，直接用其 index 打开即可内录对应输出（与 web/audio_devices 的
        [内录] 条目同一数据源）。
        """
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise AudioInputError("pyaudiowpatch 未安装，无法枚举回环设备（pip install pyaudiowpatch）") from exc
        pa = None
        try:
            pa = pyaudio.PyAudio()
            loopbacks = []
            for idx in range(pa.get_device_count()):
                try:
                    dev = pa.get_device_info_by_index(idx)
                    name = str(dev.get("name", "") or "")
                    if int(dev.get("maxInputChannels", 0) or 0) > 0 and _PYAUDIO_LOOPBACK_MARK in name:
                        loopbacks.append(
                            {
                                "index": idx,
                                "name": name,
                                "default_samplerate": dev.get("defaultSampleRate"),
                                "max_input_channels": dev.get("maxInputChannels"),
                            }
                        )
                except Exception:
                    continue
            return loopbacks
        except Exception as exc:
            raise AudioInputError(f"枚举回环设备失败: {exc}") from exc
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
