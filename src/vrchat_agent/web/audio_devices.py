"""音频设备枚举（Web 控制台设备下拉框数据源，W 增量 + 问题 2）。

- GET /api/audio/devices 数据源：sounddevice.query_devices() 枚举系统设备，
  按 输入/输出 分类，标注 hostapi 名称（枚举逻辑与参考项目 vrc_chat_translator
  /voice.py 的 list_input_devices 一致，仅扩展为输入+输出双向）。
- 问题 2（电脑内录）：除普通麦克风外，为每个 **WASAPI 输出设备**生成一条
  `[内录]`（loopback）条目放进 input 列表——WASAPI 回环 = 内录该系统输出的声音
  （如 VRChat 里其他人的声音）。条目带 `loopback: True` / `kind: "loopback"` /
  `device_index`（关联输出设备 index），前端据此分组展示。
- P0 重名修复：同 name 多 hostapi（DirectSound/WASAPI/WDM-KS 同名麦克风）的
  条目，展示 name 加 ` [hostapi]` 后缀区分，并新增 `raw_name`（原始名）供前端
  兼容旧配置（存的设备名）；选中值始终用 **index**（前端 value=index）。
- 需求 2（16k 兼容屏蔽）：对每个普通输入/输出设备**实测 16k 兼容性**（尝试
  InputStream/OutputStream(device=i, samplerate=16000, channels=1) 打开即关），
  条目加 `supports_16k: bool`——WASAPI/WDM-KS 只认 48k 或打开失败 → False，
  MME/DirectSound 支持 16k → True；前端据此屏蔽不支持 16k 的设备，避免启动崩溃
  或无声。`[内录]`（loopback）回环按默认采样率打开 + 回调重采样（见
  audio/audio_input.py），不受 16k 限制，恒标记 supports_16k=true。
- 优雅降级：sounddevice 未安装 / 无可用声卡 / CI 环境 → 返回空列表，不崩溃；
  非 Windows 或无 WASAPI hostapi → 不生成 loopback 条目（前端拿到空列表时回退
  为文本编辑框）。
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from ..audio.audio_input import LOOPBACK_SUFFIX

logger = logging.getLogger("vrchat_agent.web.audio_devices")

# 惰性导入：sounddevice 属平台相关依赖，缺失时 _sd=None，
# list_audio_devices() 返回空列表（端点保持可用，前端回退编辑框）。
try:
    import sounddevice as _sd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - 依赖缺失路径
    _sd = None

# WASAPI 回环（电脑内录）用 PyAudioWPatch：sounddevice 官方不支持
# WasapiSettings(loopback=True)（实机验证 0.5.5 无此参数），pyaudiowpatch 把
# WASAPI 输出设备的 [Loopback] 变体作为输入设备枚举，可稳定回环。
try:
    import pyaudiowpatch as _pya  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - 依赖缺失路径
    _pya = None


def probe_supports_16k(index: int, is_input: bool, sd=None) -> bool:
    """实测单个设备是否支持 16k mono float32（需求 2：屏蔽不支持 16k 的设备）。

    对设备 index 尝试打开 `InputStream/OutputStream(device=i, samplerate=16000,
    channels=1, dtype='float32')` 并立即关闭（快测，不启动流不采声）：
    - 打开成功 → True（MME/DirectSound 麦克风/扬声器实测支持 16k）；
    - 打开抛异常（Invalid sample rate / Invalid number of channels）→ False
      （WASAPI 只认 48k、WDM-KS 独占失败等，正是启动崩溃/无声根因）；
    - 设备忙（-9984 paDeviceUnavailable / -9985 paInUse，如本进程麦克风已打开）
      → True（保守放行：无法验证但不该把"正在被占用"误判为"不支持 16k"）。
    - sd 不可用（未安装/CI）→ True（保守放行：此时枚举本就返回空列表）。
    测试可注入 fake sd（无回调 InputStream/OutputStream）。
    """
    if sd is None:
        return True
    try:
        if is_input:
            stream = sd.InputStream(device=index, samplerate=16000, channels=1, dtype="float32")
        else:
            stream = sd.OutputStream(device=index, samplerate=16000, channels=1, dtype="float32")
    except Exception as exc:
        raw = str(exc)
        # 忙/占用 ≠ 不支持 16k：设备已打开（如本进程麦克风）时 PortAudio 返回
        # Device unavailable / In use，此时无法实测 → 保守放行（避免误屏蔽）。
        if "-9984" in raw or "-9985" in raw or "Device unavailable" in raw or "In use" in raw:
            logger.debug("设备 index=%s 忙（无法实测 16k，保守放行）: %s", index, raw)
            return True
        logger.debug(
            "设备 index=%s %s 不支持 16k: %s", index, "输入" if is_input else "输出", raw
        )
        return False
    try:
        stream.close()
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# A-25：设备枚举结果 TTL 缓存（5s）。list_audio_devices() 会对每个设备开流实测
# 16k（probe_supports_16k，开销大），而设备插拔频率远低于 Web 查询频率；
# 缓存 5s 可接受（热插拔容忍：5s 内新插设备不立即出现，任务约定）。线程安全
# （uvicorn 多请求并发）。测试可用 _clear_devices_cache() 复位 / ttl<=0 绕过。
# ---------------------------------------------------------------------------
_DEVICES_CACHE_TTL = 5.0
_devices_cache_lock = threading.Lock()
_devices_cache_ts = 0.0
_devices_cache_value: dict[str, list[dict]] = {"input": [], "output": []}


def cached_list_audio_devices(ttl: float = _DEVICES_CACHE_TTL) -> dict[str, list[dict]]:
    """带 TTL 缓存的设备枚举（Web GET /api/audio/devices 数据源，A-25）。

    - ttl<=0 → 每次重新实测（测试/调试用）；
    - 命中缓存返回新容器（内层条目为新建 dict，调用方修改返回值不影响缓存）；
    - 未命中则调用 list_audio_devices() 实测并填充缓存。
    """
    global _devices_cache_ts, _devices_cache_value
    if ttl > 0:
        now = time.monotonic()
        with _devices_cache_lock:
            if now - _devices_cache_ts < ttl:
                return {
                    "input": list(_devices_cache_value["input"]),
                    "output": list(_devices_cache_value["output"]),
                }
    value = list_audio_devices()
    if ttl > 0:
        with _devices_cache_lock:
            _devices_cache_ts = time.monotonic()
            _devices_cache_value = value
    return value


def _clear_devices_cache() -> None:
    """清空设备枚举缓存（测试隔离：模拟设备插拔立即可见）。"""
    global _devices_cache_ts
    with _devices_cache_lock:
        _devices_cache_ts = 0.0


def list_audio_devices() -> dict[str, list[dict]]:
    """枚举系统音频设备，返回 {"input": [...], "output": [...]}。

    普通输入设备条目：{index, name, channels, samplerate, hostapi, supports_16k}
    （6 字段契约）；loopback（[内录]）条目额外带 {loopback: True, kind: "loopback",
    device_index, supports_16k: True}，放进 input 列表，供前端分组为
    「麦克风 / 电脑内录（回环）」。
    """
    result: dict[str, list[dict]] = {"input": [], "output": []}
    if _sd is None:
        logger.debug("sounddevice 不可用，音频设备枚举返回空列表")
        return result
    try:
        devices = _sd.query_devices()
        hostapis = _sd.query_hostapis()
    except Exception as exc:
        logger.warning("枚举音频设备失败（返回空列表）: %s", exc)
        return result

    for index, device in enumerate(devices):
        try:
            in_ch = int(device.get("max_input_channels", 0) or 0)
            out_ch = int(device.get("max_output_channels", 0) or 0)
            if in_ch <= 0 and out_ch <= 0:
                continue
            ha_index = device.get("hostapi")
            ha_name = ""
            if ha_index is not None and 0 <= int(ha_index) < len(hostapis):
                ha_name = str(hostapis[int(ha_index)].get("name", "") or "")
            info = {
                "index": int(index),
                "name": str(device.get("name", "") or ""),
                "channels": max(in_ch, out_ch),
                "samplerate": int(device.get("default_samplerate", 0) or 0),
                "hostapi": ha_name,
                # 需求 2：实测 16k 兼容性（MME/DirectSound=true，WASAPI/WDM-KS=false）
                "supports_16k": probe_supports_16k(int(index), is_input=in_ch > 0, sd=_sd),
            }
            if in_ch > 0:
                result["input"].append(info)
            if out_ch > 0:
                result["output"].append(info)
        except Exception as exc:
            logger.debug("跳过异常设备条目 index=%s: %s", index, exc)
            continue

    # P0 重名修复：同 name 多 hostapi（如 DirectSound/WASAPI/WDM-KS 同名麦克风）
    # 的条目，name 加 [hostapi] 后缀区分展示（value 始终用 index，见前端）。
    _disambiguate_names(result["input"])
    _disambiguate_names(result["output"])

    # 问题 2：WASAPI 回环（电脑内录）——仅 Windows + WASAPI hostapi 可用时生成，
    # 为每个 WASAPI 输出设备生成一条 [内录] 输入条目。
    result["input"].extend(_list_loopback_entries(devices, hostapis))
    return result


def _disambiguate_names(devices: list[dict]) -> None:
    """原地更新普通设备条目：同 name 多 hostapi → name 加 ` [hostapi]` 后缀；补 raw_name。

    - 前端展示用 name（重名时区分 hostapi），选中值始终用 index（P0 修复）；
    - raw_name 保留原始设备名（不含后缀），供前端/运行时匹配旧配置（存的设备名）；
    - loopback（[内录]）条目不处理：其 name 必须以 LOOPBACK_SUFFIX 结尾，
      运行时靠后缀识别回环（剥后缀后匹配 [Loopback] 变体），加后缀会破坏契约。
    """
    counts: dict[str, int] = {}
    for d in devices:
        if d.get("kind") == "loopback":
            continue
        counts[d["name"]] = counts.get(d["name"], 0) + 1
    for d in devices:
        if d.get("kind") == "loopback":
            continue
        d["raw_name"] = d["name"]
        if counts.get(d["name"], 0) > 1 and d.get("hostapi"):
            d["name"] = f"{d['name']} [{d['hostapi']}]"


def _list_loopback_entries(devices, hostapis) -> list[dict]:
    """为每个 WASAPI 输出设备生成 loopback（[内录]）输入条目。

    生成条件（优雅降级）：
    - 仅 Windows（sys.platform 以 win 开头）；
    - 优先 pyaudiowpatch：枚举其 [Loopback] 变体输入设备（真实支持 WASAPI 回环）；
    - 降级 sounddevice 路径：系统存在 WASAPI hostapi 时，为每个 WASAPI 输出
      设备生成条目（未来支持 WasapiSettings(loopback=True) 的环境可用）。

    条目字段：{index, name: "<输出设备名> [内录]", channels, samplerate, hostapi,
    kind: "loopback", loopback: True, device_index, supports_16k: True}——运行时用
    name（剥掉后缀）在 pyaudiowpatch 中匹配 [Loopback] 变体打开。
    """
    if not sys.platform.startswith("win"):
        return []
    # 优先 pyaudiowpatch（真实可用，实机验证）
    entries = _list_loopback_entries_pyaudio()
    if entries:
        return entries
    # 降级：sounddevice WASAPI hostapi 路径（任务要求保留的兼容写法）
    wasapi_indices = {
        i
        for i, ha in enumerate(hostapis)
        if "wasapi" in str(ha.get("name", "") or "").lower()
    }
    if not wasapi_indices:
        return []
    entries = []
    for index, device in enumerate(devices):
        try:
            out_ch = int(device.get("max_output_channels", 0) or 0)
            if out_ch <= 0:
                continue
            ha_index = device.get("hostapi")
            if ha_index is None or int(ha_index) not in wasapi_indices:
                continue
            name = str(device.get("name", "") or "")
            if not name:
                continue
            entries.append(
                {
                    "index": int(index),
                    "name": f"{name}{LOOPBACK_SUFFIX}",
                    "channels": out_ch,
                    "samplerate": int(device.get("default_samplerate", 0) or 0),
                    "hostapi": "WASAPI",
                    "kind": "loopback",
                    "loopback": True,
                    "device_index": int(index),
                    # 回环按默认采样率打开 + 回调重采样到 16k，不受 16k 限制
                    "supports_16k": True,
                }
            )
        except Exception as exc:
            logger.debug("跳过异常回环设备条目 index=%s: %s", index, exc)
            continue
    return entries


def _list_loopback_entries_pyaudio() -> list[dict]:
    """用 pyaudiowpatch 枚举 [Loopback] 变体（真实 WASAPI 回环，[Win]）。

    PyAudioWPatch 把每个 WASAPI 输出设备的 loopback 变体作为输入设备（name 含
    "[Loopback]"）追加到设备列表末尾；条目 name 还原为 "<输出设备名> [内录]"，
    device_index 为该变体在 pyaudiowpatch 设备列表中的 index（打开时直接用）。
    """
    if _pya is None:
        return []
    pa = None
    entries: list[dict] = []
    try:
        pa = _pya.PyAudio()
    except Exception as exc:
        logger.debug("pyaudiowpatch 初始化失败（跳过回环枚举）: %s", exc)
        return entries
    try:
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                name = str(info.get("name", "") or "")
                in_ch = int(info.get("maxInputChannels", 0) or 0)
                if in_ch <= 0 or "[Loopback]" not in name:
                    continue
                out_name = name.replace("[Loopback]", "").strip()
                if not out_name:
                    continue
                entries.append(
                    {
                        "index": int(i),
                        "name": f"{out_name}{LOOPBACK_SUFFIX}",
                        "channels": in_ch,
                        "samplerate": int(info.get("defaultSampleRate", 0) or 0),
                        "hostapi": "WASAPI",
                        "kind": "loopback",
                        "loopback": True,
                        "device_index": int(i),
                        # 回环按默认采样率打开 + 回调重采样到 16k，不受 16k 限制
                        "supports_16k": True,
                    }
                )
            except Exception as exc:
                logger.debug("跳过异常回环条目 index=%s: %s", i, exc)
                continue
    finally:
        try:
            pa.terminate()
        except Exception:
            pass
    return entries
