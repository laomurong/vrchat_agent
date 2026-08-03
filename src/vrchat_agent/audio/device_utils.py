"""音频设备标识解析公共工具（输入/输出两侧对称，P0 重名修复）。

输入侧 AudioInput._resolve_mic_device 保持自有实现（错误文案被既有测试锁定）；
输出侧 AudioOutput 使用本模块的 resolve_output_device_index（同一套语义）：

- None / 空串        → None（系统默认设备，sounddevice 语义）；
- ASCII 数字字符串    → int(index)（Web 下拉框新契约：value 用 index）；
- 非数字（旧配置名） → sounddevice.query_devices() **精确匹配** name（完全相等，
  不子串模糊），取唯一 index；重名 / 找不到 → 抛 AudioOutputError 列出候选
  （含 VB-Audio 缺失提示），而不是让 sounddevice 按名称模糊搜索崩溃；
- sounddevice 无法枚举（测试注入 fake 缺 query_devices）→ 原样返回旧行为。

背景（QA 遗留隐患）：输入侧已改为 index 解析，输出侧仍"名字直传"——配置里的
数字字符串（如 "15"）会被 sounddevice 当作**设备名**做子串搜索（不是 index！），
多 hostapi 重名或找不到时抛 ValueError，导致 TTS 播放链路整体无声。本模块修复
输出侧，使字符串 index / 设备名 / 空串三种语义与输入侧完全一致。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vrchat_agent.audio.device_utils")

# VB-Audio Virtual Cable 输出端设备名特征（找不到设备时给出安装/选择提示）。
_VB_AUDIO_HINT = (
    "；提示：若目标是 VB-Audio 虚拟声卡（TTS → VRChat 麦克风），请先安装 "
    "VB-Audio Virtual Cable（安装后设备名形如 'CABLE Input (VB-Audio Virtual Cable)'），"
    "再到 Web 控制台「参数 → 音频输出 → 输出设备」下拉框中按具体设备（index）重新选择"
)


class DeviceResolveError(RuntimeError):
    """设备标识解析错误（找不到 / 重名 / 枚举失败）。"""


def resolve_output_device_index(device: str | None) -> int | str | None:
    """把输出侧配置里的设备标识解析为 sounddevice 可用参数（P0 修复）。

    返回：
    - None（默认设备）；
    - int index（数字字符串 / 精确匹配的设备名）；
    - 原样字符串（sounddevice 无法枚举时保持旧行为，仅测试 fake 场景）。

    抛 DeviceResolveError：设备名找不到（含 VB-Audio 提示）或重名（列候选）。
    """
    if device is None:
        return None
    s = str(device).strip()
    if s == "":
        return None
    # 仅 ASCII 数字才按 index 解析（与输入侧一致）：unicode 数字落入设备名匹配。
    if s.isascii() and s.isdigit():
        return int(s)
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise DeviceResolveError(
            "sounddevice 未安装，无法解析音频输出设备（pip install -r requirements.txt）"
        ) from exc
    if not hasattr(sd, "query_devices"):
        # 测试注入的 fake sounddevice 无枚举能力 → 原样直传（保持旧行为）。
        return s
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise DeviceResolveError(f"无法枚举音频设备以解析输出设备名 {s!r}: {exc}") from exc

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
    candidates = "；".join(
        f"index={i}（{name}，{ha or '未知 hostapi'}）" for i, name, ha in matches
    )
    if len(matches) > 1:
        raise DeviceResolveError(
            f"输出设备名 {s!r} 在多个 hostapi 下重名（{len(matches)} 个），无法自动选择：{candidates}。"
            "请在 Web 控制台「参数 → 音频输出 → 输出设备」下拉框中按具体设备（index）重新选择"
        )
    raise DeviceResolveError(
        f"找不到音频输出设备 {s!r}{_VB_AUDIO_HINT}。请在 Web 控制台重新选择输出设备（index）"
    )
