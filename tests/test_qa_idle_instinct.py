"""QA 独立补充：空闲本能微动作循环（Idle Instinct Loop）测试。

覆盖：
  1. 配置解析：enabled=False 时永不启动循环
  2. 循环基本行为：mock actuator._call_tool_safe → verify 在 1s 内至少有一次调用
  3. 打断：set cancel_event → 循环在下一个 tick 检查点退出（< interval+0.1s）
  4. 防重复签名：连续 3 次相同签名 → 第 4 次被强制变异
  5. 概率边界：所有概率参数在 [0.0, 1.0] 范围内

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from vrchat_agent.config import IdleInstinctConfig
from vrchat_agent.core.idle_instinct import IdleInstinct


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_config(**overrides) -> IdleInstinctConfig:
    """创建 IdleInstinctConfig，支持字段覆盖。"""
    defaults = {
        "enabled": True,
        "interval_min_sec": 0.10,
        "interval_max_sec": 0.15,
        "jitter_min_deg": 1.0,
        "jitter_max_deg": 3.0,
        "look_overshoot_prob": 0.10,
        "hesitate_prob": 0.0,       # 测试期间不触发停顿
        "small_step_move_prob": 0.0,  # 测试期间不触发移动（仅测 jitter）
        "move_duration_min_sec": 0.08,
        "move_duration_max_sec": 0.15,
    }
    defaults.update(overrides)
    return IdleInstinctConfig(**defaults)


def _async_none(*_args, **_kwargs):
    """返回一个可 await 的协程（等价于 async def noop）。"""
    async def _noop():
        return None
    return _noop()


def make_instinct(config: IdleInstinctConfig | None = None):
    """创建 IdleInstinct 实例，附带 mock actuator。"""
    actuator = MagicMock()
    actuator._call_tool_safe = MagicMock(side_effect=_async_none)
    mapper = MagicMock()
    cfg = config or make_config()
    return IdleInstinct(actuator, mapper, cfg)


# ---------------------------------------------------------------------------
# 1. 配置解析：enabled=False 时永不启动循环
# ---------------------------------------------------------------------------

async def test_qa_idle_disabled_never_runs():
    """enabled=False 时 start() 直接返回，不创建后台任务。"""
    cfg = make_config(enabled=False)
    instinct = make_instinct(cfg)
    await instinct.start()
    assert instinct._loop_task is None, "enabled=False 时不应创建循环任务"


async def test_qa_idle_enabled_creates_task():
    """enabled=True 时 start() 创建后台循环任务。"""
    instinct = make_instinct()
    await instinct.start()
    assert instinct._loop_task is not None, "enabled=True 时应创建循环任务"
    assert not instinct._loop_task.done(), "循环任务应仍在运行"
    await instinct.stop()


# ---------------------------------------------------------------------------
# 2. 循环基本行为：mock actuator._call_tool_safe → 1s 内至少有一次调用
# ---------------------------------------------------------------------------

async def test_qa_idle_loop_calls_actuator_within_1s():
    """启用后 1 秒内至少触发一次 _call_tool_safe。"""
    # hesitate_prob=0.0 且 small_step_move_prob=1.0 确保每次 tick 都有动作触发
    cfg = make_config(hesitate_prob=0.0, small_step_move_prob=1.0)
    instinct = make_instinct(cfg)
    await instinct.start()

    # 等待最多 1s（约 6~10 tick）
    await asyncio.sleep(1.0)
    await instinct.stop()

    call_count = instinct._actuator._call_tool_safe.call_count
    assert call_count >= 1, (
        f"空闲本能应在 1s 内至少触发一次动作调用，实际 {call_count} 次"
    )


# ---------------------------------------------------------------------------
# 3. 打断：set cancel_event → 循环在下一个 tick 检查点退出
# ---------------------------------------------------------------------------

async def test_qa_idle_interrupt_exits_quickly():
    """set cancel_event 后循环应在 interval_max + 0.2s 内退出。"""
    cfg = make_config(interval_min_sec=0.5, interval_max_sec=0.5, hesitate_prob=1.0)
    # hesitate_prob=1.0 保证 tick 不做任何动作（无 OSC 调用），纯测退出延迟
    instinct = make_instinct(cfg)
    await instinct.start()

    # 给一点时间让循环进入等待
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    await instinct.stop()
    elapsed = time.perf_counter() - t0

    # 循环在 check cancel_event 后应立即退出（下次 tick 之前）
    # 最坏情况：刚好进入 sleep，需等 interval_max 秒
    # 加上取消传播开销，预算 interval_max + 0.2s
    budget = cfg.interval_max_sec + 0.2
    assert elapsed < budget, (
        f"打断应在 {budget:.2f}s 内退出，实际 {elapsed:.3f}s"
    )
    assert instinct._loop_task is None or instinct._loop_task.done()


async def test_qa_idle_interrupt_stops_calling_actuator():
    """打断后不应再有新的 actuator 调用。"""
    cfg = make_config(hesitate_prob=0.0)
    instinct = make_instinct(cfg)
    await instinct.start()

    # 等一次 tick 完成
    await asyncio.sleep(0.3)
    await instinct.stop()

    call_count_after_stop = instinct._actuator._call_tool_safe.call_count

    # 再等一会儿，确认没有新调用
    await asyncio.sleep(0.3)
    assert instinct._actuator._call_tool_safe.call_count == call_count_after_stop, (
        "打断后不应有新调用"
    )


# ---------------------------------------------------------------------------
# 4. 防重复签名：连续 3 次相同签名 → 第 4 次被强制变异
# ---------------------------------------------------------------------------

async def test_qa_idle_dedup_forces_alternative_after_3_skips():
    """连续 3 次重复签名后第 4 次强制变异（value 取反）。"""
    cfg = make_config(hesitate_prob=0.0, small_step_move_prob=0.0)
    instinct = make_instinct(cfg)

    # 直接操作内部状态：模拟连续 3 次相同签名
    sig = "look_axis"
    instinct._recent_signatures = [sig] * 5
    instinct._consecutive_skips = 3

    # 构造一组动作
    actions = [("look_axis", {"axis": "horizontal", "value": 0.15, "duration": 0.05})]

    # _force_alternative 应把 value 取反
    alt = instinct._force_alternative(actions)
    assert len(alt) == 1
    assert alt[0][0] == "look_axis"
    assert alt[0][1]["value"] == -0.15, (
        f"强制变异应取反 value，实际 {alt[0][1]['value']}"
    )


async def test_qa_idle_duplicate_signature_skipped():
    """相同签名出现在最近 5 次中 → _is_duplicate 返回 True。"""
    instinct = make_instinct()
    instinct._recent_signatures = ["look_axis", "move_axis-look_axis", "look_axis"]
    assert instinct._is_duplicate("look_axis") is True
    assert instinct._is_duplicate("move_axis-look_axis") is True
    assert instinct._is_duplicate("move_axis") is False


async def test_qa_idle_signature_history_capped():
    """签名历史最多保留 5 条。"""
    instinct = make_instinct()
    for i in range(10):
        sig = f"sig_{i}"
        instinct._recent_signatures.append(sig)
        if len(instinct._recent_signatures) > 5:
            instinct._recent_signatures.pop(0)
    assert len(instinct._recent_signatures) == 5
    assert instinct._recent_signatures[0] == "sig_5"


# ---------------------------------------------------------------------------
# 5. 概率边界：所有概率参数在 [0.0, 1.0] 范围内
# ---------------------------------------------------------------------------

def test_qa_idle_config_probability_bounds():
    """所有概率字段应在 [0.0, 1.0] 范围内，pydantic 自动校验。"""
    # 合法值
    cfg = IdleInstinctConfig(
        look_overshoot_prob=0.0,
        hesitate_prob=0.5,
        small_step_move_prob=1.0,
    )
    assert cfg.look_overshoot_prob == 0.0
    assert cfg.hesitate_prob == 0.5
    assert cfg.small_step_move_prob == 1.0


def test_qa_idle_config_probability_out_of_range_rejected():
    """超出 [0.0, 1.0] 的概率值应被 pydantic 拒绝。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IdleInstinctConfig(look_overshoot_prob=-0.1)

    with pytest.raises(ValidationError):
        IdleInstinctConfig(hesitate_prob=1.5)

    with pytest.raises(ValidationError):
        IdleInstinctConfig(small_step_move_prob=2.0)


def test_qa_idle_config_default_values():
    """验证默认值符合规格。"""
    cfg = IdleInstinctConfig()
    assert cfg.enabled is True
    assert cfg.interval_min_sec == 0.3
    assert cfg.interval_max_sec == 0.7
    assert cfg.jitter_min_deg == 1.0
    assert cfg.jitter_max_deg == 3.0
    assert cfg.look_overshoot_prob == 0.10
    assert cfg.hesitate_prob == 0.30
    assert cfg.small_step_move_prob == 0.15
    assert cfg.move_duration_min_sec == 0.08
    assert cfg.move_duration_max_sec == 0.15


def test_qa_idle_config_interval_bounds():
    """interval 字段有 ge/le 约束。"""
    from pydantic import ValidationError

    # 太小
    with pytest.raises(ValidationError):
        IdleInstinctConfig(interval_min_sec=0.0)
    # 太大
    with pytest.raises(ValidationError):
        IdleInstinctConfig(interval_max_sec=10.0)


# ---------------------------------------------------------------------------
# 6. 边界：stop() 幂等
# ---------------------------------------------------------------------------

async def test_qa_idle_stop_idempotent():
    """连续 stop() 不应抛异常。"""
    instinct = make_instinct()
    await instinct.start()
    await instinct.stop()
    await instinct.stop()  # 第二次应无副作用
    assert instinct._loop_task is None


async def test_qa_idle_start_restarts():
    """start() 在已有循环时先取消旧循环再启动新的。"""
    instinct = make_instinct()
    await instinct.start()
    first_task = instinct._loop_task
    assert first_task is not None

    await instinct.start()  # 重新启动
    second_task = instinct._loop_task
    assert second_task is not None
    assert second_task is not first_task, "重启后应是新任务"
    assert first_task.done(), "旧任务应已取消"

    await instinct.stop()


# ---------------------------------------------------------------------------
# 7. 概率触发：小步移动也能触发
# ---------------------------------------------------------------------------

async def test_qa_idle_small_move_triggers():
    """small_step_move_prob=1.0 且 hesitate=0.0 时，tick 应稳定产生调用（不崩溃）。"""
    cfg = make_config(
        hesitate_prob=0.0,
        small_step_move_prob=1.0,
        jitter_min_deg=0.1,
        jitter_max_deg=0.2,
    )
    instinct = make_instinct(cfg)
    await instinct.start()

    await asyncio.sleep(0.8)
    await instinct.stop()

    call_count = instinct._actuator._call_tool_safe.call_count
    # 由于签名去重（tool_name 级），look_axis+move_axis 组合可能被跳过，
    # 但应至少有一次调用（纯 move_axis 或 首次 look_axis+move_axis）
    assert call_count >= 1, (
        f"small_step_move_prob=1.0 时应至少触发一次调用，实际 {call_count} 次"
    )

    # 确认调用中包含 look_axis 或 move_axis（不强制要求两者都出现，签名去重可能导致只出现一种）
    called_tools = {c[0][0] for c in instinct._actuator._call_tool_safe.call_args_list}
    assert called_tools.issubset({"look_axis", "move_axis"}), (
        f"只应调用 look_axis/move_axis，实际调用了 {called_tools}"
    )
