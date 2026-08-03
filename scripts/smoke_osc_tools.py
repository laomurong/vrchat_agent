"""OSC 桥接冒烟：拉起真实 server.js，列出实际工具名并与映射表对齐（问题 2 验证）。

用法（conda 环境 vrchat-agent，需 node 可用；config.yaml 的 action.bridge 指向真实 server.js）：
    conda activate vrchat-agent
    python scripts/smoke_osc_tools.py [--config config.yaml]

输出：
    1) server.js 实际暴露的 MCP 工具名（MCPClient.list_tools）
    2) mappings/avatar_default.yaml 使用的工具名集合
    3) 对齐结论：映射表全部工具名 ∈ 真实工具名集合 → 退出码 0；否则退出码 1；
       拉起失败退出码 2。

不依赖 VRChat 是否开启 OSC：list_tools 只与 server.js 进程通信。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from vrchat_agent.bridge.mcp_client import MCPClient  # noqa: E402
from vrchat_agent.config import Config  # noqa: E402


def load_mapping_tools(mapping_path: Path) -> set[str]:
    """读取映射表使用的全部工具名。"""
    with open(mapping_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    actions = data.get("actions", data) or {}
    tools: set[str] = set()
    for key, entry in actions.items():
        if not isinstance(entry, dict):
            continue
        items = entry.get("tools") if isinstance(entry.get("tools"), list) else [entry]
        for item in items:
            if isinstance(item, dict) and item.get("tool"):
                tools.add(str(item["tool"]))
    return tools


async def run(config: Config) -> int:
    bridge = config.action.bridge
    client = MCPClient(command=bridge.command, args=bridge.args, timeout=bridge.timeout)
    print(f"[1/3] 拉起桥接: {bridge.command} {' '.join(bridge.args)}")
    try:
        await client.start()
    except Exception as exc:
        print(f"[FAIL] 拉起 server.js 失败: {exc}", file=sys.stderr)
        return 2
    try:
        real_tools = await client.list_tools()
    finally:
        await client.stop()

    real_set = set(real_tools)
    mapping_path = PROJECT_ROOT / config.action.mapping.path
    mapping_tools = load_mapping_tools(mapping_path)

    print(f"\n[2/3] 真实工具名（server.js list_tools，共 {len(real_set)} 个）:")
    for name in sorted(real_set):
        print(f"      - {name}")
    print(f"\n[2/3] 映射表工具名（{mapping_path}，共 {len(mapping_tools)} 个）:")
    for name in sorted(mapping_tools):
        print(f"      - {name}")

    missing = sorted(mapping_tools - real_set)
    unused = sorted(real_set - mapping_tools)
    print("\n[3/3] 对齐结论:")
    if missing:
        print(f"  [FAIL] 映射表使用了真实工具名集合中不存在的工具: {missing}", file=sys.stderr)
        return 1
    print("  [OK] 映射表全部工具名 ∈ 真实工具名集合")
    if unused:
        print(f"  [INFO] 真实工具未在映射表中使用（不阻塞）: {unused}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OSC 桥接工具名冒烟对齐")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    config = Config.load(args.config)
    return asyncio.run(run(config))


if __name__ == "__main__":
    raise SystemExit(main())
