"""验证修复：用 MCPClient 在"PATH 不含 node"环境下应成功；构造失败场景错误信息非空。"""

from __future__ import annotations

import asyncio
import os
import sys

NODE = r"C:/Users/14718/.workbuddy/binaries/node/versions/22.22.2/node.exe"
SERVER = (
    r"E:/vrc具身agent/vrchat_agent/node_bridge/node_modules/"
    r"@iflow-mcp/krekun-vrchat-mcp-osc/packages/mcp-server/dist/server.js"
)

sys.path.insert(0, r"E:/vrc具身agent/vrchat_agent/src")


async def main() -> None:
    from vrchat_agent.bridge.mcp_client import MCPClient, MCPError

    # 0) 只临时去掉 PATH 中的 node 目录（保留其余环境，模拟用户 agent 进程）
    node_dir = os.path.dirname(NODE)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and os.path.normcase(p) != os.path.normcase(node_dir)]
    saved_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(parts)
    print("PATH 含 node?", node_dir in os.environ.get("PATH", ""))

    # 1) 模拟用户实机环境：PATH 不含 node → 修复后应握手成功（PATH 注入）
    client = MCPClient(command=NODE, args=[SERVER], timeout=5.0)
    try:
        await client.start()
        print("PASS: 模拟用户环境启动成功（PATH 注入生效）")
        print("  _init_timeout =", client._init_timeout)
    except MCPError as exc:
        print("FAIL: 模拟用户环境启动失败:", exc)
    finally:
        await client.stop()
        os.environ["PATH"] = saved_path

    # 2) 失败场景：命令不存在 → 错误信息非空且可操作
    client2 = MCPClient(command="definitely-not-node", args=[SERVER], timeout=5.0)
    try:
        await client2.start()
        print("FAIL: 不应启动成功")
    except MCPError as exc:
        msg = str(exc)
        print("PASS: 命令不存在错误非空:", msg[:120])
        assert "未找到可执行文件" in msg

    # 3) 失败场景：子进程立即退出（传一个不存在的 server）→ 错误信息非空且含诊断
    client3 = MCPClient(command=NODE, args=[r"E:/vrc具身agent/vrchat_agent/nonexistent_server.js"], timeout=5.0)
    try:
        await client3.start()
        print("FAIL: 不应启动成功")
    except MCPError as exc:
        msg = str(exc)
        print("PASS: 子进程退出错误非空:", msg[:200].replace("\n", " | "))
        assert msg.strip().endswith(":") is False, "错误信息不应以冒号结尾（空）"


if __name__ == "__main__":
    asyncio.run(main())
