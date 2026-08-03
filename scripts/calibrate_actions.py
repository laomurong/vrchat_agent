"""动作校准工具：拉取当前 avatar 的真实 OSC 参数，生成建议映射（任务 3）。

用法（conda 环境 vrchat-agent，需 VRChat 已开启 OSC）：
    conda activate vrchat-agent
    python scripts/calibrate_actions.py [--config config.yaml] [--probe]

流程：
    1) 拉起 OSC 桥接（MCPClient，复用 config.action.bridge）
    2) get_avatar_name → 当前 avatar 名
    3) get_avatar_parameters → 当前 avatar 真实参数名列表（容错解析多种返回形态）
    4) 按关键词自动关联语义 → 生成建议映射 YAML 片段（用户据此更新映射表）
    5) --probe（可选）：对常见候选参数名调用 set_avatar_parameter 探测
       （value=true 后立即复位 false），打印工具返回文本供人工判断。

设计选择（回传说明）：采用"脚本 + 文档"方案，不做运行时自动 get_avatar_parameters
自适应。理由：运行时探测需每次启动连接 VRChat 查询参数，增加启动延迟、依赖 VRChat
在线、且 avatar 参数结构多样需解析；脚本按需跑更可控、更简单、对测试友好。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vrchat_agent.bridge.mcp_client import MCPClient  # noqa: E402
from vrchat_agent.config import Config  # noqa: E402

# 常见语义 → 建议动作 key（与 mappings/avatar_default.yaml 的 key 对齐）。
# 关键词按小写匹配参数名；命中多条时按顺序取第一条。
_PARAM_SUGGESTIONS: list[tuple[str, str]] = [
    # (参数名小写包含的关键词, 建议动作 key)
    ("wave", "gesture:wave_hand"),
    ("thumbsup", "gesture:thumbs_up"),
    ("thumbs", "gesture:thumbs_up"),
    ("point", "gesture:point"),
    ("bow", "emote:bow"),
    ("clap", "emote:clap"),
    ("dance", "emote:dance"),
    ("laugh", "emote:laugh"),
    ("jump", "jump:jump"),
]

# --probe 模式下探测的常见候选参数名（按 avatar 常见手势/表情命名习惯）
_PROBE_CANDIDATES = [
    "Wave",
    "ThumbsUp",
    "Point",
    "Bow",
    "Clap",
    "Dance",
    "Laugh",
    "Jump",
]


def _extract_texts(result) -> list[str]:
    """从 MCP CallToolResult / dict 提取文本内容（容错）。"""
    if result is None:
        return []
    texts: list[str] = []
    if isinstance(result, dict):
        content = result.get("content") or []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    texts.append(str(c.get("text", "")))
                elif hasattr(c, "text"):
                    texts.append(str(c.text))
                else:
                    texts.append(str(c))
        elif isinstance(content, str):
            texts.append(content)
        err = result.get("error")
        if err:
            texts.append(str(err))
    elif hasattr(result, "content"):
        content = getattr(result, "content", None) or []
        for c in content:
            texts.append(getattr(c, "text", str(c)))
    else:
        texts.append(str(result))
    return [t for t in texts if t.strip()]


def _walk_parameter_names(node, out: list[str]) -> None:
    """递归提取参数名：兼容 list[str] / list[dict] / dict{parameters: ...} 等形态。"""
    if isinstance(node, dict):
        for key in ("name", "parameter_name", "parameterName", "parameter"):
            if key in node and isinstance(node[key], str) and node[key].strip():
                out.append(node[key])
        for value in node.values():
            _walk_parameter_names(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_parameter_names(item, out)
    elif isinstance(node, str):
        s = node.strip()
        if s:
            out.append(s)


def _extract_parameters(result) -> list[str]:
    """从 get_avatar_parameters 结果提取参数名（容错：JSON 文本 / 裸文本 / 嵌套）。"""
    raw_names: list[str] = []
    for text in _extract_texts(result):
        data = None
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            data = text
        _walk_parameter_names(data, raw_names)
    # 去重保序
    seen: set[str] = set()
    names: list[str] = []
    for n in raw_names:
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


def _suggest_mapping(param_names: list[str]) -> dict[str, list[str]]:
    """按关键词把真实参数名关联到建议动作 key。"""
    suggestions: dict[str, list[str]] = {}
    for name in param_names:
        low = name.lower()
        for keyword, action_key in _PARAM_SUGGESTIONS:
            if keyword in low:
                suggestions.setdefault(action_key, []).append(name)
                break
    return suggestions


def _format_suggested_yaml(suggestions: dict[str, list[str]]) -> str:
    """把建议映射渲染成 YAML 片段（用户可直接复制进映射表）。"""
    if not suggestions:
        return "# （未匹配到常见语义参数名，请人工查看上面的参数列表后手动填写）\n"
    lines = ["# ---- 建议映射（由 calibrate_actions.py 生成，请人工确认后使用）----"]
    for action_key in sorted(suggestions):
        param_names = suggestions[action_key]
        if action_key.startswith("gesture:"):
            lines.append(f"  {action_key}:")
            lines.append("    tools:")
            for pn in param_names[:1]:  # 每个语义取第一个命中参数即可
                lines.append(f"      - tool: set_avatar_parameter")
                lines.append(f"        args: {{ parameter_name: {pn}, value: true }}")
        elif action_key.startswith("jump"):
            lines.append(f"  {action_key}:")
            lines.append("    tools:")
            lines.append(f"      - tool: jump")
            lines.append(f"        args: {{}}")
        else:  # emote 语义（VRCEmote 索引，无法自动确定序号，仅提示）
            lines.append(f"  # {action_key}: VRCEmote 索引需按 Expressions 菜单顺序人工填写")
            lines.append(f"  #   相关参数: {', '.join(param_names)}")
    return "\n".join(lines) + "\n"


async def _probe_parameters(client: MCPClient, candidates: list[str]) -> None:
    """（可选）对候选参数名 set_avatar_parameter 探测，打印返回文本。"""
    print("\n[probe] 对常见候选参数名执行 set_avatar_parameter(value=true→false) 探测:")
    for pname in candidates:
        try:
            await client.call_tool(
                "set_avatar_parameter", {"parameter_name": pname, "value": True}
            )
            await client.call_tool(
                "set_avatar_parameter", {"parameter_name": pname, "value": False}
            )
            print(f"      {pname}: 调用成功（未返回错误；是否生效需在 VRChat 中观察）")
        except Exception as exc:
            print(f"      {pname}: 调用失败 -> {exc}")


async def run(config: Config, probe: bool = False) -> int:
    bridge = config.action.bridge
    client = MCPClient(command=bridge.command, args=bridge.args, timeout=bridge.timeout)
    print(f"[1/5] 拉起桥接: {bridge.command} {' '.join(bridge.args)}")
    try:
        await client.start()
    except Exception as exc:
        print(f"[FAIL] 拉起 server.js 失败: {exc}", file=sys.stderr)
        return 2
    try:
        tools = await client.list_tools()
        print(f"\n[2/5] 桥接实际暴露工具（{len(tools)} 个）:")
        print("      " + ", ".join(sorted(tools)))

        print("\n[3/5] 当前 avatar:")
        try:
            avatar_result = await client.call_tool("get_avatar_name", {})
            avatar_text = " | ".join(_extract_texts(avatar_result))
            print(f"      {avatar_text}")
        except Exception as exc:
            print(f"      （获取 avatar 名失败: {exc}，继续）")

        print("\n[4/5] get_avatar_parameters → 当前 avatar 真实参数名:")
        try:
            params_result = await client.call_tool("get_avatar_parameters", {})
        except Exception as exc:
            print(f"[FAIL] get_avatar_parameters 调用失败: {exc}", file=sys.stderr)
            return 3
        param_names = _extract_parameters(params_result)
        if not param_names:
            print("      （未解析到参数名；原始返回如下，供人工判断）")
            for t in _extract_texts(params_result):
                print(f"      {t[:500]}")
        else:
            for name in param_names:
                print(f"      - {name}")

        suggestions = _suggest_mapping(param_names)
        print("\n[5/5] 建议映射（按关键词自动关联，请人工确认后更新映射表）:")
        print(_format_suggested_yaml(suggestions))
        print("下一步：把上面建议写入 mappings/avatar_default.yaml（或复制为")
        print("mappings/avatar_<名字>.yaml 后在 config.yaml → action.mapping.path 切换），")
        print("重启 Agent 生效。")
    finally:
        if probe:
            await _probe_parameters(client, _PROBE_CANDIDATES)
        await client.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VRChat Agent 动作校准：列出当前 avatar 真实参数并生成建议映射"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--probe",
        action="store_true",
        help="额外对常见候选参数名执行 set_avatar_parameter 探测（可选）",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    return asyncio.run(run(config, probe=args.probe))


if __name__ == "__main__":
    raise SystemExit(main())
