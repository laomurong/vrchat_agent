# 交付概览：上下转向 + LLM纯文本降级修复 + 看向目标引导

## TL;DR
为 VRChat Agent 添加了上下转向（抬头/低头）能力，修复了 LLM 返回纯文本时用户话被静默吞掉的 Bug，并新增了"看向目标"的 prompt 引导。

## 交付概览
- **交付状态**：✅ 完成
- **测试通过率**：685 passed, 1 skipped, 3 failed（3 个预存环境问题，与本次改动无关）
- **已知问题**：3 个预存测试失败（bootfix2/live/live2，环境相关）

## 改动清单

### 源码（5 个文件）
| 文件 | 改动说明 |
|------|----------|
| `src/vrchat_agent/brain/schema.py` | 新增纯文本回退：LLM 返回不含 `{` 的文本 → 降级为 respond=True 回复（而非静默丢弃） |
| `src/vrchat_agent/brain/prompts.py` | look 枚举新增 up/down/up_big/down_big + "看向目标"组合动作引导规则 |
| `mappings/avatar_default.yaml` | 新增 look:up / look:down / look:up_big / look:down_big 映射条目 |
| `node_bridge/scripts/patch-krekun-osc-look-updown.js` | **新建** patch 脚本：扩展 look_direction 支持 up/down 方向 |
| `node_bridge/package.json` | postinstall 追加新 patch 脚本 |

### 测试（4 个文件）
| 文件 | 改动说明 |
|------|----------|
| `tests/test_qa_brain_fallback.py` | 纯文本样本拆分 + 新增纯文本回退测试 |
| `tests/test_qa_reply_chain.py` | bad_json 测试改用含 `{` 的坏 JSON + 新增纯文本 WARNING 测试 |
| `tests/test_qa_oscmap.py` | look_direction 枚举加 up/down |
| `tests/test_qa_oscmap2.py` | 同上 |

## 功能说明

### 1. 上下转向
- LLM 现在可以输出 `{"kind":"look","params":{"name":"up"}}` 或 `down` 来抬头/低头
- 大角度变体 `up_big` / `down_big` 用于更大幅度
- 底层通过 VRChat OSC `/input/LookUp` 和 `/input/LookDown` 实现

### 2. 看向目标
- Prompt 新增规则：当玩家说"看向XXX"时，LLM 组合多个 look 动作近似朝向
- 例如"看向左上方" → 先 `look:left` 再 `look:up`
- OSC 只有相对方向输入（无绝对定位），所以用组合动作近似

### 3. LLM 纯文本降级修复
- **旧行为**：LLM 返回纯文本 → 解析失败 → respond=False（用户话被吞掉）
- **新行为**：LLM 返回不含 `{` 的纯文本 → 降级为 respond=True 纯文本回复
- 含 `{` 但 JSON 语法错误的仍走 respond=False（可能是截断/乱码）
- 新增 WARNING 日志 event=decision_plain_text_fallback 便于诊断

## 用户下一步建议
1. 重启 VRChat Agent 即可生效（patch 已自动应用）
2. 试试对 Agent 说"向上转头"、"低头看地面"、"看向左上方"
3. 如果上下转向幅度不合适，可在 `mappings/avatar_default.yaml` 中调整 duration 参数
4. 如需重新安装 Node 依赖，postinstall 会自动运行所有 patch 脚本
