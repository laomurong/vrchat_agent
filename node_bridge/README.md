# Node OSC 桥接（vrchat-mcp-osc）

本目录是 **Node.js OSC 桥接子进程**，复用社区 MCP server `vrchat-mcp-osc`。

> 架构决策：Python 主编排 + Node 仅作 OSC 桥接子进程。
> `vrchat-mcp-osc` 仅通过 **VRChat 官方 OSC 协议**发送输入级指令（表情/参数/移动/聊天框），
> 不涉及任何内存读取 Mod，符合 ToS 合规红线。

## 依赖说明

原 `vrchat-mcp-osc` 在 npm 上**不存在**（仅 GitHub: github.com/Krekun/vrchat-mcp-osc）。
本桥接改用 npm 社区包装包 **`@iflow-mcp/krekun-vrchat-mcp-osc@^0.1.3`**，其 bin 入口为
`packages/mcp-server/dist/server.js`（bin 名为 `iflow-mcp_krekun-vrchat-mcp-osc`）。

> ⚠️ 该包装包存在两处发布缺陷，`npm install` 后由 `scripts/patch-krekun-utils.js`
> （postinstall）自动修复：
> 1. `packages/mcp-server/dist` 相对 import `../../../utils|types/dist` 指向包根，
>    但发布包只有 `packages/*` → 已同步到包根 `utils/dist`、`types/dist`。
> 2. relay-server 子进程 import workspace 包 `@vrchat-mcp-osc/utils`（npm 上不存在）
>    → 已在 `node_modules/@vrchat-mcp-osc/utils` 创建虚拟别名包。
> 另：包装包依赖声明缺少 `winston`，已在本目录 `package.json` 显式声明 `winston@^3.11.0`。

## 安装

```powershell
cd node_bridge
npm install --registry=https://registry.npmmirror.com
```

## 启动方式

主进程（`python -m vrchat_agent.main`）会自动以 stdio 方式拉起：

```
node E:/vrc具身agent/vrchat_agent/node_bridge/node_modules/@iflow-mcp/krekun-vrchat-mcp-osc/packages/mcp-server/dist/server.js
```

命令与参数在 `config.yaml` → `action.bridge` 配置（默认 `node` + 上述 server.js 绝对路径）。
用 `node` + 绝对路径而非 `npx <bin>` 的原因：桥接子进程的 cwd 不固定，`npx` 无法可靠
解析本地 node_modules 的 bin 链接；绝对路径直连最稳定。

## 说明

- 桥接启动失败或中途退出时，主进程进入**降级模式**（仅语音无动作）并 ERROR 告警，进程不崩溃。
- 具体 OSC 工具名（emote / gesture / move / jump / look / chatbox 等）以该包 README 为准，
  映射表见 `mappings/avatar_default.yaml`。
- 使用前请确保：VRChat 已开启 **OSC**（菜单 → OSC → Enable），目标 avatar 支持相应表情/参数。
