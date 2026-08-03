"""Web 控制台（M4 增量，W-01~W-05）。

同进程内嵌 FastAPI + uvicorn asyncio task；只通过三条既有通道与系统交互：
① EventBus（stream() 订阅 / publish_threadsafe() 发急停）
② Config 门面（get()/path/reload()）
③ UsageTracker.summary()
不直接 import A~F 模块内部类或调用 AgentLoop 私有方法（共享知识约定第 1 条）。
"""

from .server import WebServer, create_app

__all__ = ["WebServer", "create_app"]
