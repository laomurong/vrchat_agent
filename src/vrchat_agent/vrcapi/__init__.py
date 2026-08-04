"""VRChat API 集成包（任务①：换头像 / 查好友 / 找世界 / 更新状态）。

模块划分：
- client.py   懒加载登录 + 2FA 处理 + 全局限速器（≤1 次/60s，全局串行节流）
- actions.py  四个能力的异步封装（change_avatar / list_friends /
              search_worlds / update_status），动作系统 vrc:* 的执行端

共享知识约定：API key / 账号密码一律从环境变量读取（VRC_USERNAME /
VRC_PASSWORD / VRC_API_KEY），绝不硬编码、绝不落盘明文；所有外部 API 异常
统一包装为 vrcapi.client.VRCAPIError，不裸抛。
"""

from .actions import VRCActionExecutor
from .client import RateLimiter, VRCAPIClient, VRCAPIError

__all__ = ["RateLimiter", "VRCAPIClient", "VRCAPIError", "VRCActionExecutor"]
