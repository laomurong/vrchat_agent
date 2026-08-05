"""VRChat API 集成包（任务① + T03：vrchatapi 全 API 接入）。

模块划分：
- client.py       懒加载登录 + 2FA 处理 + 全局限速器（≤1 次/60s，全局串行节流）
                  + 19 类 API 句柄（authentication/avatars/.../worlds）
- two_factor.py   two_factor_code_provider 标准实现：Web 等待输入 + .env
                  VRC_2FA_CODE 兜底（P2-1：Web 2FA 验证码交互）
- actions.py      精选 16 动作 + call_api 通用层（VRCActionExecutor），
                  动作系统 vrc:* 的执行端；社交写操作审计日志
- api_catalog.py  APICatalog（274 方法目录）+ SecurityPolicy（只读开放/
                  白名单审计/allow_destructive/危险关闭）
- catalog.yaml    机器可读目录（scripts/gen_vrcapi_catalog.py 从已装 SDK 生成）

共享知识约定：API key / 账号密码一律从环境变量读取（VRC_USERNAME /
VRC_PASSWORD / VRC_API_KEY），绝不硬编码、绝不落盘明文；所有外部 API 异常
统一包装为 vrcapi.client.VRCAPIError，不裸抛。
"""

from .actions import VRCActionExecutor
from .api_catalog import APICatalog, CatalogEntry, SecurityPolicy
from .client import RateLimiter, VRCAPIClient, VRCAPIError
from .two_factor import DEFAULT_2FA_CODE_ENV, TwoFactorCodeError, TwoFactorCodeService

__all__ = [
    "APICatalog",
    "CatalogEntry",
    "DEFAULT_2FA_CODE_ENV",
    "RateLimiter",
    "SecurityPolicy",
    "TwoFactorCodeError",
    "TwoFactorCodeService",
    "VRCAPIClient",
    "VRCAPIError",
    "VRCActionExecutor",
]
