"""B 大脑：长期记忆存储（MemoryStore，标准版增量）。

本地持久化记忆系统：
- SQLite（Python 内置 sqlite3，零新依赖），文件默认 ``data/memory.db``（相对项目根）；
- AI 记住长期事实/喜好，重启不丢、跨会话生效；
- 表结构：``memories(id INTEGER PK, content TEXT, kind TEXT, created_at TEXT,
  updated_at TEXT, last_used_at TEXT, use_count INTEGER)``，kind ∈ fact/preference/note；
- 线程安全：实例锁 + 每次操作独立连接（sqlite3 check_same_thread 天然安全）；
- 失败降级：磁盘不可写/库损坏 → WARNING 日志 + 返回空/False/None，绝不抛异常崩主循环；
- 裁剪：超过 ``max_items`` 删除最久未用（last_used_at 最旧优先）的记忆。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vrchat_agent.brain.memory")

# 允许的 kind 枚举（宽松：其它字符串也接受并原样存储，Web 只提供这三种）
KINDS = ("fact", "preference", "note")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at);
"""


def _now() -> str:
    """当前本地时间（可读 + 字典序可排序，Web 直接展示）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryStore:
    """SQLite 长期记忆库。

    API 约定（标准版）：
    - ``add(content, kind)``：新增/去重（同 content 命中则 use_count+1 并刷新时间），
      返回记录 id（失败返回 None）；
    - ``list(kind=None)``：按 updated_at 倒序列出；
    - ``search(keyword)``：内容模糊搜索；
    - ``update(id, content, kind=None)``：编辑；
    - ``delete(id)``：删除；
    - ``recall(limit)``：按 use_count 高 + 最近使用优先取常用记忆（供注入 LLM）；
    - ``mark_used(ids)``：标记记忆被注入使用（提升 recall 优先级）；
    - ``count()``：总条数。
    所有写操作失败只记录 WARNING，不抛异常（主循环不受影响）。
    """

    def __init__(self, db_path: str | Path, max_items: int = 200, enabled: bool = True) -> None:
        self._db_path = Path(db_path)
        self._max_items = max(1, int(max_items))
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # 内部：连接 / 建表 / 裁剪
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """创建独立连接（每次操作一个连接，避免跨线程共享句柄）。"""
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化建表（失败仅告警，后续操作继续降级）。"""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                conn = self._connect()
                try:
                    conn.executescript(_SCHEMA)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "记忆库初始化失败（记忆系统降级为不可用，不影响主循环）: %s", exc,
                extra={"event": "memory_warn"},
            )

    def _prune(self) -> None:
        """超过 max_items：删除最久未用（last_used_at 最旧、use_count 最低）的记忆。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
                    total = int(row["n"]) if row else 0
                    excess = total - self._max_items
                    if excess > 0:
                        conn.execute(
                            "DELETE FROM memories WHERE id IN ("
                            "SELECT id FROM memories "
                            "ORDER BY last_used_at ASC, use_count ASC, id ASC LIMIT ?"
                            ")",
                            (excess,),
                        )
                        conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆裁剪失败（忽略）: %s", exc, extra={"event": "memory_warn"})

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def add(self, content: str, kind: str = "fact") -> int | None:
        """新增一条记忆；同 content 去重（命中则 use_count+1 并刷新时间）。

        返回记录 id（去重命中返回原 id）；content 为空/写入失败返回 None。
        """
        text = (content or "").strip()
        if not text:
            return None
        kind = (kind or "fact").strip() or "fact"
        now = _now()
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT id, use_count FROM memories WHERE content = ?", (text,)
                    ).fetchone()
                    if row is not None:
                        conn.execute(
                            "UPDATE memories SET use_count = use_count + 1, "
                            "updated_at = ?, last_used_at = ? WHERE id = ?",
                            (now, now, row["id"]),
                        )
                        conn.commit()
                        mem_id = int(row["id"])
                    else:
                        cur = conn.execute(
                            "INSERT INTO memories (content, kind, created_at, updated_at, "
                            "last_used_at, use_count) VALUES (?, ?, ?, ?, ?, 1)",
                            (text, kind, now, now, now),
                        )
                        conn.commit()
                        mem_id = int(cur.lastrowid)
                finally:
                    conn.close()
            self._prune()
            return mem_id
        except Exception as exc:
            logger.warning("记忆写入失败（降级忽略）: %s", exc, extra={"event": "memory_warn"})
            return None

    def update(self, mem_id: int, content: str, kind: str | None = None) -> bool:
        """编辑记忆内容（可选更新 kind）。成功 True；未找到/失败 False。"""
        text = (content or "").strip()
        if not text:
            return False
        try:
            with self._lock:
                conn = self._connect()
                try:
                    if kind is None:
                        cur = conn.execute(
                            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                            (text, _now(), mem_id),
                        )
                    else:
                        cur = conn.execute(
                            "UPDATE memories SET content = ?, kind = ?, updated_at = ? WHERE id = ?",
                            (text, (kind or "fact").strip() or "fact", _now(), mem_id),
                        )
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆更新失败（降级忽略）: %s", exc, extra={"event": "memory_warn"})
            return False

    def delete(self, mem_id: int) -> bool:
        """删除一条记忆。成功 True；未找到/失败 False。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    cur = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                    conn.commit()
                    return cur.rowcount > 0
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆删除失败（降级忽略）: %s", exc, extra={"event": "memory_warn"})
            return False

    def mark_used(self, mem_ids: list[int]) -> None:
        """标记记忆被注入使用（use_count+1 + 刷新 last_used_at，提升 recall 优先级）。"""
        ids = [int(i) for i in mem_ids if i is not None]
        if not ids:
            return
        now = _now()
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.executemany(
                        "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                        [(now, i) for i in ids],
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆使用标记失败（忽略）: %s", exc, extra={"event": "memory_warn"})

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """列出记忆（按 updated_at 倒序）；kind 过滤可选。失败返回空列表。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    if kind:
                        rows = conn.execute(
                            "SELECT * FROM memories WHERE kind = ? "
                            "ORDER BY updated_at DESC, id DESC",
                            (kind,),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT * FROM memories ORDER BY updated_at DESC, id DESC"
                        ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆读取失败（降级为空列表）: %s", exc, extra={"event": "memory_warn"})
            return []

    def search(self, keyword: str) -> list[dict[str, Any]]:
        """按内容模糊搜索（LIKE，%keyword%）；空关键词等价 list()。失败返回空列表。"""
        keyword = (keyword or "").strip()
        if not keyword:
            return self.list()
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE content LIKE ? "
                        "ORDER BY use_count DESC, last_used_at DESC, id DESC",
                        (f"%{keyword}%",),
                    ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆搜索失败（降级为空列表）: %s", exc, extra={"event": "memory_warn"})
            return []

    def recall(self, limit: int = 5) -> list[dict[str, Any]]:
        """取常用记忆（use_count 高 + 最近使用优先），用于注入 LLM。失败返回空列表。"""
        limit = max(1, int(limit))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM memories "
                        "ORDER BY use_count DESC, last_used_at DESC, id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆召回失败（降级为空列表）: %s", exc, extra={"event": "memory_warn"})
            return []

    def count(self) -> int:
        """记忆总条数。失败返回 0。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
                    return int(row["n"]) if row else 0
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("记忆计数失败（降级为 0）: %s", exc, extra={"event": "memory_warn"})
            return 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def max_items(self) -> int:
        return self._max_items

    @property
    def enabled(self) -> bool:
        return self._enabled
