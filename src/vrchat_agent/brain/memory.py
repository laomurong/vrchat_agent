"""B 大脑：长期记忆存储（MemoryStore，标准版增量 + 语义检索层）。

本地持久化记忆系统：
- SQLite（Python 内置 sqlite3，零新依赖），文件默认 ``data/memory.db``（相对项目根）；
- AI 记住长期事实/喜好，重启不丢、跨会话生效；
- 表结构：``memories(id INTEGER PK, content TEXT, kind TEXT, created_at TEXT,
  updated_at TEXT, last_used_at TEXT, use_count INTEGER)``，kind ∈ fact/preference/note；
- 线程安全：实例锁 + 每次操作独立连接（sqlite3 check_same_thread 天然安全）；
- 失败降级：磁盘不可写/库损坏 → WARNING 日志 + 返回空/False/None，绝不抛异常崩主循环；
- 裁剪：超过 ``max_items`` 删除最久未用（last_used_at 最旧优先）的记忆；
- 语义检索（任务②）：可选注入 ``SemanticMemoryStore``（懒加载 embedding），
  写入时同步算向量（独立表 memory_embeddings，不动 memories 表 → Web CRUD 零破坏），
  检索时先语义余弦 top-k、再关键词 LIKE 兜底；模型缺失自动降级，不影响主流程。
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
    - ``search(keyword, top_k=None)``：语义优先检索（余弦 top-k），语义不可用/无
      命中回退内容模糊搜索（LIKE）；
    - ``update(id, content, kind=None)``：编辑（同步重算向量）；
    - ``delete(id)``：删除（同步清理向量）；
    - ``recall(limit)``：按 use_count 高 + 最近使用优先取常用记忆（供注入 LLM）；
    - ``mark_used(ids)``：标记记忆被注入使用（提升 recall 优先级）；
    - ``count()``：总条数。
    所有写操作失败只记录 WARNING，不抛异常（主循环不受影响）。
    """

    def __init__(
        self,
        db_path: str | Path,
        max_items: int = 200,
        enabled: bool = True,
        semantic: Any = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._max_items = max(1, int(max_items))
        self._enabled = bool(enabled)
        # 语义检索层（任务②，可选）：懒加载 embedding；模型缺失自动降级关键词，
        # 所有语义操作失败只记 WARNING，绝不抛异常影响记忆主流程。
        self._semantic: Any = semantic
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
        doomed_ids: list[int] = []
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
                    total = int(row["n"]) if row else 0
                    excess = total - self._max_items
                    if excess > 0:
                        doomed = conn.execute(
                            "SELECT id FROM memories "
                            "ORDER BY last_used_at ASC, use_count ASC, id ASC LIMIT ?",
                            (excess,),
                        ).fetchall()
                        doomed_ids = [int(r["id"]) for r in doomed]
                        if doomed_ids:
                            placeholders = ",".join("?" * len(doomed_ids))
                            conn.execute(
                                f"DELETE FROM memories WHERE id IN ({placeholders})",
                                doomed_ids,
                            )
                            conn.commit()
                finally:
                    conn.close()
            if doomed_ids:
                self._semantic_delete_many(doomed_ids)
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
        inserted = False
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
                        inserted = True
                finally:
                    conn.close()
            self._prune()
            # 新记忆才需要算向量；去重命中（内容未变）跳过，省一次 embedding
            if inserted and mem_id is not None:
                self._semantic_add(mem_id, text)
            return mem_id
        except Exception as exc:
            logger.warning("记忆写入失败（降级忽略）: %s", exc, extra={"event": "memory_warn"})
            return None

    def update(self, mem_id: int, content: str, kind: str | None = None) -> bool:
        """编辑记忆内容（可选更新 kind）。成功 True；未找到/失败 False。"""
        text = (content or "").strip()
        if not text:
            return False
        ok = False
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
                    ok = cur.rowcount > 0
                finally:
                    conn.close()
            if ok:
                self._semantic_update(mem_id, text)
            return ok
        except Exception as exc:
            logger.warning("记忆更新失败（降级忽略）: %s", exc, extra={"event": "memory_warn"})
            return False

    def delete(self, mem_id: int) -> bool:
        """删除一条记忆（同步清理语义向量）。成功 True；未找到/失败 False。"""
        ok = False
        try:
            with self._lock:
                conn = self._connect()
                try:
                    cur = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                    conn.commit()
                    ok = cur.rowcount > 0
                finally:
                    conn.close()
            if ok:
                self._semantic_delete(mem_id)
            return ok
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

    def search(self, keyword: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """检索记忆：语义优先（余弦 top-k），语义不可用/无命中 → 关键词 LIKE 兜底。

        未注入语义层 / 模型缺失 / 语义失败均自动回退关键词（旧行为不回归）；
        空关键词等价 list()。失败返回空列表。
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return self.list()
        # 1) 语义检索（可选层；返回 None=降级，返回 []=无命中/无向量 → 都走关键词兜底）
        if self._semantic is not None:
            try:
                hits = self._semantic.search(keyword, top_k=top_k)
            except Exception as exc:
                logger.warning(
                    "语义检索失败，回退关键词匹配: %s", exc,
                    extra={"event": "memory_warn"},
                )
                hits = None
            if hits:
                ids = [int(h["memory_id"]) for h in hits]
                return self._fetch_by_ids(ids)
        # 2) 关键词 LIKE 兜底
        return self._keyword_search(keyword)

    def _keyword_search(self, keyword: str) -> list[dict[str, Any]]:
        """按内容模糊搜索（LIKE，%keyword%）。失败返回空列表。"""
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

    def _fetch_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        """按 id 列表取完整记忆行，保持传入顺序（语义 top-k 排序不丢失）。"""
        if not ids:
            return []
        try:
            with self._lock:
                conn = self._connect()
                try:
                    placeholders = ",".join("?" * len(ids))
                    rows = conn.execute(
                        f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
                    ).fetchall()
                    by_id = {int(r["id"]): dict(r) for r in rows}
                finally:
                    conn.close()
            out: list[dict[str, Any]] = []
            for mem_id in ids:
                row = by_id.get(int(mem_id))
                if row is not None:
                    out.append(row)
            return out
        except Exception as exc:
            logger.warning("记忆按 id 读取失败（降级为空列表）: %s", exc, extra={"event": "memory_warn"})
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
    # 语义检索层同步（任务②）：全部静默降级，绝不抛异常影响记忆主流程
    # ------------------------------------------------------------------
    def _semantic_add(self, mem_id: int, text: str) -> None:
        """新记忆写入后同步算向量（失败仅告警，记忆本身已落库）。"""
        if self._semantic is None:
            return
        try:
            self._semantic.add(mem_id, text)
        except Exception as exc:
            logger.warning(
                "语义向量写入同步失败（忽略，检索将回退关键词）: %s", exc,
                extra={"event": "memory_warn"},
            )

    def _semantic_update(self, mem_id: int, text: str) -> None:
        """记忆内容更新后同步重算向量。"""
        if self._semantic is None:
            return
        try:
            self._semantic.update(mem_id, text)
        except Exception as exc:
            logger.warning(
                "语义向量更新同步失败（忽略，检索将回退关键词）: %s", exc,
                extra={"event": "memory_warn"},
            )

    def _semantic_delete(self, mem_id: int) -> None:
        """记忆删除后同步清理向量。"""
        if self._semantic is None:
            return
        try:
            self._semantic.delete(mem_id)
        except Exception as exc:
            logger.warning(
                "语义向量删除同步失败（忽略）: %s", exc,
                extra={"event": "memory_warn"},
            )

    def _semantic_delete_many(self, ids: list[int]) -> None:
        """批量清理向量（max_items 裁剪时同步）。"""
        if self._semantic is None or not ids:
            return
        try:
            self._semantic.delete_many(ids)
        except Exception as exc:
            logger.warning(
                "语义向量批量删除同步失败（忽略）: %s", exc,
                extra={"event": "memory_warn"},
            )

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

    @property
    def semantic(self) -> Any:
        """语义检索层实例（未注入返回 None）。"""
        return self._semantic
