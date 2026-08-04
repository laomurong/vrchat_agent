"""空间导航：拓扑图 SQLite 存储（TopologyStore）。

拓扑导航的数据底座：节点（视觉地标指纹）+ 边（相对动作序列）。
- 节点表 nodes：id / fingerprint BLOB（LandmarkFingerprinter 产出的特征向量）/
  描述 / 创建时间；
- 边表 edges：from_id / to_id / action_seq JSON（相对动作序列，
  形如 ``[{"tool": "look_axis", "args": {"axis": "horizontal", "value": -0.6, "duration": 0.5}}, ...]``）/ 创建时间。

设计（复用 brain/memory.py 的成熟模式）：
- SQLite（Python 内置 sqlite3，零新依赖），默认 ``data/nav_topology.db``
  （data/ 已在 .gitignore，不入 git）；
- 每次操作独立连接（sqlite3 check_same_thread 天然安全），构造期不建连——
  懒加载：未启用 nav 或未写入时零开销，未配置也不崩；
- 幂等写入：同指纹节点 / 同 (from,to) 边重复登记返回既有 id，不产生脏数据。

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("vrchat_agent.nav.topology")

# 建表语句（节点 + 边）
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint BLOB NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    action_seq TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(from_id, to_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
"""


class TopologyError(RuntimeError):
    """拓扑图存储错误（异常包装约定）。"""


class TopologyStore:
    """拓扑图存储：节点（视觉指纹）+ 边（相对动作序列）。

    懒加载：连接在首次读写时建立；db_path 的父目录自动创建。
    用法::

        store = TopologyStore("data/nav_topology.db")
        nid = store.add_node(fp, description="喷泉")
        store.add_edge(nid, other, [{"tool": "look_axis", "args": {...}}])
        store.close()
    """

    def __init__(self, db_path: str | Path) -> None:
        """初始化存储。

        Args:
            db_path: SQLite 文件路径（相对/绝对均可；父目录自动创建）。
        """
        self._db_path = Path(db_path)
        self._closed = False
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # 目录不可写等 → 包装为 TopologyError（不裸抛）
            raise TopologyError(f"创建拓扑图目录失败: {exc}") from exc

    # ------------------------------------------------------------------
    # 连接管理（懒加载 + 每次操作独立连接，线程安全）
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """建立新连接并确保表结构存在。"""
        if self._closed:
            raise TopologyError("拓扑图存储已关闭")
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA_SQL)
            return conn
        except sqlite3.Error as exc:
            raise TopologyError(f"打开拓扑图数据库失败: {exc}") from exc

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Connection:
        """执行写语句（事务提交），返回连接（调用方负责 close）。"""
        conn = self._connect()
        try:
            conn.execute(sql, params)
            conn.commit()
            return conn
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"拓扑图写入失败: {exc}") from exc

    @property
    def db_path(self) -> Path:
        """数据库文件路径（供日志/测试断言）。"""
        return self._db_path

    def close(self) -> None:
        """关闭存储（幂等）。"""
        self._closed = True

    # ------------------------------------------------------------------
    # 节点
    # ------------------------------------------------------------------
    def add_node(self, fingerprint: bytes, description: str = "") -> int:
        """登记一个节点（视觉地标）。

        幂等：同指纹节点已存在时直接返回既有 id（指纹即节点唯一键）。

        Args:
            fingerprint: 截图像素指纹（bytes）。
            description: 人类可读描述（如"喷泉""门口"），可为空。

        Returns:
            节点 id（新插入或既有）。
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM nodes WHERE fingerprint = ?", (sqlite3.Binary(fingerprint),)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO nodes (fingerprint, description, created_at) VALUES (?, ?, ?)",
                (sqlite3.Binary(fingerprint), description or "", time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"登记节点失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        """按 id 查询节点；不存在返回 None。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (int(node_id),)).fetchone()
            if row is None:
                return None
            return self._node_to_dict(row)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"查询节点失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def all_nodes(self) -> list[dict[str, Any]]:
        """返回全部节点（按创建时间升序）。"""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM nodes ORDER BY created_at ASC, id ASC").fetchall()
            return [self._node_to_dict(r) for r in rows]
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"查询全部节点失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def node_count(self) -> int:
        """节点总数。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()
            return int(row["c"] if row is not None else 0)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"统计节点失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    @staticmethod
    def _node_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """sqlite Row → dict（fingerprint BLOB 还原为 bytes）。"""
        return {
            "id": int(row["id"]),
            "fingerprint": bytes(row["fingerprint"]),
            "description": str(row["description"] or ""),
            "created_at": float(row["created_at"]),
        }

    # ------------------------------------------------------------------
    # 边
    # ------------------------------------------------------------------
    def add_edge(self, from_id: int, to_id: int, action_seq: list[dict]) -> int:
        """登记一条边（相对动作序列）。

        幂等：同 (from_id, to_id) 已存在时更新 action_seq 并返回既有 id。

        Args:
            from_id: 起点节点 id。
            to_id: 终点节点 id。
            action_seq: 相对动作序列（list[dict]，每个元素形如
                {"tool": "look_axis", "args": {...}}）。

        Returns:
            边 id。
        """
        if from_id == to_id:
            # 自环允许（explorer 回到原位也登记边，便于后续反向利用）
            pass
        payload = json.dumps(list(action_seq or []), ensure_ascii=False)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM edges WHERE from_id = ? AND to_id = ?",
                (int(from_id), int(to_id)),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE edges SET action_seq = ?, created_at = ? WHERE id = ?",
                    (payload, time.time(), int(row["id"])),
                )
                conn.commit()
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO edges (from_id, to_id, action_seq, created_at) VALUES (?, ?, ?, ?)",
                (int(from_id), int(to_id), payload, time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"登记边失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def get_edge(self, from_id: int, to_id: int) -> dict[str, Any] | None:
        """按 (from, to) 查询边；不存在返回 None。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM edges WHERE from_id = ? AND to_id = ?",
                (int(from_id), int(to_id)),
            ).fetchone()
            if row is None:
                return None
            return self._edge_to_dict(row)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"查询边失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def neighbors(self, node_id: int) -> list[tuple[int, list[dict]]]:
        """返回节点的出边邻居：[(to_id, action_seq), ...]。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM edges WHERE from_id = ? ORDER BY id ASC", (int(node_id),)
            ).fetchall()
            return [(int(r["to_id"]), self._parse_action_seq(r["action_seq"])) for r in rows]
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"查询邻居失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def all_edges(self) -> list[dict[str, Any]]:
        """返回全部边（按创建时间升序）。"""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM edges ORDER BY created_at ASC, id ASC").fetchall()
            return [self._edge_to_dict(r) for r in rows]
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"查询全部边失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def edge_count(self) -> int:
        """边总数。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()
            return int(row["c"] if row is not None else 0)
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"统计边失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def clear(self) -> None:
        """清空全部节点与边（测试/重置用）。"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.commit()
        except sqlite3.Error as exc:
            conn.close()
            raise TopologyError(f"清空拓扑图失败: {exc}") from exc
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    @staticmethod
    def _parse_action_seq(raw: str) -> list[dict]:
        """把 action_seq JSON 字符串解析为 list[dict]（损坏时返回空列表）。"""
        try:
            data = json.loads(raw or "[]")
        except (json.JSONDecodeError, TypeError):
            logger.warning("边 action_seq JSON 解析失败（视为空序列）: %r", (raw or "")[:100])
            return []
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        return []

    @staticmethod
    def _edge_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """sqlite Row → dict（action_seq 解析为 list）。"""
        return {
            "id": int(row["id"]),
            "from_id": int(row["from_id"]),
            "to_id": int(row["to_id"]),
            "action_seq": TopologyStore._parse_action_seq(row["action_seq"]),
            "created_at": float(row["created_at"]),
        }
