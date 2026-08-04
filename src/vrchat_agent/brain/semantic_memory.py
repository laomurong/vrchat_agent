"""B 大脑：记忆语义检索层（任务②，替代纯关键词匹配）。

设计要点（参考 Project Gabriel Remastered 的分层记忆 + 语义检索思路）：
- 轻量本地 embedding：sentence-transformers ``paraphrase-multilingual-MiniLM-L12-v2``
  （多语言），模型权重放 ``model/`` 目录（不入 git，见 .gitignore）；
- 向量存储：与记忆库同一个 SQLite 文件的独立表 ``memory_embeddings``（不动
  ``memories`` 表 → 现有 Web CRUD 零破坏，旧库直接复用无需迁移）；
- 懒加载：模型只在首次 ``add/search`` 时才加载；import 本模块不触发任何模型
  下载/加载（测试可完全不接触真实模型）；
- 优雅降级：sentence-transformers 未安装 / 模型目录缺失 / 加载失败 →
  WARNING 日志 + 置为不可用（``ready()`` 为 False），调用方回退关键词匹配，
  绝不抛异常崩主循环；
- 余弦相似度：写入时 L2 归一化（存单位向量），查询时点积即余弦，O(N) 暴力
  扫描（max_items 默认 200，性能足够）；
- 线程安全：实例锁 + 每次操作独立连接（与 MemoryStore 同风格）。

编码器可注入（``encoder`` 参数，duck-typing：``encode(list[str]) -> np.ndarray``）：
测试注入 mock，运行时使用真实 sentence-transformers 模型。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vrchat_agent.brain.semantic_memory")

# 向量表：与 memories 表 1:1（memory_id 主键即外键），独立表避免改动 memories 结构
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# 默认模型名（多语言，用户可下载放入 model/ 目录）
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _now() -> str:
    """当前本地时间（与 MemoryStore 同格式，可读 + 字典序可排序）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _try_import_sentence_transformers():
    """懒加载 sentence-transformers（import 只在此处发生）。

    独立函数便于测试 monkeypatch 短路（模拟未安装，避免真实加载/拖慢测试）。
    """
    import sentence_transformers  # noqa: PLC0415

    return sentence_transformers


class SemanticMemoryError(RuntimeError):
    """语义检索层异常（项目内异常类型；调用方捕获后降级，不崩主流程）。"""


class SemanticMemoryStore:
    """embedding 语义检索存储：向量写入 + 余弦相似度 top-k。

    API 约定：
    - ``add(memory_id, text)``：算 embedding 并 upsert 向量行（内容更新走同一路径）；
    - ``update(memory_id, text)``：同 add（内容更新后重算向量）；
    - ``delete(memory_id)``：删除向量行（记忆删除同步清理）；
    - ``delete_many(ids)``：批量删除（max_items 裁剪时同步清理）；
    - ``search(query, top_k)``：余弦相似度 top-k，返回
      ``[{"memory_id": int, "score": float}, ...]``（按相似度降序）；
      模型不可用返回 None（调用方回退关键词）；库中无向量返回 []；
    - ``ready()``：模型是否已加载（懒加载，首次 add/search 后为 True）；
    - ``count()`` / ``get_vector(memory_id)`` / ``status()``：诊断/测试辅助。
    所有失败只记录 WARNING 并返回安全值，绝不抛异常。
    """

    def __init__(
        self,
        db_path: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        model_dir: str | Path | None = None,
        enabled: bool = True,
        top_k: int = 5,
        device: str = "cpu",
        encoder: Any = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._model_name = (model_name or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        self._model_dir = Path(model_dir) if model_dir else None
        self._enabled = bool(enabled)
        self._top_k = max(1, int(top_k))
        self._device = (device or "cpu").strip() or "cpu"
        self._encoder = encoder  # 测试注入 mock（duck-typing：encode(list[str]))
        self._model: Any = None  # 懒加载：首次使用才装载
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # 内部：连接 / 建表
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """创建独立连接（每次操作一个连接，避免跨线程共享句柄）。"""
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化向量表（失败仅告警，后续操作继续降级）。"""
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
                "语义向量表初始化失败（语义检索降级为不可用）: %s", exc,
                extra={"event": "memory_warn"},
            )

    # ------------------------------------------------------------------
    # 模型懒加载
    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> Path | None:
        """解析本地模型路径：``model_dir/model_name`` 且含 config.json 才算可用。

        模型缺失时返回 None（调用方降级），**不自动联网下载**——模型权重由用户
        手动放入 model/ 目录（不入 git）。
        """
        if self._model_dir is None:
            return None
        path = self._model_dir / self._model_name
        if path.is_dir() and (path / "config.json").exists():
            return path
        # 兼容：model_dir 本身就是模型目录（未按 model_name 分子目录）
        if self._model_dir.is_dir() and (self._model_dir / "config.json").exists():
            return self._model_dir
        return None

    def _ensure_model(self) -> bool:
        """确保 embedding 模型已加载（懒加载 + 失败自动禁用）。

        返回 True 表示可编码；任何失败 → WARNING + 置为不可用 → False。
        """
        if self._model is not None:
            return True
        if not self._enabled:
            return False
        with self._model_lock:
            if self._model is not None:
                return True
            if not self._enabled:
                return False
            # 1) 测试注入的 mock 编码器：直接使用（不 import 真实库）
            if self._encoder is not None:
                self._model = self._encoder
                return True
            # 2) 真实 sentence-transformers：import 只在此处发生（懒加载）
            try:
                st = _try_import_sentence_transformers()
            except Exception as exc:
                logger.warning(
                    "sentence-transformers 未安装，语义检索降级为关键词匹配: %s", exc,
                    extra={"event": "memory_warn"},
                )
                self._enabled = False
                return False
            # 3) 本地模型目录必须存在（不自动下载）
            model_path = self._resolve_model_path()
            if model_path is None:
                logger.warning(
                    "embedding 模型缺失（预期 %s/%s），语义检索降级为关键词匹配；"
                    "请手动下载模型放入 model/ 目录",
                    self._model_dir or "model", self._model_name,
                    extra={"event": "memory_warn"},
                )
                self._enabled = False
                return False
            # 4) 加载模型（CPU/GPU 按配置；失败降级）
            try:
                self._model = st.SentenceTransformer(str(model_path), device=self._device)
                return True
            except Exception as exc:
                logger.warning(
                    "embedding 模型加载失败，语义检索降级为关键词匹配: %s", exc,
                    extra={"event": "memory_warn"},
                )
                self._enabled = False
                return False

    # ------------------------------------------------------------------
    # 编码 / 归一化
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2 归一化为单位向量（零向量保持零向量，避免除零）。"""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm > 1e-8:
            return v / norm
        return v

    def _embed(self, text: str) -> np.ndarray | None:
        """编码单条文本为归一化单位向量；模型不可用/编码失败返回 None。"""
        if not self._ensure_model():
            return None
        try:
            out = self._model.encode([str(text)])
            vec = np.asarray(out[0], dtype=np.float32).reshape(-1)
            return self._normalize(vec)
        except Exception as exc:
            logger.warning(
                "embedding 编码失败（忽略本条，不影响记忆主流程）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return None

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def add(self, memory_id: int, text: str) -> bool:
        """算 embedding 并写入向量行（upsert）。

        模型不可用/编码失败返回 False（记忆本身已写入，仅缺向量 → 检索走关键词）。
        """
        vec = self._embed(text)
        if vec is None:
            return False
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT INTO memory_embeddings (memory_id, vector, dim, updated_at) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(memory_id) DO UPDATE SET "
                        "vector = excluded.vector, dim = excluded.dim, "
                        "updated_at = excluded.updated_at",
                        (int(memory_id), vec.tobytes(), int(vec.size), _now()),
                    )
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "语义向量写入失败（忽略，不影响记忆主流程）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return False

    def update(self, memory_id: int, text: str) -> bool:
        """内容更新后重算向量（upsert，语义同 add）。"""
        return self.add(memory_id, text)

    def delete(self, memory_id: int) -> bool:
        """删除向量行（记忆删除时同步清理）。成功/已不存在返回 True 语义由调用方判定。"""
        return self.delete_many([memory_id]) > 0

    def delete_many(self, memory_ids: list[int]) -> int:
        """批量删除向量行，返回删除条数；失败返回 0（不抛异常）。"""
        ids = [int(i) for i in memory_ids if i is not None]
        if not ids:
            return 0
        try:
            with self._lock:
                conn = self._connect()
                try:
                    placeholders = ",".join("?" * len(ids))
                    cur = conn.execute(
                        f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                        ids,
                    )
                    conn.commit()
                    return int(cur.rowcount)
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "语义向量批量删除失败（忽略）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return 0

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]] | None:
        """余弦相似度 top-k 检索（按相似度降序）。

        返回 ``[{"memory_id": int, "score": float}, ...]``；
        模型不可用返回 None（调用方回退关键词匹配）；库中无向量返回 []。
        """
        q = (query or "").strip()
        if not q:
            return []
        if not self._ensure_model():
            return None
        qv = self._embed(q)
        if qv is None:
            return None
        limit = max(1, int(top_k)) if top_k is not None else self._top_k
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT memory_id, vector FROM memory_embeddings"
                    ).fetchall()
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "语义向量读取失败（降级为不可用）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return None
        scored: list[tuple[int, float]] = []
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            score = float(np.clip(np.dot(vec, qv), -1.0, 1.0))
            scored.append((int(row["memory_id"]), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [{"memory_id": mid, "score": score} for mid, score in scored[:limit]]

    def count(self) -> int:
        """向量行总数。失败返回 0。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT COUNT(*) AS n FROM memory_embeddings"
                    ).fetchone()
                    return int(row["n"]) if row else 0
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "语义向量计数失败（降级为 0）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return 0

    def get_vector(self, memory_id: int) -> np.ndarray | None:
        """返回某记忆的归一化向量（测试/诊断用）；不存在返回 None。"""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT vector FROM memory_embeddings WHERE memory_id = ?",
                        (int(memory_id),),
                    ).fetchone()
                    return np.frombuffer(row["vector"], dtype=np.float32) if row else None
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning(
                "语义向量读取失败（忽略）: %s", exc,
                extra={"event": "memory_warn"},
            )
            return None

    # ------------------------------------------------------------------
    # 属性 / 诊断
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        """模型是否已加载（懒加载：首次 add/search 后才为 True）。"""
        return self._model is not None

    def status(self) -> dict[str, Any]:
        """诊断信息（Web 快照 / 日志用）。"""
        return {
            "enabled": self._enabled,
            "model_loaded": self.ready(),
            "model_name": self._model_name,
            "device": self._device,
            "top_k": self._top_k,
            "vector_count": self.count(),
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def db_path(self) -> Path:
        return self._db_path
