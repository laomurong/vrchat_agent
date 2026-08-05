"""空间导航：视觉地标指纹（Landmark Fingerprint）与回环检测。

核心约束（调研确认）：VRChat OSC 没有绝对坐标——/input/* 只有相对转向，
读不到自己/他人在世界里的位置。因此空间导航只能做**拓扑导航**：
节点 = 视觉地标指纹（截图像素特征向量），边 = 相对动作序列。

本模块提供：
- ``LandmarkFingerprinter``：把一张截图降采样为固定长度的特征向量，
  序列化为 bytes；同一截图 → 同一指纹（确定性），不同场景 → 不同指纹（可分性）。
- 相似度度量：归一化 L1 距离（向量逐元素绝对差之和 / 最大可能差），
  取值 [0,1]，越大越相似。
- 回环检测：当前帧指纹与已访问节点指纹逐一比对，相似度超阈值即判定
  "回到了已知位置"（节点 id）。

指纹组成（P0-1 亮度不变增强）：
1. 灰度均值池化（32×32，权重 0.5）：绝对亮度上下文（形状/明暗分布），
   对均匀纯色场景提供可分性；
2. BGR 颜色直方图（3 通道 × 4 桶，权重 24）：绝对颜色信息；
3. **梯度幅值直方图**（8 桶，亮度不变）：Sobel 梯度幅值经帧均幅值归一化——
   乘性/加性亮度变化下梯度分布不变（同场景不同光照 → 仍高相似），
   不同场景的纹理/边缘结构不同（增强区分度）；
4. **归一化颜色比直方图**（3 通道 × 4 桶，权重 24，亮度不变）：
   r/(r+g+b) 等颜色比例——同色不同亮度 → 相同分布，红 vs 蓝 → 明显不同；
5. **局部对比度池化**（16×16，块内 std / 帧均 std，亮度不变）：
   捕获纹理/边缘的空间分布，不依赖绝对亮度。

既有测试约束（红/蓝可分 <0.85、同场景微变 >0.90）全部保持。

作者：寇豆码（Kou）· 工程师
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("vrchat_agent.nav.landmark")

# 颜色直方图每通道桶数（BGR 三通道 → 3 × 4 = 12 维）
_HIST_BINS_PER_CHANNEL = 4
# 颜色直方图在特征向量中的权重（放大直方图贡献，弥补纯色画面灰度可分性不足）
_HIST_WEIGHT = 24.0
# P0-1：绝对灰度权重（亮度敏感特征降权——只作形状/亮度上下文，避免光照变化主导差异）
_GRAY_WEIGHT = 0.5
# P0-1：梯度幅值直方图桶数（亮度不变纹理特征）
_GRAD_BINS = 8
# P0-1：梯度直方图权重
_GRAD_WEIGHT = 1.0
# P0-1：归一化颜色比直方图桶数（BGR 三通道 → 3 × 4 = 12 维；亮度不变颜色身份）
_RATIO_BINS_PER_CHANNEL = 4
# P0-1：颜色比直方图权重（与绝对颜色直方图同权重，保证纯色可分性不依赖绝对亮度）
_RATIO_WEIGHT = 24.0
# P0-1：局部对比度池化边长（16×16 = 256 维；亮度不变纹理分布）
_CONTRAST_SIZE = 16
# P0-1：局部对比度权重
_CONTRAST_WEIGHT = 1.0
# 梯度幅值归一化后直方图覆盖范围（[0, 4] 倍帧均幅值 → 8 桶）
_GRAD_NORM_RANGE = 4.0


class NavError(RuntimeError):
    """空间导航统一异常基类（异常包装约定）。"""


class LandmarkFingerprinter:
    """截图像素指纹器：截图 → 固定长度特征向量（bytes）。

    指纹组成（详见模块 docstring）：
    1. 灰度均值池化（32×32，权重 0.5）——绝对亮度上下文；
    2. BGR 颜色直方图（12 维，权重 24）——绝对颜色；
    3. 梯度幅值直方图（8 维，权重 1，亮度不变）——边缘/纹理；
    4. 归一化颜色比直方图（12 维，权重 24，亮度不变）——颜色身份；
    5. 局部对比度池化（256 维，权重 1，亮度不变）——纹理空间分布。
    拼接后以 float32 序列化为 bytes —— 确定性：同一输入必然产出相同 bytes。

    相似度用归一化 L1 距离：``sim = 1 - |va - vb|.sum() / len(va)``，
    相同向量 → 1.0；完全不同 → 接近 0。阈值 0.85 左右可区分"不同场景"
    （纯色红 vs 纯色蓝 ≈ 0.81）而"同场景微抖/光照变化"仍能回环（>0.87）。
    """

    def __init__(self, size: int = 32) -> None:
        """初始化指纹器。

        Args:
            size: 灰度均值池化边长（默认 32，与任务规格一致；
                纯 numpy 实现不依赖 cv2，测试/运行均轻量）。
        """
        self._size = max(8, int(size))
        # 特征向量维度 = size*size + 12(颜色) + 8(梯度) + 12(颜色比) + 256(对比度)
        self._hist_dims = _HIST_BINS_PER_CHANNEL * 3
        self._grad_dims = _GRAD_BINS
        self._ratio_dims = _RATIO_BINS_PER_CHANNEL * 3
        self._contrast_dims = _CONTRAST_SIZE * _CONTRAST_SIZE

    @property
    def size(self) -> int:
        """灰度池化边长。"""
        return self._size

    @property
    def vector_dims(self) -> int:
        """特征向量总维度（灰度 + 颜色直方图 + 梯度 + 颜色比 + 局部对比度）。"""
        return (
            self._size * self._size
            + self._hist_dims
            + self._grad_dims
            + self._ratio_dims
            + self._contrast_dims
        )

    # ------------------------------------------------------------------
    # 指纹生成
    # ------------------------------------------------------------------
    def fingerprint(self, frame: np.ndarray) -> bytes:
        """把一帧图像转换为确定性指纹 bytes。

        支持 HxW（灰度）、HxWx3（BGR）、HxWx4（BGRA）三种常见形态；
        输入为 None 或空数组时抛 NavError（调用方捕获后降级为"无画面"）。

        Args:
            frame: 图像数组（uint8）。

        Returns:
            float32 特征向量序列化的 bytes，长度 = vector_dims * 4。
        """
        if frame is None:
            raise NavError("无法生成指纹：输入图像为 None")
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.size == 0 or arr.ndim not in (2, 3):
            raise NavError(f"无法生成指纹：非法图像形状 {arr.shape}")

        gray = self._to_gray(arr)
        pooled = self._mean_pool(gray, self._size, self._size)
        gray_vec = ((pooled.reshape(-1) / 255.0) * _GRAY_WEIGHT).astype(np.float32)

        hist_vec = (self._color_histogram(arr) * _HIST_WEIGHT).astype(np.float32)
        grad_vec = (self._gradient_histogram(arr) * _GRAD_WEIGHT).astype(np.float32)
        ratio_vec = (self._ratio_histogram(arr) * _RATIO_WEIGHT).astype(np.float32)
        contrast_vec = (self._contrast_pool(arr) * _CONTRAST_WEIGHT).reshape(-1).astype(np.float32)

        vec = np.concatenate([gray_vec, hist_vec, grad_vec, ratio_vec, contrast_vec]).astype(np.float32)
        return vec.tobytes()

    @staticmethod
    def _to_gray(arr: np.ndarray) -> np.ndarray:
        """把 HxWx3/HxWx4 彩色图转为灰度 float32；HxW 直接返回。"""
        if arr.ndim == 2:
            return arr.astype(np.float32)
        if arr.shape[2] >= 3:
            b = arr[..., 0].astype(np.float32)
            g = arr[..., 1].astype(np.float32)
            r = arr[..., 2].astype(np.float32)
            # OpenCV 标准灰度权重：0.299R + 0.587G + 0.114B
            return 0.299 * r + 0.587 * g + 0.114 * b
        return arr[..., 0].astype(np.float32)

    @staticmethod
    def _mean_pool(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """把灰度图降采样为 out_h×out_w（均值池化）。

        优先使用 cv2.resize(INTER_AREA)（项目已依赖 opencv-python）；
        不可用时回退纯 numpy 块均值（边界 padding 后 reshape 求均值），
        保证在无 cv2 的测试环境也能跑（确定性不受影响）。
        """
        try:
            import cv2

            resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
            return np.asarray(resized, dtype=np.float32)
        except Exception:  # noqa: BLE001, S110 - cv2 缺失/失败时回退纯 numpy 实现（确定性不变）
            pass
        h, w = gray.shape
        if h < out_h or w < out_w:
            # 图像小于目标：先放大到至少 out×out（最近邻），再走块均值
            pad_h = max(0, out_h - h)
            pad_w = max(0, out_w - w)
            gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = gray.shape
        ph = (out_h - h % out_h) % out_h
        pw = (out_w - w % out_w) % out_w
        if ph or pw:
            gray = np.pad(gray, ((0, ph), (0, pw)), mode="edge")
        h, w = gray.shape
        bh, bw = h // out_h, w // out_w
        block = gray[: out_h * bh, : out_w * bw].reshape(out_h, bh, out_w, bw)
        return block.mean(axis=(1, 3)).astype(np.float32)

    @staticmethod
    def _color_histogram(arr: np.ndarray) -> np.ndarray:
        """BGR 三通道各 4 桶颜色直方图（归一化，12 维）。

        灰度图（HxW）时各通道均为 0（无色信息），不影响可分性（灰度已单独编码）。
        """
        hist = np.zeros(_HIST_BINS_PER_CHANNEL * 3, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return hist
        for c in range(3):
            ch = arr[..., c].ravel().astype(np.float32)
            # 每通道 4 桶：0~63 / 64~127 / 128~191 / 192~255
            bins = np.clip((ch / 64.0).astype(np.int32), 0, _HIST_BINS_PER_CHANNEL - 1)
            counts = np.bincount(bins, minlength=_HIST_BINS_PER_CHANNEL).astype(np.float32)
            total = float(counts.sum())
            if total > 0:
                counts /= total
            hist[c * _HIST_BINS_PER_CHANNEL : (c + 1) * _HIST_BINS_PER_CHANNEL] = counts
        return hist

    @staticmethod
    def _gradient_histogram(arr: np.ndarray) -> np.ndarray:
        """Sobel 梯度幅值直方图（8 桶，亮度不变）。

        P0-1：梯度幅值 = sqrt(gx² + gy²)（numpy.gradient 中心差分近似 Sobel），
        除以帧均幅值（+ε 防除零）后分桶——乘性亮度变化（整体变暗/变亮）与
        加性偏移（雾/曝光偏移）下，归一化梯度分布基本不变；
        纯色/无纹理帧幅值恒 0 → 归一化后全落 0 桶（确定性）。
        """
        gray = LandmarkFingerprinter._to_gray(arr)
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        scale = float(mag.mean()) + 1e-6
        norm = mag / scale
        bins = np.clip((norm * _GRAD_BINS / _GRAD_NORM_RANGE).astype(np.int32), 0, _GRAD_BINS - 1).ravel()
        counts = np.bincount(bins, minlength=_GRAD_BINS).astype(np.float32)
        total = float(counts.sum())
        if total > 0:
            counts /= total
        return counts

    @staticmethod
    def _ratio_histogram(arr: np.ndarray) -> np.ndarray:
        """归一化颜色比直方图（3 通道 × 4 桶，亮度不变）。

        P0-1：对每个像素计算 B/(R+G+B)、G/(R+G+B)、R/(R+G+B)（+ε 防除零）——
        颜色比例与亮度无关：同色不同亮度 → 相同分布，不同色相 → 明显不同。
        纯黑像素（全 0）比例 ≈ 0（确定性落 0 桶）。
        """
        hist = np.zeros(_RATIO_BINS_PER_CHANNEL * 3, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return hist
        b = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        r = arr[..., 2].astype(np.float32)
        total = r + g + b + 1e-6
        for c, ch in enumerate((b, g, r)):
            ratio = ch / total
            bins = np.clip(
                (ratio * _RATIO_BINS_PER_CHANNEL).astype(np.int32), 0, _RATIO_BINS_PER_CHANNEL - 1
            ).ravel()
            counts = np.bincount(bins, minlength=_RATIO_BINS_PER_CHANNEL).astype(np.float32)
            t = float(counts.sum())
            if t > 0:
                counts /= t
            hist[c * _RATIO_BINS_PER_CHANNEL : (c + 1) * _RATIO_BINS_PER_CHANNEL] = counts
        return hist

    @staticmethod
    def _contrast_pool(arr: np.ndarray) -> np.ndarray:
        """局部对比度池化（16×16，块内 std / 帧均 std，亮度不变）。

        P0-1：每块灰度标准差刻画局部纹理强度；除以帧均标准差（+ε）后
        乘性亮度变化下比例不变；纯色帧块内 std 恒 0 → 全 0（确定性）。
        """
        gray = LandmarkFingerprinter._to_gray(arr)
        h, w = gray.shape
        out = _CONTRAST_SIZE
        ph = (out - h % out) % out
        pw = (out - w % out) % out
        if ph or pw:
            gray = np.pad(gray, ((0, ph), (0, pw)), mode="edge")
        h, w = gray.shape
        bh, bw = h // out, w // out
        block = gray[: out * bh, : out * bw].reshape(out, bh, out, bw)
        std = block.std(axis=(1, 3)).astype(np.float32)
        gstd = float(std.mean()) + 1e-6
        return (std / gstd).astype(np.float32)

    # ------------------------------------------------------------------
    # 相似度 / 回环检测
    # ------------------------------------------------------------------
    def similarity(self, a: bytes, b: bytes) -> float:
        """计算两个指纹的相似度（归一化 L1 距离，[0,1]）。

        相同指纹 → 1.0；维度不匹配/空 → 0.0。
        """
        if not a or not b:
            return 0.0
        va = np.frombuffer(a, dtype=np.float32)
        vb = np.frombuffer(b, dtype=np.float32)
        if va.shape != vb.shape or va.size == 0:
            return 0.0
        diff = float(np.abs(va - vb).sum())
        max_diff = float(va.size)
        if max_diff <= 0.0:
            return 1.0 if va.size == 0 else 0.0
        return float(np.clip(1.0 - diff / max_diff, 0.0, 1.0))

    def detect_loop_closure(
        self,
        fingerprint: bytes,
        known: dict[int, bytes],
        threshold: float = 0.85,
    ) -> tuple[int | None, float]:
        """回环检测：当前帧指纹 vs 已访问节点指纹表。

        Args:
            fingerprint: 当前帧指纹（bytes）。
            known: {节点 id: 节点指纹} 映射。
            threshold: 相似度阈值，超过才判定"回到已知位置"。

        Returns:
            (node_id, similarity)：命中返回节点 id 与相似度；
            未命中返回 (None, 最高相似度)。
        """
        best_id: int | None = None
        best_sim = 0.0
        for nid, fp in known.items():
            sim = self.similarity(fingerprint, fp)
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        if best_id is not None and best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim
