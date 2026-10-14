"""Avisoft SASLab `.son`（SONA）文件解析器（① 直接查看用）。

`.son` 是 Avisoft SASLab 保存的声谱图工程文件：头部记录采样率、帧移、
频率分辨率、帧数与频点数，其后是逐帧 FFT 幅度谱（int16），末尾是一段
ASCII 注释文本。它不含原始波形——声音本体在同名 `.wav` 里。

本模块把频谱缓存读成 (帧 × 频点) 幅度矩阵，供线性声谱图直接出图，
查看不再依赖 Avisoft 软件。

布局（以 CF1_47K1.son 逆向验证，字段偏移固定）::

    0   b"SONA"                 魔数
    4   u32  采样率（Hz）
    8   f32  帧移（秒）
    12  f32  频率分辨率（Hz/bin）
    24  u32  帧数
    28  u32  频点数（= FFT/2+1，如 1024 点 FFT → 513）
    88  u32  尾部注释字节数
    ... 帧数 × 频点数 × int16 幅度值
    末尾 ASCII 注释文本
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

_MAGIC = b"SONA"
_OFF_SAMPLE_RATE = 4
_OFF_WINDOW_S = 8
_OFF_BIN_HZ = 12
_OFF_N_FRAMES = 24
_OFF_N_BINS = 28
_OFF_COMMENT_LEN = 88
_HEADER_MIN = _OFF_COMMENT_LEN + 4


@dataclass
class SonSnapshot:
    """`.son` 频谱缓存：幅度矩阵 + 坐标轴定义。"""

    path: Path
    sample_rate: int
    window_s: float
    bin_hz: float
    magnitudes: np.ndarray  # (n_frames, n_bins)，float64，非负
    comment: str

    @property
    def n_frames(self) -> int:
        return int(self.magnitudes.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.magnitudes.shape[1])

    def frequencies(self) -> np.ndarray:
        """频点轴（Hz），长度 n_bins。"""
        return np.arange(self.n_bins, dtype=np.float64) * self.bin_hz

    def times(self) -> np.ndarray:
        """帧时间轴（秒），长度 n_frames。"""
        return np.arange(self.n_frames, dtype=np.float64) * self.window_s

    def power_db(self) -> np.ndarray:
        """最大值归一化功率谱（dB），形状 (n_bins, n_frames) 供绘图直接使用。"""
        mag = np.maximum(self.magnitudes.T.astype(np.float64), 1e-9)
        power = mag ** 2
        return 10.0 * np.log10(power / float(power.max()))


def _trailing_text_len(raw: bytes, data_bytes: int) -> int:
    """从文件尾部数出 ASCII 注释长度（回退方案）。"""
    limit = len(raw) - data_bytes
    count = 0
    index = len(raw) - 1
    while index >= limit and count < 4096:
        byte = raw[index]
        if not (32 <= byte <= 126 or byte in (9, 10, 13)):
            break
        count += 1
        index -= 1
    return count


def _locate_frames(
    raw: bytes, n_frames: int, n_bins: int
) -> tuple[int, int]:
    """返回 (帧数据起点, 尾部注释字节数)，失败抛 ValueError。"""
    data_bytes = n_frames * n_bins * 2
    if data_bytes <= 0 or _HEADER_MIN + data_bytes > len(raw):
        raise ValueError(
            f".son 尺寸不匹配：帧数据需要 {data_bytes} 字节，文件只有 {len(raw)}"
        )

    candidates: list[int] = []
    if len(raw) >= _OFF_COMMENT_LEN + 4:
        comment_len = struct.unpack_from(
            "<I", raw, _OFF_COMMENT_LEN
        )[0]
        if 0 <= comment_len <= 4096:
            candidates.append(int(comment_len))
    trailing_len = _trailing_text_len(raw, data_bytes)
    candidates.append(trailing_len)

    for comment_len in candidates:
        # 注释长度为 0 只在确实没有尾部文本时才可信，
        # 否则会让"空注释"永远抢先命中，掩盖真实注释长度。
        if comment_len == 0 and trailing_len > 0:
            continue
        start = len(raw) - comment_len - data_bytes
        if start < _HEADER_MIN:
            continue
        frames = np.frombuffer(
            raw[start : start + data_bytes], dtype="<i2"
        )
        if frames.size == n_frames * n_bins and (frames >= 0).all():
            return start, comment_len

    raise ValueError("无法定位 .son 的帧数据（布局超出已验证范围）")


def read_son(path: Path | str) -> SonSnapshot:
    """读取 `.son`，返回频谱缓存快照；格式不符抛 ValueError。"""
    source = Path(path)
    raw = source.read_bytes()
    if len(raw) < _HEADER_MIN or raw[:4] != _MAGIC:
        raise ValueError(f"{source.name} 不是 SONA 格式的 .son 文件")

    sample_rate = struct.unpack_from("<I", raw, _OFF_SAMPLE_RATE)[0]
    window_s = struct.unpack_from("<f", raw, _OFF_WINDOW_S)[0]
    bin_hz = struct.unpack_from("<f", raw, _OFF_BIN_HZ)[0]
    n_frames = struct.unpack_from("<I", raw, _OFF_N_FRAMES)[0]
    n_bins = struct.unpack_from("<I", raw, _OFF_N_BINS)[0]

    if not (1 <= n_frames <= 10_000_000):
        raise ValueError(f".son 帧数异常：{n_frames}")
    if not (2 <= n_bins <= 65_536):
        raise ValueError(f".son 频点数异常：{n_bins}")
    if sample_rate <= 0 or bin_hz <= 0 or window_s <= 0:
        raise ValueError(
            f".son 参数异常：sr={sample_rate} bin={bin_hz} win={window_s}"
        )

    start, comment_len = _locate_frames(raw, n_frames, n_bins)
    magnitudes = (
        np.frombuffer(
            raw[start : start + n_frames * n_bins * 2], dtype="<i2"
        )
        .astype(np.float64)
        .reshape(n_frames, n_bins)
    )
    comment = raw[len(raw) - comment_len :].decode("ascii", "replace").strip()

    return SonSnapshot(
        path=source,
        sample_rate=int(sample_rate),
        window_s=float(window_s),
        bin_hz=float(bin_hz),
        magnitudes=magnitudes,
        comment=comment,
    )
