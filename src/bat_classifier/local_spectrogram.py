"""局部查看的数据计算与有界内存缓存；不创建或读取中间 PNG。"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np

from .audio_io import load_audio_interval
from .visualization import (
    MAX_DETAIL_SECONDS, SPECTROGRAM_CMAP, _spectrogram_display_limits,
    _waveform_trace, compute_overview_spectrum,
)


class _MemoryCache:
    def __init__(self, limit: int):
        self.limit, self.nbytes = limit, 0
        self.items = OrderedDict()

    def get(self, key):
        entry = self.items.get(key)
        if entry is not None:
            self.items.move_to_end(key)
            return entry[0]
        return None

    def put(self, key, value, nbytes: int):
        previous = self.items.pop(key, None)
        if previous is not None:
            self.nbytes -= previous[1]
        if nbytes > self.limit:
            return
        self.items[key] = (value, nbytes)
        self.nbytes += nbytes
        while self.nbytes > self.limit:
            _, (_, size) = self.items.popitem(last=False)
            self.nbytes -= size

    def clear(self):
        self.items.clear()
        self.nbytes = 0


@dataclass(frozen=True)
class LocalSpectrogramData:
    meta: dict
    power_db: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    wave_times: np.ndarray
    wave_values: np.ndarray
    rgb: np.ndarray
    # 彩色频谱矩阵的单元格外沿（秒、Hz）。
    image_extent: tuple[float, float, float, float]
    title: str

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.power_db, self.frequencies, self.times,
                                     self.wave_times, self.wave_values, self.rgb))


class LocalSpectrogramCache:
    """每个查看窗口至多缓存 16 MiB 音频和 64 MiB 选区结果，关闭时释放。"""
    def __init__(self, audio_limit=16 * 1024**2, result_limit=64 * 1024**2):
        self.audio = _MemoryCache(audio_limit)
        self.results = _MemoryCache(result_limit)
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self.audio.clear()
            self.results.clear()

    def compute(self, source: dict, low: float, high: float,
                frequency_limits: tuple[float, float]) -> LocalSpectrogramData:
        if not np.isfinite([low, high, *frequency_limits]).all() or high <= low:
            raise ValueError("时间选区无效")
        if high - low > MAX_DETAIL_SECONDS + 1e-9:
            raise ValueError(f"局部精细查看请选择 {MAX_DETAIL_SECONDS:g} 秒以内的范围")
        if frequency_limits[1] <= frequency_limits[0]:
            raise ValueError("频率范围无效")
        path = Path(source["path"]).resolve()
        stat = path.stat()  # 每次请求都检查，文件更新/丢失后不命中旧缓存。
        offset = float(source.get("offset_s", 0.))
        preprocessed = source.get("preprocessed") is True
        file_key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                    offset, preprocessed)
        audio_key = (*file_key, float(low), float(high))
        result_key = (*audio_key, *frequency_limits)
        with self._lock:
            result = self.results.get(result_key)
            if result is not None:
                return result
            audio = self.audio.get(audio_key)
            if audio is None:
                audio = load_audio_interval(path, low + offset, high + offset,
                                            preprocessed=preprocessed)
                self.audio.put(audio_key, audio, audio[0].nbytes)
            y, sr, start = audio
            db, times, frequencies, power, n_fft, hop = compute_overview_spectrum(y, sr, 8000)
            if not power.size:
                raise ValueError("选区没有可绘制的声谱")
            if np.any(power > 0.):
                vmin, vmax = _spectrogram_display_limits(db, frequencies, frequency_limits)
                display = db
            else:
                vmin, vmax = -80., 0.
                display = np.full_like(db, -80.)
            times = times + start - offset
            trace = _waveform_trace(y, sr)
            wave_times, wave_values = trace
            wave_times = wave_times + start - offset
            rgb = np.ascontiguousarray(SPECTROGRAM_CMAP(
                np.clip((display - vmin) / (vmax - vmin), 0., 1.), bytes=True,
            )[::-1, :, :3])
            df = sr / n_fft
            extent = (float(times[0] - hop / sr / 2), float(times[-1] + hop / sr / 2),
                      float(frequencies[0] - df / 2), float(frequencies[-1] + df / 2))
            duration = len(y) / sr
            meta = {
                "axes_bbox": [.08, .09, .905, .38],
                "waveform_bbox": [.08, .49, .905, .44],
                "freq_ylim": list(frequency_limits),
                "time_xlim": [start - offset, start - offset + duration],
                "spectrogram_colormap": SPECTROGRAM_CMAP.name,
                "waveform_mode": "wav-waveform", "waveform_ylim": [-1.08, 1.08],
                "time_axis_unit": "ms" if duration <= .25 else "s",
                "frequency_axis_unit": "kHz", "display_db_range": [vmin, vmax],
                "source_audio": dict(source), "stft_n_fft": n_fft,
                "stft_hop_length": hop, "time_step_s": hop / sr, "time_bins": len(times),
            }
            title = f"{path.name} | 局部声谱 | {low:.6f}–{high:.6f} s"
            result = LocalSpectrogramData(meta, db, frequencies, times, wave_times,
                                          wave_values, rgb, extent, title)
            # 缓存结果只读，界面/后续请求不会意外修改共用数组。
            for a in (db, frequencies, times, wave_times, wave_values, rgb):
                a.flags.writeable = False
            self.results.put(result_key, result, result.nbytes)
            return result
