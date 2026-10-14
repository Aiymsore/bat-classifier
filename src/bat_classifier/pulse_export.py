"""独立 pulse WAV 导出，保留已分离的音频。"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio_io import load_audio
from .config import BASE_DIR

def export_pulse_audio(
    row: "pd.Series | dict",
    destination: Path,
    audio_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> str:
    """把一个脉冲写到 destination，返回写入路径（相对 BASE_DIR，无法相对时为绝对路径）。

    代表脉冲必须沿用已分离波形（row["pulse_file"]），绝不能再从原始
    混合录音切一遍把两个声源混回去；只有未发生分离的普通脉冲才回退
    到按 start_s/end_s 从源录音切割。

    audio_cache 由调用方提供，以复用当前音频块。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        audio_cache if audio_cache is not None else {}
    )

    pulse_path_text = str(row.get("pulse_file", "") or "")
    source_path_text = str(row.get("source_path", "") or "")

    separated_source = None
    if pulse_path_text:
        separated_source = Path(pulse_path_text)
        if not separated_source.is_absolute():
            separated_source = BASE_DIR / separated_source

    if separated_source is not None and separated_source.exists():
        shutil.copy2(separated_source, destination)
    elif source_path_text:
        if source_path_text not in cache:
            cache[source_path_text] = load_audio(
                Path(source_path_text)
            )
        source_y, source_sr = cache[source_path_text]
        start_sample = max(
            0,
            int(round(float(row["start_s"]) * source_sr)),
        )
        end_sample = min(
            len(source_y),
            int(round(float(row["end_s"]) * source_sr)),
        )
        segment = source_y[start_sample:end_sample]
        if len(segment) < 32:
            return ""
        sf.write(
            destination,
            segment.astype(np.float32),
            source_sr,
            subtype="PCM_16",
        )
    else:
        return ""

    try:
        return str(destination.relative_to(BASE_DIR))
    except ValueError:
        return str(destination)
