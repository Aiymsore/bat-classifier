"""共享 pulse 检测结果建表与分量特征提取。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import extract_pulse_features
from .separation import (
    _unchanged_component,
    save_separated_pulse,
    separate_simultaneous_pulse_components,
)
from .config import SAVE_SEPARATED_PULSE_AUDIO


def create_pulse_rows(
    audio_path: Path,
    y: np.ndarray,
    sr: int,
    pulses: list[dict[str, float]],
    separated_dir: Path | None = None,
    max_pulses: int | None = None,
    save_separated_audio: bool = True,
    include_template_track: bool = False,
    separation_window_mode: str = "fixed",
    harmonic_grouping_mode: str = "basic",
    component_audio_sink: dict[tuple[int, int], np.ndarray] | None = None,
) -> list[dict[str, float | int | str | bool]]:
    """提取每个分离分量的特征；默认处理全部 pulse，导出不影响计数。"""
    if max_pulses and len(pulses) > max_pulses:
        pulses = sorted(
            pulses,
            key=lambda pulse: float(
                pulse.get("snr_db", -999.0)
            ),
            reverse=True,
        )[:max_pulses]
        pulses = sorted(
            pulses,
            key=lambda pulse: float(
                pulse["start_s"]
            ),
        )

    rows: list[
        dict[str, float | int | str | bool]
    ] = []

    pulse_id = 0

    for candidate_id, pulse in enumerate(
        pulses,
        start=1,
    ):
        pulse_start_sample = max(
            0,
            int(round(
                float(pulse["start_s"]) * sr
            )),
        )
        pulse_end_sample = min(
            len(y),
            int(round(
                float(pulse["end_s"]) * sr
            )),
        )
        pulse_y = y[
            pulse_start_sample:pulse_end_sample
        ]

        try:
            components = (
                separate_simultaneous_pulse_components(
                    pulse_y,
                    sr,
                    window_mode=separation_window_mode,
                    harmonic_grouping_mode=harmonic_grouping_mode,
                )
            )
        except Exception:
            components = [
                _unchanged_component(pulse_y, 0.0)
            ]

        for component in components:
            component_audio = np.asarray(
                component["audio"],
                dtype=np.float32,
            )
            if len(component_audio) < 32:
                continue

            cached_features = component.get(
                "_preview_features"
            )
            if cached_features is not None:
                features = dict(cached_features)
            else:
                try:
                    features = extract_pulse_features(
                        component_audio,
                        sr,
                    )
                except Exception:
                    continue

            template_track = None
            if include_template_track:
                try:
                    # 模板匹配使用 32 点归一时间频率轨迹。
                    from .template_matching import extract_template_descriptor

                    template_track = extract_template_descriptor(
                        component_audio,
                        sr,
                    ).get("template_track_khz")
                except Exception:
                    template_track = None

            relative_db = float(
                component["track_relative_db"]
            )
            component_start_sample = (
                pulse_start_sample
                + int(component["start_sample"])
            )
            component_end_sample = (
                pulse_start_sample
                + int(component["end_sample"])
            )
            component_pulse = dict(pulse)
            component_pulse["start_s"] = (
                component_start_sample / sr
            )
            component_pulse["end_s"] = (
                component_end_sample / sr
            )
            component_pulse["duration_ms"] = (
                component_end_sample
                - component_start_sample
            ) / sr * 1000
            component_pulse["snr_db"] = max(
                0.0,
                float(pulse["snr_db"])
                + relative_db,
            )

            pulse_id += 1

            pulse_file = ""
            if (
                save_separated_audio
                and SAVE_SEPARATED_PULSE_AUDIO
                and bool(
                    component["separation_applied"]
                )
            ):
                pulse_file = save_separated_pulse(
                    audio_path,
                    candidate_id,
                    int(component["component_id"]),
                    component_audio,
                    sr,
                    output_dir=separated_dir,
                )

            row = {
                    "source_file": audio_path.name,
                    "source_path": str(audio_path),
                    "species": "unlabeled",
                    "pulse_id": pulse_id,
                    "candidate_id": candidate_id,
                    "component_id": int(
                        component["component_id"]
                    ),
                    "component_count": int(
                        component["component_count"]
                    ),
                    "separation_applied": bool(
                        component["separation_applied"]
                    ),
                    "separation_confidence": float(
                        component[
                            "separation_confidence"
                        ]
                    ),
                    "harmonic_grouping_mode": harmonic_grouping_mode,
                    "harmonic_group_size": int(component.get("harmonic_group_size", 1)),
                    "harmonic_frequencies_khz": json.dumps(
                        component.get("harmonic_frequencies_khz", []),
                        allow_nan=False,
                    ),
                    "harmonic_evidence": json.dumps(
                        component.get("harmonic_evidence", []),
                        allow_nan=False,
                    ),
                    "track_median_freq_khz": float(
                        component[
                            "track_median_freq_khz"
                        ]
                    ),
                    "track_relative_db": relative_db,
                    "pulse_file": pulse_file,
                    "candidate_start_s": float(
                        pulse["candidate_start_s"]
                    ),
                    "candidate_end_s": float(
                        pulse["candidate_end_s"]
                    ),
                    "start_s": float(
                        component_pulse["start_s"]
                    ),
                    "end_s": float(
                        component_pulse["end_s"]
                    ),
                    "duration_ms": float(
                        component_pulse["duration_ms"]
                    ),
                    "snr_db": float(
                        component_pulse["snr_db"]
                    ),
                    "core_threshold_db": float(
                        pulse["core_threshold_db"]
                    ),
                    **features,
                }
            if template_track is not None:
                row["template_track_khz"] = template_track
            if component_audio_sink is not None and bool(component["separation_applied"]):
                component_audio_sink[(candidate_id, int(component["component_id"]))] = component_audio
            rows.append(row)

    return rows
