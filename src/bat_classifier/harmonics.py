"""谐波轨迹分组：只合并时间、倍率、轮廓和包络同时提供证据的轨迹。"""

from __future__ import annotations

import numpy as np

from .config import (
    SEPARATION_HARMONIC_MAX_ORDER,
    SEPARATION_HARMONIC_MIN_COMMON_FRAMES,
    SEPARATION_HARMONIC_MIN_CONTOUR_CORR,
    SEPARATION_HARMONIC_MIN_ENVELOPE_CORR,
    SEPARATION_HARMONIC_MIN_OVERLAP,
    SEPARATION_HARMONIC_MIN_RATIO_COVERAGE,
    SEPARATION_HARMONIC_RATIO_MAD,
    SEPARATION_HARMONIC_TOLERANCE,
)


def harmonic_track_evidence(first: dict, second: dict) -> dict | None:
    """低频已观测轨迹作倍率参照，不把它直接认定为真实基频。"""
    lower, upper = sorted(
        (first, second),
        key=lambda track: float(np.median(track["frequencies_hz"])),
    )
    low_frames = np.asarray(lower["frames"], dtype=int)
    high_frames = np.asarray(upper["frames"], dtype=int)
    common = np.intersect1d(low_frames, high_frames)
    if len(common) < SEPARATION_HARMONIC_MIN_COMMON_FRAMES:
        return None
    overlap = len(common) / max(len(low_frames), len(high_frames))
    if overlap < SEPARATION_HARMONIC_MIN_OVERLAP:
        return None
    span = max(
        int(low_frames[-1] - low_frames[0] + 1),
        int(high_frames[-1] - high_frames[0] + 1),
    )
    edge_tolerance = max(2, int(np.ceil(0.20 * span)))
    if (
        abs(int(low_frames[0] - high_frames[0])) > edge_tolerance
        or abs(int(low_frames[-1] - high_frames[-1])) > edge_tolerance
    ):
        return None

    low_hz = np.interp(common, low_frames, lower["frequencies_hz"])
    high_hz = np.interp(common, high_frames, upper["frequencies_hz"])
    if (
        not np.isfinite(low_hz).all()
        or not np.isfinite(high_hz).all()
        or np.any(low_hz <= 0)
    ):
        return None
    ratios = high_hz / low_hz
    median_ratio = float(np.median(ratios))
    order = int(round(median_ratio))
    if not 2 <= order <= SEPARATION_HARMONIC_MAX_ORDER:
        return None
    ratio_coverage = float(np.mean(
        np.abs(ratios / order - 1.0) <= SEPARATION_HARMONIC_TOLERANCE
    ))
    ratio_mad = float(np.median(np.abs(ratios - median_ratio)) / order)
    if (
        ratio_coverage < SEPARATION_HARMONIC_MIN_RATIO_COVERAGE
        or ratio_mad > SEPARATION_HARMONIC_RATIO_MAD
    ):
        return None

    # 平直 CF 轨迹不能单凭相关系数认定同源，仍需有变化的包络证据。
    scaled_high = high_hz / order
    contour_corr = None
    low_span = float(np.ptp(low_hz))
    high_span = float(np.ptp(scaled_high))
    if max(low_span, high_span) > 750.0:
        if min(low_span, high_span) < 375.0:
            return None
        contour_corr = float(np.corrcoef(low_hz, scaled_high)[0, 1])
        if (
            not np.isfinite(contour_corr)
            or contour_corr < SEPARATION_HARMONIC_MIN_CONTOUR_CORR
        ):
            return None
    residual = scaled_high - low_hz
    frequency_tolerance_hz = max(500.0, float(np.median(low_hz)) * 0.015)
    if abs(float(np.median(residual))) > frequency_tolerance_hz:
        return None
    residual_deviation = float(
        np.percentile(np.abs(residual - np.median(residual)), 90)
    )
    if residual_deviation > frequency_tolerance_hz:
        return None

    # 同时刻比较包络，不通过延迟对齐来把回声当成谐波。
    low_power = np.interp(common, low_frames, lower["powers"])
    high_power = np.interp(common, high_frames, upper["powers"])
    low_db = 10.0 * np.log10(np.maximum(low_power, 1e-18))
    high_db = 10.0 * np.log10(np.maximum(high_power, 1e-18))
    if min(float(np.std(low_db)), float(np.std(high_db))) < 1.0:
        return None
    envelope_corr = float(np.corrcoef(low_db, high_db)[0, 1])
    if (
        not np.isfinite(envelope_corr)
        or envelope_corr < SEPARATION_HARMONIC_MIN_ENVELOPE_CORR
    ):
        return None
    return {
        "order": order,
        "lower_frequency_khz": float(np.median(low_hz)) / 1000.0,
        "upper_frequency_khz": float(np.median(high_hz)) / 1000.0,
        "common_frames": int(len(common)),
        "overlap": float(overlap),
        "ratio_coverage": ratio_coverage,
        "ratio_mad": ratio_mad,
        "contour_correlation": contour_corr,
        "envelope_correlation": envelope_corr,
    }


def group_harmonic_tracks(tracks: list[dict]) -> list[dict]:
    """仅对已观测参照作直接匹配，不通过链式关系扩张谐波组。"""
    families = []
    for track in sorted(tracks, key=lambda item: float(item["median_frequency_hz"])):
        matches = []
        for family in families:
            evidence = harmonic_track_evidence(family["anchor"], track)
            if evidence is not None:
                matches.append((family, evidence))
        if len(matches) == 1 and all(
            previous["order"] != matches[0][1]["order"]
            for previous in matches[0][0]["evidence"]
        ):
            family, evidence = matches[0]
            family["members"].append(track)
            family["evidence"].append(evidence)
        else:
            # 归属或阶次不明确时保留为独立候选。
            families.append({"anchor": track, "members": [track], "evidence": []})

    grouped = []
    for family in families:
        members = family["members"]
        representative = dict(max(members, key=lambda item: float(item["total_power"])))
        representative["total_power"] = float(
            sum(float(member["total_power"]) for member in members)
        )
        representative["_harmonic_members"] = members
        representative["_harmonic_evidence"] = family["evidence"]
        grouped.append(representative)
    return sorted(grouped, key=lambda item: float(item["total_power"]), reverse=True)


def harmonic_component_metadata(track: dict) -> dict:
    members = track.get("_harmonic_members", [track])
    return {
        "harmonic_group_size": len(members),
        "harmonic_frequencies_khz": [
            float(member["median_frequency_hz"]) / 1000.0 for member in members
        ],
        "harmonic_evidence": track.get("_harmonic_evidence", []),
    }
