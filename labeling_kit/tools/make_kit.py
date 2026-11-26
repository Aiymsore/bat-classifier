"""生成人工标注工具包：分层抽取 2 秒窗口，输出空白声谱图、程序框对照图、音频片段和待填表格。

用法（在项目根目录）：
    .venv/Scripts/python.exe labeling_kit/tools/make_kit.py <程序结果目录> <人工计数 CSV>
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import librosa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

KIT = ROOT / "labeling_kit"
RAW = ROOT.parent / "声波文件" / "2025罗汉洞" / "20250807"
WINDOW_S = 2.0
ROW_S = 0.5
FMIN, FMAX = 35.0, 115.0
CLASSES = ["CF1_47K", "CF2_58K", "CF3_66K", "CF4_72K", "CF5_83K", "CF6_108K", "CFFM1_67K", "CFFM2_86K"]
REFERENCE_KHZ = {"CF1 47": 46.4, "CF2 59": 59.1, "CF3/CFFM1 66-67": 66.6, "CF4 72": 71.6,
                 "CF5 80-86 / CFFM2 85": 84.0, "CF6 104": 104.2}
COLORS = {"CF1_47K": "#ff4d4d", "CF2_58K": "#ffa500", "CF3_66K": "#ffff00", "CF4_72K": "#7fff00",
          "CF5_83K": "#00ffff", "CF6_108K": "#ff00ff", "CFFM1_67K": "#ffffff", "CFFM2_86K": "#ff69b4"}
RNG = np.random.default_rng(20261104)


def file_duration(name: str) -> float:
    return sf.info(str(RAW / name)).duration


def best_window(calls: pd.DataFrame, duration: float) -> float:
    """该文件中该类程序检出最多的 2 秒窗口起点（步长 0.25 s，避开首尾 0.5 s）。"""
    starts = np.arange(0.5, max(0.5, duration - WINDOW_S - 0.5), 0.25)
    counts = [((calls.start_s < s + WINDOW_S) & (calls.end_s > s)).sum() for s in starts]
    return float(starts[int(np.argmax(counts))])


def select_windows(pred: pd.DataFrame, manual: pd.DataFrame) -> list[dict]:
    bat = pred[pred.gate_status == "bat"]
    used: set[str] = set()
    windows: list[dict] = []
    for cls in CLASSES:
        ranked = manual[cls].sort_values(ascending=False)
        ranked = ranked[ranked > 0]
        # 密集：人工计数最多的文件；中等：人工计数居中的文件。
        picks = [("密集", ranked.index.tolist())]
        middle = ranked.iloc[len(ranked) // 3: 2 * len(ranked) // 3].index.tolist()
        picks.append(("中等", list(RNG.permutation(middle)) + ranked.index.tolist()))
        for stratum, files in picks:
            name = next((f for f in files if f not in used), None)
            if name is None:
                continue
            used.add(name)
            duration = file_duration(name)
            calls = bat[(bat.segment_file == name) & (bat.predicted_label == cls)]
            start = best_window(calls, duration) if len(calls) else float(RNG.uniform(0.5, duration - WINDOW_S - 0.5))
            windows.append({"stratum": f"{cls} {stratum}", "file": name, "start": round(start, 2),
                            "manual_named": int(manual.at[name, cls])})
    # 安静窗口：随机文件、随机位置，用来查漏检。
    rest = [f for f in manual.index if f not in used]
    for name in RNG.permutation(rest)[:4]:
        duration = file_duration(name)
        windows.append({"stratum": "随机安静", "file": str(name),
                        "start": round(float(RNG.uniform(0.5, duration - WINDOW_S - 0.5)), 2),
                        "manual_named": int(manual.loc[name].sum())})
    for index, window in enumerate(windows, start=1):
        window["id"] = f"W{index:02d}"
        in_window = bat[(bat.segment_file == window["file"]) & (bat.start_s < window["start"] + WINDOW_S)
                        & (bat.end_s > window["start"])]
        window["program_bat"] = int(len(in_window))
    return windows


def draw(window: dict, pred: pd.DataFrame | None, out: Path) -> None:
    t0 = window["start"]
    y, sr = sf.read(str(RAW / window["file"]), start=int(t0 * 250000), stop=int((t0 + WINDOW_S) * 250000))
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        y = y.mean(axis=1)
    rows = int(WINDOW_S / ROW_S)
    fig, axes = plt.subplots(rows, 1, figsize=(24, 4.2 * rows))
    power = np.abs(librosa.stft(y - y.mean(), n_fft=512, hop_length=32)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512) / 1000
    times = np.arange(power.shape[1]) * 32 / sr
    band = (freqs >= FMIN) & (freqs <= FMAX)
    ref = np.percentile(power[band], 99.7)
    db = 10 * np.log10(power / ref + 1e-12)
    for row, ax in enumerate(axes):
        lo, hi = row * ROW_S, (row + 1) * ROW_S
        k = (times >= lo - 0.005) & (times <= hi + 0.005)
        ax.pcolormesh(times[k], freqs[band], db[band][:, k], vmin=-45, vmax=0, shading="auto", cmap="viridis")
        ax.set_xlim(lo, hi)
        ax.set_ylim(FMIN, FMAX)
        ax.set_xticks(np.arange(lo, hi + 1e-9, 0.05))
        ax.set_xticks(np.arange(lo, hi + 1e-9, 0.01), minor=True)
        ax.grid(which="major", axis="x", color="white", alpha=0.35, lw=0.6)
        ax.grid(which="minor", axis="x", color="white", alpha=0.12, lw=0.4)
        for label, f in REFERENCE_KHZ.items():
            ax.axhline(f, color="white", ls=":", lw=0.6, alpha=0.45)
            ax.text(hi + 0.002, f, label, fontsize=7, va="center", color="black")
        ax.set_ylabel("kHz")
        if pred is not None:
            rows_in = pred[(pred.segment_file == window["file"]) & (pred.gate_status == "bat")
                           & (pred.end_s > t0 + lo) & (pred.start_s < t0 + hi)]
            for _, r in rows_in.iterrows():
                f = r.cf_plateau_khz if np.isfinite(r.cf_plateau_khz) else r.peak_freq_khz
                c = COLORS.get(r.predicted_label, "#aaaaaa")
                ax.add_patch(patches.Rectangle((r.start_s - t0, f - 2.5), r.end_s - r.start_s, 5,
                                               fill=False, ec=c, lw=1.4))
                ax.text(r.start_s - t0, f + 3, r.predicted_label.split("_")[0], color=c, fontsize=7)
    axes[-1].set_xlabel("相对窗口起点的时间 (s)")
    title = f"{window['id']}  {window['file']}  {t0:.2f}–{t0 + WINDOW_S:.2f} s  （{window['stratum']}）"
    if pred is not None:
        title += "  ——程序结果对照，框颜色=类别"
    fig.suptitle(title, fontsize=13)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.tight_layout(rect=(0, 0, 0.985, 0.98))
    plt.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    run_dir, manual_csv = Path(sys.argv[1]), Path(sys.argv[2])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pred = pd.read_csv(run_dir / "pulse_predictions.csv", low_memory=False)
    manual = pd.read_csv(manual_csv, encoding="utf-8-sig", index_col=0)
    windows = select_windows(pred, manual)
    for sub in ("空白声谱图", "程序结果对照图", "音频片段"):
        (KIT / sub).mkdir(parents=True, exist_ok=True)
    for window in windows:
        stem = f"{window['id']}_{window['file'][:-4]}_{window['start']:.2f}s"
        draw(window, None, KIT / "空白声谱图" / f"{stem}.png")
        draw(window, pred, KIT / "程序结果对照图" / f"{stem}_程序.png")
        y, sr = sf.read(str(RAW / window["file"]), start=int(window["start"] * 250000),
                        stop=int((window["start"] + WINDOW_S) * 250000))
        sf.write(KIT / "音频片段" / f"{stem}.wav", y, sr, subtype="PCM_16")
        print(window["id"], window["file"], window["start"], window["stratum"], "program bat", window["program_bat"], flush=True)
    with open(KIT / "窗口清单.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["窗口", "录音文件", "起点_s", "终点_s", "抽样依据", "人工表该类有名字数(整个录音)", "程序bat数(本窗口)"])
        for w in windows:
            writer.writerow([w["id"], w["file"], f"{w['start']:.2f}", f"{w['start'] + WINDOW_S:.2f}", w["stratum"],
                             w["manual_named"], w["program_bat"]])
    with open(KIT / "标注表.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["窗口", "开始_s", "结束_s", "频率_kHz", "类别", "清晰度", "备注"])
        writer.writerow(["W01", "0.123", "0.168", "84.5", "CF5_83K", "清楚", "示例行，填写前删除"])
    print("program run:", run_dir.name)


if __name__ == "__main__":
    main()
