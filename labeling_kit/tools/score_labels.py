"""把人工标注表和程序结果逐条配对，统计漏检、多检、分错类和碎片。

用法（在项目根目录）：
    .venv/Scripts/python.exe labeling_kit/tools/score_labels.py <程序结果目录> [标注表.csv]

配对规则：时间有重叠，且频率相差不超过 FREQ_TOL_KHZ；按时间重叠比例从高到低一对一配对。
程序叫声按“中心时间落在窗口内”计入该窗口；人工标注按表中填写的窗口计入。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path(__file__).resolve().parents[1]
FREQ_TOL_KHZ = 3.0


def load_inputs(run_dir: Path, labels_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = pd.read_csv(KIT / "窗口清单.csv", encoding="utf-8-sig")
    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    labels = labels[~labels["备注"].fillna("").str.contains("示例行")].copy()
    labels = labels.merge(windows[["窗口", "录音文件", "起点_s"]], on="窗口", how="left")
    if labels["录音文件"].isna().any():
        bad = sorted(labels.loc[labels["录音文件"].isna(), "窗口"].astype(str).unique())
        raise SystemExit(f"标注表里有不在窗口清单中的窗口：{bad}")
    labels["abs_start"] = labels["起点_s"] + labels["开始_s"].astype(float)
    labels["abs_end"] = labels["起点_s"] + labels["结束_s"].astype(float)
    labels["类别"] = labels["类别"].fillna("?").astype(str).str.strip()
    labels["清晰度"] = labels["清晰度"].fillna("清楚").astype(str).str.strip()
    pred = pd.read_csv(run_dir / "pulse_predictions.csv", low_memory=False)
    pred = pred[pred.gate_status.isin(["bat", "uncertain"])].copy()
    pred["freq"] = pred.cf_plateau_khz.where(np.isfinite(pred.cf_plateau_khz), pred.peak_freq_khz)
    rows = []
    for _, w in windows.iterrows():
        center = (pred.start_s + pred.end_s) / 2
        k = (pred.segment_file == w["录音文件"]) & (center >= w["起点_s"]) & (center < w["终点_s"])
        part = pred[k].copy()
        part["窗口"] = w["窗口"]
        rows.append(part)
    program = pd.concat(rows, ignore_index=True) if rows else pred.iloc[:0]
    return windows, labels, program


def match(labels: pd.DataFrame, program: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    program = program.copy()
    labels["配对程序行"] = -1
    program["配对标注行"] = -1
    program["重叠到已配对标注"] = False
    for window in labels["窗口"].unique():
        li = labels.index[labels["窗口"] == window]
        pi = program.index[program["窗口"] == window]
        pairs = []
        for i in li:
            for j in pi:
                overlap = min(labels.at[i, "abs_end"], program.at[j, "end_s"]) - max(labels.at[i, "abs_start"], program.at[j, "start_s"])
                if overlap <= 0 or abs(float(labels.at[i, "频率_kHz"]) - program.at[j, "freq"]) > FREQ_TOL_KHZ:
                    continue
                union = max(labels.at[i, "abs_end"], program.at[j, "end_s"]) - min(labels.at[i, "abs_start"], program.at[j, "start_s"])
                # bat 优先于 uncertain 配对，再按重叠比例。
                pairs.append((program.at[j, "gate_status"] == "bat", overlap / union, i, j))
        for _, _, i, j in sorted(pairs, reverse=True):
            if labels.at[i, "配对程序行"] == -1 and program.at[j, "配对标注行"] == -1:
                labels.at[i, "配对程序行"] = j
                program.at[j, "配对标注行"] = i
        for _, _, i, j in pairs:
            if program.at[j, "配对标注行"] == -1 and labels.at[i, "配对程序行"] != -1:
                program.at[j, "重叠到已配对标注"] = True
    return labels, program


def report(labels: pd.DataFrame, program: pd.DataFrame) -> str:
    lines = []
    matched = labels["配对程序行"] >= 0
    matched_status = labels["配对程序行"].map(lambda j: program.at[j, "gate_status"] if j >= 0 else "")
    for scope, mask in (("只算清楚", labels["清晰度"] == "清楚"), ("清楚+勉强", labels.index == labels.index)):
        sub = labels[mask]
        hit_bat = (matched_status[mask] == "bat").sum()
        hit_unc = (matched_status[mask] == "uncertain").sum()
        lines.append(f"[{scope}] 人工标注 {len(sub)} 条：程序判 bat 找到 {hit_bat}（召回 {hit_bat / max(len(sub), 1):.1%}），"
                     f"只在 uncertain 里 {hit_unc}，完全漏检 {len(sub) - hit_bat - hit_unc}")
    bat = program[program.gate_status == "bat"]
    tp = (bat["配对标注行"] >= 0).sum()
    split = (bat["重叠到已配对标注"] & (bat["配对标注行"] < 0)).sum()
    other = len(bat) - tp - split
    lines.append(f"程序 bat {len(bat)} 条：配上人工标注 {tp}（精确率 {tp / max(len(bat), 1):.1%}），"
                 f"同一条人工叫声上多出来的碎片 {split}，人工没标的 {other}")
    pairs = labels[matched & (matched_status == "bat")]
    if len(pairs):
        predicted = pairs["配对程序行"].map(lambda j: program.at[j, "predicted_label"])
        known = pairs["类别"] != "?"
        accuracy = (predicted[known] == pairs.loc[known, "类别"]).mean() if known.any() else float("nan")
        lines.append(f"类别（配上的 bat 中、人工类别已知的 {int(known.sum())} 条）：一致 {accuracy:.1%}")
        lines.append("混淆表（行=人工，列=程序）：")
        lines.append(pd.crosstab(pairs["类别"], predicted).to_string())
    lines.append("按人工类别的召回（清楚+勉强，程序判 bat）：")
    per = labels.assign(hit=(matched_status == "bat")).groupby("类别")["hit"].agg(["size", "sum"])
    per["召回"] = (per["sum"] / per["size"]).map(lambda v: f"{v:.0%}")
    lines.append(per.rename(columns={"size": "人工条数", "sum": "程序找到"}).to_string())
    lines.append("按窗口：")
    per_w = labels.groupby("窗口").size().rename("人工").to_frame()
    per_w["程序bat"] = bat.groupby("窗口").size()
    per_w["配上"] = labels[matched_status == "bat"].groupby("窗口").size()
    lines.append(per_w.fillna(0).astype(int).to_string())
    return "\n".join(lines)


def main() -> None:
    run_dir = Path(sys.argv[1])
    labels_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else KIT / "标注表.csv"
    _, labels, program = load_inputs(run_dir, labels_csv)
    labels, program = match(labels, program)
    text = report(labels, program)
    out = KIT / "评估结果"
    out.mkdir(exist_ok=True)
    (out / f"评估_{run_dir.name}.txt").write_text(text, encoding="utf-8")
    labels.to_csv(out / f"人工标注配对_{run_dir.name}.csv", index=False, encoding="utf-8-sig")
    program.to_csv(out / f"程序结果配对_{run_dir.name}.csv", index=False, encoding="utf-8-sig")
    print(text)


if __name__ == "__main__":
    main()
