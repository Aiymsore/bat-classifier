"""离线序列判族与逐WAV/标注窗口对照。只生成新目录，不覆盖输入。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bat_classifier.sequence_analysis import SequenceConfig, analyze_sequences


def table(frame):
    # 不额外依赖 tabulate。
    vals = frame.fillna('').astype(str)
    return '\n'.join(['| ' + ' | '.join(vals.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(vals.columns)) + ' |'] +
                     ['| ' + ' | '.join(r) + ' |' for r in vals.to_numpy()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('output_dir', type=Path)
    ap.add_argument('--duration-only', action='store_true', help='敏感性对照：短平台不要求多数有尾巴')
    args = ap.parse_args()
    if args.output_dir.exists():
        raise SystemExit('输出目录已存在，请指定新目录以保留旧实验')
    args.output_dir.mkdir(parents=True)
    path = args.run_dir / 'pulse_predictions.csv'
    original = pd.read_csv(path, low_memory=False)
    meta = json.loads((args.run_dir / 'run_metadata.json').read_text(encoding='utf-8-sig'))
    config = SequenceConfig(plateau_threshold_ms=float(meta['config']['cf_min_plateau_ms']),
                            min_similarity=float(meta['config']['min_confidence']),
                            short_requires_tail=not args.duration_only)
    result, sequences = analyze_sequences(original, config)
    out = args.output_dir
    result.to_csv(out / 'pulse_sequence_predictions.csv', index=False, encoding='utf-8-sig')
    sequences.to_csv(out / 'sequences.csv', index=False, encoding='utf-8-sig')
    b = result[result.gate_status.eq('bat')]
    classes = sorted(set(b.single_predicted_label) | set(b.sequence_predicted_label))
    perwav = []
    manual_path = ROOT / '.tmp' / 'manual_20250807.csv'
    manual = pd.read_csv(manual_path, index_col=0, encoding='utf-8-sig') if manual_path.exists() else pd.DataFrame()
    for file, g in b.groupby('segment_file'):
        for label in classes:
            perwav.append(dict(segment_file=file, label=label,
                single_count=int(g.single_predicted_label.eq(label).sum()),
                sequence_count=int(g.sequence_predicted_label.eq(label).sum()),
                reference_count=(manual.at[file, label] if file in manual.index and label in manual else np.nan)))
    pw = pd.DataFrame(perwav)
    pw.to_csv(out/'per_wav_comparison.csv', index=False, encoding='utf-8-sig')
    totals = pw.groupby('label')[['single_count','sequence_count','reference_count']].sum(min_count=1).reset_index()
    windows = pd.read_csv(ROOT/'labeling_kit'/'窗口清单.csv', encoding='utf-8-sig')
    wc = []
    for _, w in windows.iterrows():
        center = (b.start_s+b.end_s)/2
        g = b[b.segment_file.eq(w['录音文件']) & center.ge(w['起点_s']) & center.lt(w['终点_s'])]
        for label in classes:
            a = int(g.single_predicted_label.eq(label).sum()); z = int(g.sequence_predicted_label.eq(label).sum())
            if a or z:
                wc.append(dict(window=w['窗口'], label=label, before=a, after=z))
    windows_out = pd.DataFrame(wc)
    windows_out.to_csv(out/'window_comparison.csv', index=False, encoding='utf-8-sig')
    (out/'parameters.json').write_text(json.dumps(dict(source_run=str(args.run_dir.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), parameters=asdict(config),
        note='Offline experiment. Detection, gate, source classifications and counts unchanged.'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    focus = windows_out[windows_out.window.isin(['W13','W14','W15','W16'])]
    review = sequences[sequences.eligible_calls.gt(0)].family_source.value_counts().to_dict()
    text = ('# 序列判族离线试验\n\n'
        f'输入：{args.run_dir.name}。分量总数 {len(result)}，bat 总数 {len(b)} 均保持不变。'
        f'改类建议 {int(result.sequence_changed.sum())} 条。含双族达标行的序列状态：{review}。\n\n'
        '## 重点窗口\n\n'+table(focus)+'\n\n'
        '人工窗口结论：W13/W14 的66–67kHz叫声按CF复核；W15/W16人工CF-FM约25/26条。'
        '该试验不补检，不能找回原来未进入bat的叫声。\n\n'
        '## 全日类别数（仅作参考）\n\n'+table(totals)+'\n\n'
        '## 规则与边界\n\n'
        '同录音、同CF候选类别连串，频率距串内中位数≤1kHz，前后起点间隔≤200ms。'
        '至少3条；中位平台≥阈值则CF；'+
        ('短平台且尾巴占比>50%才CF-FM，否则CF。' if config.short_requires_tail else '短平台判CF-FM（尾巴只记录）。')+
        '只对CF区间命中且CF-FM相似度达标的行建议改类；同频重叠、连接歧义、明显双时长群保留单条结果。'
        '尾巴仅为序列辅助证据，此规则尚未验证为生物学真值。同一串不保证同一个体。'
        '原始逐条结论在single_predicted_label，建议在sequence_predicted_label。\n\n'
        '完整逐WAV对照见per_wav_comparison.csv；每串理由见sequences.csv；参数和输入摘要见parameters.json。\n')
    (out/'对比报告.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
