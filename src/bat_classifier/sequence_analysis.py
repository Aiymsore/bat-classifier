"""默认连串判族：仅使用可靠完整成员的时长，保留实测值供追溯。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

SEQUENCE_COLUMNS = ('single_predicted_label', 'single_confidence', 'sequence_id',
    'sequence_size', 'sequence_plateau_median_ms', 'sequence_family',
    'sequence_reference_count', 'sequence_reference_ids', 'sequence_observed_median_ms',
    'sequence_review_reason', 'sequence_predicted_label', 'sequence_changed', 'family_source',
    'single_gate_status', 'single_is_uncertain', 'single_gate_reason', 'single_gate_reason_code',
    'single_gate_pass_path', 'single_gate_promoted',
    'cf_duration_source', 'cf_duration_accepted', 'cf_duration_review')


@dataclass(frozen=True)
class SequenceConfig:
    max_start_gap_ms: float = 200.0
    frequency_tolerance_khz: float = 1.0
    min_calls: int = 3
    min_reference_calls: int = 2
    plateau_threshold_ms: float = 17.0
    tail_ratio_threshold: float = 0.10
    short_requires_tail: bool = False  # 兼容旧参数；尾段不再参与判定
    min_similarity: float = 0.60
    mixed_min_fraction: float = 0.30
    mixed_min_center_gap_ms: float = 10.0
    mixed_max_group_iqr_ms: float = 6.0

    def validate(self):
        for k, v in asdict(self).items():
            if k == 'short_requires_tail':
                continue
            if not np.isfinite(v) or (v < 0 if k in ('plateau_threshold_ms', 'min_similarity') else v <= 0):
                raise ValueError(f"序列参数必须为正且有限：{k}")
        if self.min_calls < 3 or int(self.min_calls) != self.min_calls:
            raise ValueError("min_calls 必须是至少 3 的整数")
        if self.min_reference_calls < 2 or int(self.min_reference_calls) != self.min_reference_calls:
            raise ValueError("min_reference_calls 必须是至少 2 的整数")
        if self.mixed_min_fraction > .5 or self.min_similarity > 1 or self.tail_ratio_threshold > 1:
            raise ValueError("比例或相似度参数越界")
        return self


def _mixed(values: np.ndarray, config: SequenceConfig) -> bool:
    """保守的双群检查：两组都占至少30%，中心差至少10ms，且各自IQR不超过6ms。

    这是复核提示，不等价于证明有两只蝙蝠。不自动拆串。
    """
    v = np.sort(values)
    if len(v) < 6:
        return False
    # 一维两组最小平方误差切分，避免随机初始化。
    best = None
    for cut in range(1, len(v)):
        a, b = v[:cut], v[cut:]
        if min(len(a), len(b)) < max(3, config.mixed_min_fraction * len(v)):
            continue
        loss = float(np.sum((a-a.mean())**2) + np.sum((b-b.mean())**2))
        if best is None or loss < best[0]:
            best = (loss, a, b)
    if best is None:
        return False
    _, a, b = best
    return bool(np.median(a) < config.plateau_threshold_ms <= np.median(b)
                and np.median(b)-np.median(a) >= config.mixed_min_center_gap_ms
                and max(np.ptp(np.percentile(a, [25, 75])),
                        np.ptp(np.percentile(b, [25, 75]))) <= config.mixed_max_group_iqr_ms)


def analyze_sequences(rows: pd.DataFrame, config: SequenceConfig | None = None):
    """返回 (带新增序列列的逐条表, 序列表)。仅改序列建议列。

    bat、有效平台参与连串，按录音和CF候选类别分开；只有CF、CF-FM两族
    都达标的行可获改类建议。参考成员必须有明确的边界可靠性证据；缺失不默认可靠。
    sequence_plateau_median_ms 是参考成员中位数，observed 列保留所有成员中位数。
    同频时间重叠的行不强行分配给同一串；连接歧义记复核，原判定保留。
    """
    c = (config or SequenceConfig()).validate()
    required = ['segment_file', 'start_s', 'end_s', 'gate_status', 'predicted_label',
                'cf_plateau_khz', 'cf_plateau_ms',
                'template_cf_label', 'template_cffm_label', 'template_cffm_score']
    missing = set(required)-set(rows.columns)
    if missing:
        raise ValueError(f"输入缺列：{sorted(missing)}")
    out = rows.copy().reset_index(drop=True)
    for col in ['sequence_id', 'sequence_family', 'sequence_review_reason']:
        out[col] = ''
    for col in ['sequence_size', 'sequence_plateau_median_ms', 'sequence_tail_fraction']:
        out[col] = np.nan
    out['sequence_predicted_label'] = out.predicted_label
    out['single_predicted_label'] = out.predicted_label
    out['family_source'] = 'single'
    out['sequence_reference_count'] = 0
    out['sequence_reference_ids'] = ''
    out['sequence_observed_median_ms'] = np.nan
    cf = out.template_cf_label.fillna('').astype(str)
    valid = (out.gate_status.eq('bat') & cf.str.startswith('CF') & ~cf.str.startswith('CFFM')
             & np.isfinite(out.cf_plateau_khz) & np.isfinite(out.cf_plateau_ms)
             & out.cf_plateau_ms.gt(0) & np.isfinite(out.start_s) & np.isfinite(out.end_s)
             & out.end_s.gt(out.start_s))
    summaries = []
    for (file, label), group in out[valid].groupby(['segment_file', 'template_cf_label'], sort=True):
        tracks = []
        for idx, r in group.sort_values(['start_s', 'end_s'], kind='stable').iterrows():
            possible = []
            overlap_seen = False
            for t in tracks:
                gap = (r.start_s - t['last_start']) * 1000
                if gap > c.max_start_gap_ms:
                    continue
                if abs(r.cf_plateau_khz - np.median(t['freqs'])) > c.frequency_tolerance_khz:
                    continue
                if r.start_s < t['last_end']:
                    t['review'].add('same_frequency_overlap')
                    overlap_seen = True
                    continue
                possible.append(t)
            if len(possible) == 1:
                t = possible[0]
            else:
                t = dict(ids=[], freqs=[], review=set())
                tracks.append(t)
                if len(possible) > 1:
                    t['review'].add('ambiguous_link')
                    for other in possible:
                        other['review'].add('ambiguous_link')
            if overlap_seen:
                t['review'].add('same_frequency_overlap')
            t['ids'].append(idx)
            t['freqs'].append(float(r.cf_plateau_khz))
            t.update(last_start=float(r.start_s), last_end=float(r.end_s))
        for number, t in enumerate(tracks, 1):
            ids = t['ids']; g = out.loc[ids]
            seqid = f'{file}:{label}:s{number:04d}'
            eligible = g.get('sequence_reference_eligible', pd.Series(False, index=g.index)).astype(str).str.lower().eq('true')
            refs = g[eligible]
            med = float(refs.cf_plateau_ms.median()) if len(refs) else np.nan
            observed_med = float(g.cf_plateau_ms.median())
            reference_ids = '|'.join(str(refs.at[i, 'pulse_id']) if 'pulse_id' in refs else str(i) for i in refs.index)
            tail_fraction = np.nan  # 兼容旧导出列，不参与判定
            both = (g.template_cffm_label.fillna('').str.startswith('CFFM')
                    & g.template_cffm_score.ge(c.min_similarity))
            family = ''; source = 'single'
            # CF-only rows also need a trustworthy sequence to support a short
            # member. They cannot obtain that support from size/median alone
            # when the track is mixed or ambiguous.
            if len(ids) >= c.min_calls:
                if len(refs) < c.min_reference_calls:
                    t['review'].add('insufficient_reliable_references')
                if len(refs) and refs.cf_plateau_ms.lt(c.plateau_threshold_ms).any() and refs.cf_plateau_ms.ge(c.plateau_threshold_ms).any():
                    t['review'].add('reference_family_conflict')
                if _mixed(refs.cf_plateau_ms.to_numpy(), c):
                    t['review'].add('mixed_duration')
                if t['review']:
                    source = 'sequence_review'
                else:
                    family = 'CF' if med >= c.plateau_threshold_ms else 'CF-FM'
                    source = 'sequence'
            out.loc[ids, 'sequence_id'] = seqid
            out.loc[ids, 'sequence_size'] = len(ids)
            out.loc[ids, 'sequence_plateau_median_ms'] = med
            out.loc[ids, 'sequence_observed_median_ms'] = observed_med
            out.loc[ids, 'sequence_reference_count'] = len(refs)
            out.loc[ids, 'sequence_reference_ids'] = reference_ids
            out.loc[ids, 'sequence_tail_fraction'] = tail_fraction
            out.loc[ids, 'sequence_family'] = family
            out.loc[ids, 'sequence_review_reason'] = '|'.join(sorted(t['review']))
            candidates = g.index[both]
            out.loc[candidates, 'family_source'] = source
            if family:
                chosen = 'template_cf_label' if family == 'CF' else 'template_cffm_label'
                out.loc[candidates, 'sequence_predicted_label'] = out.loc[candidates, chosen]
            summaries.append(dict(sequence_id=seqid, segment_file=file, cf_label=label,
                start_s=float(g.start_s.min()), end_s=float(g.end_s.max()), sequence_size=len(ids),
                sequence_plateau_median_ms=med, sequence_tail_fraction=tail_fraction,
                sequence_reference_count=len(refs), sequence_reference_ids=reference_ids,
                sequence_observed_median_ms=observed_med,
                sequence_family=family, family_source=source,
                review_reason='|'.join(sorted(t['review'])), eligible_calls=int(both.sum()),
                changed_calls=int(out.loc[ids, 'sequence_predicted_label'].ne(g.predicted_label).sum())))
    out['sequence_changed'] = out.sequence_predicted_label.ne(out.single_predicted_label)
    assert len(out) == len(rows)
    return out.set_axis(rows.index), pd.DataFrame(summaries)


def classify_sequence_rows(rows: pd.DataFrame, config: SequenceConfig | None = None):
    """Apply duration-only sequence suggestions to final labels, idempotently.

    Every final CF must pass the single-platform threshold or have a clean,
    sufficiently long sequence. Short unsupported CF candidates become review
    rows; they are never automatically reassigned to CF-FM.
    """
    base = rows.copy()
    c = (config or SequenceConfig()).validate()
    if base.empty:
        return base, pd.DataFrame()
    if 'single_predicted_label' in base:
        base['predicted_label'] = base.single_predicted_label
    if 'single_confidence' in base:
        base['confidence'] = base.single_confidence
    preserved = {'gate_status': 'bat', 'is_uncertain': False, 'gate_reason': '',
        'gate_reason_code': '', 'gate_pass_path': '', 'gate_promoted': False}
    for key, default in preserved.items():
        saved = 'single_' + key
        if saved in base:
            base[key] = base[saved]
        elif key not in base:
            base[key] = default
        base[saved] = base[key]
    single_confidence = base.get('confidence', pd.Series(np.nan, index=base.index)).copy()
    result, summary = analyze_sequences(base, c)
    result['single_confidence'] = single_confidence
    result['predicted_label'] = result.sequence_predicted_label
    changed = result.sequence_changed
    cf = changed & result.sequence_family.eq('CF')
    cffm = changed & result.sequence_family.eq('CF-FM')
    result.loc[cf, 'confidence'] = 1.0  # CF template interval passed
    result.loc[cffm, 'confidence'] = result.loc[cffm, 'template_cffm_score']
    final_cf = (result.gate_status.eq('bat') & result.predicted_label.fillna('').str.startswith('CF')
                & ~result.predicted_label.fillna('').str.startswith('CFFM'))
    single_ok = np.isfinite(result.cf_plateau_ms) & result.cf_plateau_ms.ge(c.plateau_threshold_ms)
    sequence_ok = (result.sequence_family.eq('CF') & result.sequence_size.ge(c.min_calls)
        & result.sequence_reference_count.ge(c.min_reference_calls)
        & result.sequence_plateau_median_ms.ge(c.plateau_threshold_ms)
        & result.sequence_review_reason.fillna('').eq('')
        & result.template_cf_label.eq(result.predicted_label))
    blocked = final_cf & ~single_ok & ~sequence_ok
    result['cf_duration_source'] = 'not_cf'
    result.loc[final_cf & single_ok, 'cf_duration_source'] = 'single'
    result.loc[final_cf & ~single_ok & sequence_ok, 'cf_duration_source'] = 'sequence'
    result.loc[blocked, 'cf_duration_source'] = 'review'
    result['cf_duration_accepted'] = final_cf & ~blocked
    result['cf_duration_review'] = blocked
    result.loc[final_cf & ~single_ok & sequence_ok, 'family_source'] = 'sequence'
    result.loc[blocked, 'family_source'] = 'duration_review'
    result.loc[blocked, 'predicted_label'] = 'uncertain'
    result.loc[blocked, 'sequence_predicted_label'] = 'uncertain'
    result.loc[blocked, 'gate_status'] = 'uncertain'
    result.loc[blocked, 'is_uncertain'] = True
    result.loc[blocked, 'gate_promoted'] = False
    result.loc[blocked, 'gate_pass_path'] = ''
    result.loc[blocked, 'gate_reason_code'] = 'cf_duration_insufficient'
    for idx in result.index[blocked]:
        result.at[idx, 'gate_reason'] = (f'CF平台 {result.at[idx, "cf_plateau_ms"]:.3f} ms '
            f'未达到 {c.plateau_threshold_ms:g} ms，且无合格长平台连串支持；保留待定')
    unresolved = result.get('segmentation_review', pd.Series(False, index=result.index)).astype(str).str.lower().eq('true') & result.gate_status.eq('bat')
    result.loc[unresolved, ['predicted_label', 'sequence_predicted_label', 'gate_status']] = 'uncertain'
    result.loc[unresolved, 'is_uncertain'] = True
    result.loc[unresolved, 'gate_promoted'] = False
    result.loc[unresolved, 'gate_pass_path'] = ''
    result.loc[unresolved, 'gate_reason_code'] = 'segmentation_unresolved'
    result.loc[unresolved, 'gate_reason'] = '同频相邻片段是否属于同一次叫声尚未确定；暂不计入确认叫声数'
    result.loc[unresolved, 'cf_duration_accepted'] = False
    result.loc[unresolved, 'family_source'] = 'segmentation_review'
    result['sequence_changed'] = result.predicted_label.ne(result.single_predicted_label)
    for idx, row in summary.iterrows():
        members = result.sequence_id.eq(row.sequence_id)
        summary.at[idx, 'changed_calls'] = int(result.loc[members, 'sequence_changed'].sum())
        summary.at[idx, 'duration_review_calls'] = int(result.loc[members, 'cf_duration_review'].sum())
    return result, summary
