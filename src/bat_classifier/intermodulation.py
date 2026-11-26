"""Offline third-order intermodulation audit; never changes production labels.

Only directly observed, stronger parent ridges can explain a weaker product.
Frequency, time, raw-spectrum prominence and a cubic-envelope prediction must
agree. Ambiguous or conflicting parent assignments are review-only.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft


@dataclass(frozen=True)
class IntermodulationConfig:
    n_fft: int = 2048
    hop: int = 64
    ridge_radius_hz: float = 1250.
    frequency_bins: float = 2.
    frequency_coverage: float = .90
    time_coverage: float = .90
    envelope_correlation: float = .92
    prominence_db: float = 10.
    parent_power_ratio: float = 1.15
    min_frames: int = 8
    minimum_support_windows: float = 1.
    context_s: float = .012


def evaluate_profiles(times, frequencies, powers, prominences, intervals, bin_hz,
                      config=None):
    """Profiles ordered as product, doubled parent, subtracted parent.

    Exposed for deterministic tests. Intervals and frequencies are in s and Hz.
    Every metric is observed evidence, not a calibrated probability.
    """
    c = config or IntermodulationConfig()
    t = np.asarray(times)
    active = (t >= intervals[0][0]) & (t <= intervals[0][1])
    common = active.copy()
    for a, b in intervals[1:]:
        common &= (t >= a) & (t <= b)
    if common.sum() < c.min_frames:
        return None
    time_coverage = float(common.sum()/max(active.sum(), 1))
    residual = frequencies[0] - (2*frequencies[1]-frequencies[2])
    freq_coverage = float(np.mean(abs(residual[common]) <= c.frequency_bins*bin_hz))
    prominence = [float(np.median(p[common])) for p in prominences]
    parent_ratio = [float(np.median(powers[i][common])/max(np.median(powers[0][common]),1e-30)) for i in (1,2)]
    support_s=float(t[common][-1]-t[common][0])
    logs = [10*np.log10(np.maximum(p,1e-30)) for p in powers]
    predicted = 2*logs[1]+logs[2]
    corr = (float(np.corrcoef(logs[0], predicted)[0,1])
            if min(np.std(logs[0]),np.std(predicted)) >= 2 else 0.)
    # Outside the joint parent intervals a strong product ridge is independent
    # evidence. A half-window is excluded at edges to account for STFT smearing.
    pad = .5/bin_hz
    outside = (t < max(x[0] for x in intervals[1:])-pad) | (t > min(x[1] for x in intervals[1:])+pad)
    independent = bool(np.any(outside & (prominences[0] >= c.prominence_db)
                              & (powers[0] >= np.median(powers[0][common])*.25)))
    checks = dict(time_coverage=time_coverage >= c.time_coverage,
        duration_support=support_s >= c.minimum_support_windows/bin_hz,
        frequency_coverage=freq_coverage >= c.frequency_coverage,
        raw_prominence=min(prominence) >= c.prominence_db,
        stronger_parents=min(parent_ratio) >= c.parent_power_ratio,
        envelope=np.isfinite(corr) and corr >= c.envelope_correlation,
        no_independent_activity=not independent)
    passed = all(checks.values())
    return dict(passed=bool(passed), time_coverage=time_coverage,
        frequency_coverage=freq_coverage, residual_median_hz=float(np.median(residual[common])),
        residual_p90_hz=float(np.percentile(abs(residual[common]),90)),
        envelope_correlation=corr, product_prominence_db=prominence[0],
        parent_a_prominence_db=prominence[1], parent_b_prominence_db=prominence[2],
        parent_a_power_ratio=parent_ratio[0], parent_b_power_ratio=parent_ratio[1],
        independent_activity=independent, common_frames=int(common.sum()),
        common_duration_ms=support_s*1000,
        failed_checks='|'.join(key for key,ok in checks.items() if not ok))


def relation_profiles(wav, product, parent_a, parent_b, config=None):
    """Read raw local audio and return aligned evidence for audit and plotting."""
    c = config or IntermodulationConfig()
    rows = [product,parent_a,parent_b]
    a = max(0.,float(product.start_s)-c.context_s)
    b = float(product.end_s)+c.context_s
    with sf.SoundFile(str(wav)) as h:
        sr=h.samplerate
        h.seek(int(a*sr)); y=h.read(int((b-a)*sr),always_2d=True).mean(axis=1)
    if len(y) < c.n_fft:
        return None
    f,t,z=stft(y,sr,nperseg=c.n_fft,noverlap=c.n_fft-c.hop,boundary=None)
    t=t+a; power=abs(z)**2
    freqs=[]; energies=[]; prominences=[]
    for row in rows:
        center=float(row.peak_freq_khz)*1000
        search=np.flatnonzero(abs(f-center)<=c.ridge_radius_hz)
        if not len(search):
            return None
        bins=search[np.argmax(power[search],axis=0)]
        ridge=f[bins]
        near=abs(f[:,None]-ridge[None,:])<=500.
        background=(abs(f[:,None]-ridge[None,:])>=2000.) & (abs(f[:,None]-ridge[None,:])<=4000.)
        floor=np.array([np.median(power[background[:,i],i]) for i in range(len(t))])
        peak=power[bins,np.arange(len(t))]
        freqs.append(ridge); energies.append(np.sum(np.where(near,power,0.),axis=0))
        prominences.append(10*np.log10(np.maximum(peak,1e-30)/np.maximum(floor,1e-30)))
    return (t,np.array(freqs),np.array(energies),np.array(prominences),
        [(float(r.start_s),float(r.end_s)) for r in rows],sr/c.n_fft)


def measure_relation(wav, product, parent_a, parent_b, config=None):
    profiles=relation_profiles(wav,product,parent_a,parent_b,config)
    return evaluate_profiles(*profiles,config=config) if profiles is not None else None


def resolve_relations(proposals):
    """Unique explanations only; no product can also be used as a parent."""
    good=[p for p in proposals if p['passed']]
    counts={p['product_id']:sum(q['product_id']==p['product_id'] for q in good) for p in good}
    product_ids=set(counts)
    for p in proposals:
        p['decision']=('review_short_support' if p.get('failed_checks')=='duration_support'
                       else 'insufficient_evidence')
        if p['passed']:
            p['decision']=('review_ambiguous' if counts[p['product_id']]>1 else
                'review_parent_conflict' if {p['parent_a_id'],p['parent_b_id']} & product_ids
                else 'exclude_intermodulation')
    return proposals


def audit_intermodulation(rows, wav_paths, config=None):
    """Return copied pulse rows plus evidence. All original columns preserved."""
    c=config or IntermodulationConfig()
    if rows.duplicated(['segment_file','pulse_id']).any():
        raise ValueError('Duplicate recording/pulse key')
    result=rows.copy()
    result['intermodulation_status']='unchanged'
    result['corrected_counted']=result.gate_status.eq('bat')
    evidence=[]
    for file,g in rows.groupby('segment_file'):
        candidates=g[g.gate_status.isin(['bat','uncertain']) & g.peak_freq_khz.notna()]
        for product in candidates.itertuples():
            sources=candidates[(candidates.pulse_id!=product.pulse_id)
                & (candidates.start_s<product.end_s) & (candidates.end_s>product.start_s)]
            for pa in sources.itertuples():
                for pb in sources.itertuples():
                    if pa.pulse_id==pb.pulse_id:
                        continue
                    predicted=2*pa.peak_freq_khz-pb.peak_freq_khz
                    if abs(predicted-product.peak_freq_khz)>2.5 or min(abs(pa.peak_freq_khz-product.peak_freq_khz),abs(pb.peak_freq_khz-product.peak_freq_khz))<3:
                        continue
                    overlap=min(product.end_s,pa.end_s,pb.end_s)-max(product.start_s,pa.start_s,pb.start_s)
                    if overlap/max(product.end_s-product.start_s,1e-12)<c.time_coverage:
                        continue
                    measured=measure_relation(Path(wav_paths[file]),product,pa,pb,c)
                    if measured is not None:
                        evidence.append(dict(segment_file=file,product_id=product.pulse_id,
                            parent_a_id=pa.pulse_id,parent_b_id=pb.pulse_id,
                            start_s=product.start_s,end_s=product.end_s,
                            product_khz=product.peak_freq_khz,parent_a_khz=pa.peak_freq_khz,
                            parent_b_khz=pb.peak_freq_khz,**measured))
    # Resolve each recording separately: pulse IDs are local to a recording.
    for file in rows.segment_file.unique():
        resolve_relations([p for p in evidence if p['segment_file']==file])
    for p in evidence:
        if p['decision']=='insufficient_evidence':
            continue
        mask=result.segment_file.eq(p['segment_file']) & result.pulse_id.eq(p['product_id'])
        result.loc[mask,'intermodulation_status']=p['decision']
        if p['decision']=='exclude_intermodulation':
            result.loc[mask,'corrected_counted']=False
    return result,pd.DataFrame(evidence)
