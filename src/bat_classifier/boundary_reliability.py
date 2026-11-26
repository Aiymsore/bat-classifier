"""Raw-mixture boundary evidence, kept separate from family and duration rules."""
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

BOUNDARY_COLUMNS=('boundary_status','boundary_review_reason','boundary_snr_db',
    'boundary_support_fraction','sequence_reference_eligible','segmentation_review',
    'segmentation_review_reason')


def ridge_evidence(audio,sr,offset_s,frequency_hz):
    """Observe a narrow ridge, using fixed temporal and adjacent-frequency noise."""
    if len(audio)<512:return None
    f,t,z=stft(audio,sr,nperseg=512,noverlap=480,boundary=None)
    t=t+offset_s;power=abs(z)**2
    ids=np.flatnonzero(abs(f-frequency_hz)<=1500.)
    if not len(ids):return None
    bins=ids[np.argmax(power[ids],axis=0)]
    peaks=power[bins,np.arange(len(t))]
    # For stationary Gaussian noise, STFT-bin power is approximately exponential.
    # P10 is only 0.105 times mean noise power. Using it as the floor directly
    # gives random spectral maxima an artificial ~9.8dB SNR advantage.
    noise=(np.percentile(power,10,axis=1)/(-np.log(.9)))[bins]
    snr=10*np.log10(np.maximum(peaks,1e-30)/np.maximum(noise,1e-30))
    distances=abs(f[:,None]-f[bins][None,:])
    neighbor=(distances>=1500.)&(distances<=6000.)
    floor=np.array([np.median(power[neighbor[:,i],i]) for i in range(len(t))])
    contrast=10*np.log10(np.maximum(peaks,1e-30)/np.maximum(floor,1e-30))
    local=(peaks>=power[np.maximum(0,bins-1),np.arange(len(t))]) & (peaks>=power[np.minimum(len(f)-1,bins+1),np.arange(len(t))])
    return t,local&(snr>=6.)&(contrast>=6.),snr


def gap_decision(audio,sr,offset_s,end_s,start_s,frequency_hz):
    """A gap is joined only when its interior contains an observed bridge."""
    evidence=ridge_evidence(audio,sr,offset_s,frequency_hz)
    if evidence is None:return 'uncertain'
    t,support,_=evidence
    # Exclude half a Fourier window at each edge: window smearing is not a bridge.
    mask=(t>end_s+256/sr)&(t<start_s-256/sr)
    if np.count_nonzero(mask)<3:return 'uncertain'
    values=support[mask];fraction=float(values.mean())
    longest=run=0
    for value in values:
        run=0 if value else run+1;longest=max(longest,run)
    if fraction>=.9 and longest*32/sr<=.0005:return 'continuous'
    # A max over several frequency bins admits isolated random peaks in noise.
    # Sparse support is not a continuous ridge; the middle range stays unresolved.
    if fraction<=.3:return 'separate'
    return 'uncertain'


def assess_boundaries(rows,wav_path):
    """Attach evidence; never edit measured times or fill unobserved signal."""
    out=rows.copy()
    for key in ('boundary_status','boundary_review_reason','segmentation_review_reason'):out[key]=''
    out['boundary_snr_db']=np.nan;out['boundary_support_fraction']=np.nan
    out['sequence_reference_eligible']=False;out['segmentation_review']=False
    with sf.SoundFile(str(wav_path)) as h:
        sr=h.samplerate;duration=len(h)/sr
        def read(a,b):
            a=max(0.,a);b=min(duration,b);h.seek(int(a*sr))
            return h.read(max(0,int((b-a)*sr)),always_2d=True).mean(axis=1),a
        for idx,r in out.iterrows():
            a,b=float(r.start_s),float(r.end_s)
            freq=float(r.get('cf_plateau_khz',np.nan))
            if not np.isfinite(freq):freq=float(r.get('peak_freq_khz',np.nan))
            if not np.isfinite(freq):
                out.at[idx,'boundary_status']='uncertain';out.at[idx,'boundary_review_reason']='missing_frequency';continue
            y,offset=read(a-.015,b+.015)
            evidence=ridge_evidence(y,sr,offset,freq*1000)
            if evidence is None:
                out.at[idx,'boundary_status']='uncertain';out.at[idx,'boundary_review_reason']='insufficient_audio';continue
            t,support,snr=evidence
            inside=(t>=a+.001)&(t<=b-.001)
            left=(t>=a-.006)&(t<a-.0015);right=(t>b+.0015)&(t<=b+.006)
            if not inside.any():
                out.at[idx,'boundary_status']='uncertain';out.at[idx,'boundary_review_reason']='too_short_to_resolve';continue
            out.at[idx,'boundary_snr_db']=float(np.median(snr[inside]))
            out.at[idx,'boundary_support_fraction']=float(support[inside].mean())
            def extends(mask):
                run=0
                for value in support[mask]:
                    run=run+1 if value else 0
                    if run*32/sr>=.00075:return True
                return False
            if a<.006 or b>duration-.006:
                status,reason='truncated','recording_edge'
            elif extends(left) or extends(right):
                status,reason='truncated','ridge_outside_component'
            elif left.any() and right.any() and support[left].mean()<=.2 and support[right].mean()<=.2 and support[inside].mean()>=.7:
                status,reason='complete','quiet_on_both_sides'
            else:status,reason='uncertain','weak_or_ambiguous_edges'
            out.at[idx,'boundary_status']=status;out.at[idx,'boundary_review_reason']=reason
        # Same-frequency neighboring fragments are evaluated independently of 17ms.
        candidates=out[out.gate_status.ne('noise') & out.cf_plateau_khz.notna()].sort_values('start_s')
        entries=list(candidates.iterrows())
        for n,(i,a) in enumerate(entries):
            for j,b in entries[n+1:]:
                gap=float(b.start_s-a.end_s)
                if b.start_s-a.end_s>.05:break
                if abs(float(a.cf_plateau_khz-b.cf_plateau_khz))>1.:continue
                if gap<=0:
                    for idx in (i,j):
                        out.at[idx,'boundary_status']='uncertain'
                        out.at[idx,'segmentation_review']=True
                        out.at[idx,'segmentation_review_reason']='same_frequency_overlap'
                    continue
                y,offset=read(float(a.end_s)-.100,float(b.start_s)+.100)
                decision=gap_decision(y,sr,offset,float(a.end_s),float(b.start_s),float((a.cf_plateau_khz+b.cf_plateau_khz)*500))
                if decision!='separate':
                    for idx in (i,j):
                        out.at[idx,'boundary_status']='fragment_suspected'
                        out.at[idx,'segmentation_review']=True
                        out.at[idx,'segmentation_review_reason']='continuous_bridge_unmerged' if decision=='continuous' else 'gap_unresolved'
        out['sequence_reference_eligible']=(out.gate_status.eq('bat') & out.boundary_status.eq('complete')
            & out.boundary_snr_db.ge(10.) & ~out.segmentation_review
            & pd.to_numeric(out.get('gate_quality',pd.Series(0.,index=out.index)),errors='coerce').ge(.65)
            & pd.to_numeric(out.get('gate_coverage',pd.Series(0.,index=out.index)),errors='coerce').ge(.8))
    return out
