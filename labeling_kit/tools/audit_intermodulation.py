"""Run the offline audit on existing predictions; input files are read-only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bat_classifier.intermodulation import IntermodulationConfig,audit_intermodulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--wav-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(args.predictions)
    paths={}
    for file in original.segment_file.unique():
        matches=[p for p in args.wav_dir.rglob('*') if p.is_file() and p.name==file]
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one WAV for {file}; found {len(matches)}')
        paths[file]=matches[0]
    cfg=IntermodulationConfig()
    corrected,evidence=audit_intermodulation(original,paths,cfg)
    pd.testing.assert_frame_equal(corrected[original.columns],original)
    corrected.to_csv(args.output/'pulse_predictions_corrected.csv',index=False,encoding='utf-8-sig')
    evidence.to_csv(args.output/'intermodulation_evidence.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for (file,label),g in corrected[corrected.gate_status=='bat'].groupby(['segment_file','predicted_label']):
        summaries.append(dict(segment_file=file,label=label,original_count=len(g),
            excluded_count=int((~g.corrected_counted).sum()),corrected_count=int(g.corrected_counted.sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.output/'count_comparison.csv',index=False,encoding='utf-8-sig')
    metadata=dict(source=str(args.predictions.resolve()),source_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        wav_paths={k:str(v.resolve()) for k,v in paths.items()},parameters=asdict(cfg),
        note='Offline experiment. All original columns preserved; uncertain sources are not promoted. Thresholds are not independently calibrated.')
    (args.output/'parameters.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    print('Decisions:',evidence.decision.value_counts().to_dict() if len(evidence) else {})


if __name__=='__main__':
    main()
