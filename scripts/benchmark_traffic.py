"""Reproducible first evaluation on a fixed Traffic sensor subset (not a paper benchmark)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import Windows,load_arrays
from train import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',default='outputs/traffic_effect_20261102')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2021,2022,2023])
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.csv,usecols=['date']+[str(i) for i in range(16)])
    y=frame[[str(i) for i in range(4)]].to_numpy(dtype=np.float32)
    z=frame[[str(i) for i in range(4,16)]].to_numpy(dtype=np.float32)
    dates=pd.to_datetime(frame.date)
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError('Traffic timestamps must be chronological and unique')
    futures=[]
    for step in range(1,6):
        dt=dates+pd.Timedelta(hours=step)
        hour=dt.dt.hour.to_numpy()/24*2*np.pi
        day=dt.dt.dayofweek.to_numpy()/7*2*np.pi
        futures.append(np.stack([np.sin(hour),np.cos(hour),np.sin(day),np.cos(day)],-1))
    protocol=dict(dataset='Traffic',source_csv_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
        rows=len(frame),targets=[str(i) for i in range(4)],auxiliary=[str(i) for i in range(4,16)],
        frequency='hourly',seq_len=48,pred_len=5,train_fraction=.7,val_fraction=.15,test_fraction=.15,
        epochs=args.epochs,seeds=args.seeds,batch_size=64,learning_rate=.001,device='cpu',
        caveats=['Fixed 16-sensor subset; not full 862-sensor benchmark','Not ride-hailing data',
                 'Equal epoch/width/depth budget, not matched parameter counts',
                 'Calendar is additional information available only to DLL_V4_calendar'])
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Output contains a different protocol; choose a new output directory')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    if not (out/'history_only.npz').exists(): np.savez(out/'history_only.npz',y=y,z=z)
    if not (out/'calendar.npz').exists(): np.savez(out/'calendar.npz',y=y,z=z,future=np.stack(futures,1).astype(np.float32))
    common=dict(d_model=16,n_heads=2,e_layers=1,dropout=.1,patch_len=4)
    variants=[('PatchTST','PatchTST','history_only',dict(common,d_ff=64,stride=2)),
              ('TimeXer','TimeXer','history_only',dict(common,d_ff=64)),
              ('DLL_V3_history','DLLTimeXer','history_only',dict(common,variant=3)),
              ('DLL_V4_calendar','DLLTimeXer','calendar',dict(common,variant=4))]
    records=[]
    for label,model,dataset,options in variants:
        config=out/(label+'.json');config.write_text(json.dumps(options,indent=2))
        for seed in args.seeds:
            destination=out/f'{label}_seed{seed}'
            log=out/f'{label}_seed{seed}.log'
            if not (destination/'metrics.json').exists():
                command=[sys.executable,'-u','train.py','--data',str(out/(dataset+'.npz')),
                         '--model',model,'--config',str(config),'--output',str(destination),
                         '--epochs',str(args.epochs),'--batch-size','64','--learning-rate','.001','--seed',str(seed)]
                print(f'START {label} seed={seed}',flush=True)
                with log.open('w') as stream:
                    subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=json.loads((destination/'metrics.json').read_text())
            history=json.loads((destination/'history.json').read_text())
            record=dict(model=label,seed=seed,**result,history=history)
            records.append(record)
            (out/'runs.json').write_text(json.dumps(records,indent=2))
            print(f'DONE {label} seed={seed}: MAE={result["mae"]:.8f} RMSE={result["rmse"]:.8f} seconds={result["seconds"]:.1f}',flush=True)
    windows=Windows(load_arrays(out/'history_only.npz'),48,5,'test')
    origins=np.array(windows.origins)
    true=np.stack([y[o:o+5] for o in origins])
    persistence=np.repeat(y[origins-1,None],5,axis=1)
    daily=np.stack([y[o-24:o-24+5] for o in origins])
    baselines={k:metrics(v,true) for k,v in [('Persistence',persistence),('DailySeasonal',daily)]}
    summary={}
    for name,_,_,_ in variants:
        runs=[r for r in records if r['model']==name]
        summary[name]={}
        for key in ('mae','rmse','mse','seconds'):
            values=[r[key] for r in runs]
            summary[name][key+'_mean']=float(np.mean(values))
            summary[name][key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        summary[name].update(parameters=runs[0]['parameters'],
            mae_per_horizon=np.mean([np.array(r['mae_per_horizon_target']).mean(-1) for r in runs],0).tolist(),
            mae_per_target=np.mean([np.array(r['mae_per_horizon_target']).mean(0) for r in runs],0).tolist())
    payload=dict(protocol=protocol,test_windows=len(origins),baselines=baselines,summary=summary)
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    print('SUMMARY '+str(out/'summary.json'),flush=True)

if __name__=='__main__':main()
