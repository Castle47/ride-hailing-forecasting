"""Common training, validation selection, checkpoint reload and original-unit metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import load_arrays,Windows
from models.forecast import ForecastModel


def metrics(pred,true):
    error=pred-true
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
                mse=float((error**2).mean()),mae_per_horizon_target=np.abs(error).mean(0).tolist(),
                rmse_per_horizon_target=np.sqrt((error**2).mean(0)).tolist())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='NPZ: y[T,M], optional z[T,C], future[T,H,F], static[M,S]')
    p.add_argument('--model',choices=['CEEMDAN','Autoformer','TimeXer','DDLTimeXer','DLLTimeXer','PatchTST'],default='DLLTimeXer')
    p.add_argument('--output',default='outputs/run')
    p.add_argument('--seq-len',type=int,default=48);p.add_argument('--pred-len',type=int,default=5)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--seed',type=int,default=2021)
    p.add_argument('--device',default='cpu');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--config',help='JSON model options, e.g. lags, variant, d_model, CEEMDAN trials')
    p.add_argument('--max-train-steps',type=int,default=0);p.add_argument('--max-eval-steps',type=int,default=0)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.learning_rate<=0 or min(args.max_train_steps,args.max_eval_steps)<0:
        p.error('Invalid training budget')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'best.pt'
    if checkpoint.exists() and not args.evaluate:
        p.error('Output already contains a checkpoint; choose a new output or use --evaluate')
    arrays=load_arrays(args.data)
    digest=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    options=json.loads(Path(args.config).read_text()) if args.config else {}
    spec=dict(name=args.model,seq_len=args.seq_len,pred_len=args.pred_len,
              n_targets=arrays['y'].shape[1],n_aux=arrays['z'].shape[1])
    if args.model in {'DDLTimeXer','DLLTimeXer'}:
        options.update(future_dim=arrays.get('future',np.empty((0,0,0))).shape[-1],
                       static_dim=arrays.get('static',np.empty((0,0))).shape[-1])
    state=None
    if args.evaluate:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['data_sha256']!=digest:raise ValueError('Evaluation data differs from training dataset')
        spec,options=state['spec'],state['options']
    datasets={};stats=state['stats'] if state else None
    for split in ('train','val','test'):
        datasets[split]=Windows(arrays,spec['seq_len'],spec['pred_len'],split,stats=stats)
        stats=datasets[split].stats
    loaders={k:DataLoader(v,batch_size=args.batch_size,shuffle=k=='train') for k,v in datasets.items()}
    model=ForecastModel(**spec,**options).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    def predict(batch):
        batch={k:v.to(args.device) for k,v in batch.items()}
        prediction=model(batch['y'],batch['z'],batch.get('future'),batch.get('static'))
        return prediction,batch['label']
    def evaluate(split):
        model.eval();preds=[];trues=[]
        with torch.no_grad():
            for i,batch in enumerate(loaders[split]):
                if args.max_eval_steps and i>=args.max_eval_steps:break
                pred,true=predict(batch);preds.append(pred.cpu().numpy());trues.append(true.cpu().numpy())
        return np.concatenate(preds),np.concatenate(trues)
    start=time.perf_counter();history=[];best=float('inf')
    if not args.evaluate:
        for epoch in range(args.epochs):
            model.train();total=0.;count=0
            for i,batch in enumerate(loaders['train']):
                if args.max_train_steps and i>=args.max_train_steps:break
                optimizer.zero_grad();pred,true=predict(batch);loss=(pred-true).square().mean()
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                loss.backward();optimizer.step();total+=loss.item()*true.numel();count+=true.numel()
            pred,true=evaluate('val');val=float(((pred-true)**2).mean())
            history.append(dict(epoch=epoch+1,train_mse=total/count,val_mse=val))
            print(history[-1],flush=True)
            if not np.isfinite(val):raise ValueError('Non-finite validation loss')
            if val<best:
                best=val
                safe_stats={k:tuple(torch.tensor(x) for x in v) for k,v in stats.items()}
                torch.save(dict(model=model.state_dict(),spec=spec,options=options,stats=safe_stats,
                                data_sha256=digest,epoch=epoch+1,seed=args.seed),checkpoint)
        (out/'history.json').write_text(json.dumps(history,indent=2))
    state=torch.load(checkpoint,map_location=args.device,weights_only=True)
    model.load_state_dict(state['model'])
    pred,true=evaluate('test');mean,scale=stats['y']
    if isinstance(mean,torch.Tensor):mean,scale=mean.numpy(),scale.numpy()
    pred=pred*scale+mean;true=true*scale+mean
    result=metrics(pred,true);result.update(samples=len(pred),parameters=sum(x.numel() for x in model.parameters()),
        seconds=time.perf_counter()-start,smoke_run=bool(args.max_train_steps or args.max_eval_steps),checkpoint_epoch=state['epoch'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'config.json').write_text(json.dumps(dict(args=vars(args),model=spec,options=options),indent=2))
    np.save(out/'pred.npy',pred);np.save(out/'true.npy',true)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
