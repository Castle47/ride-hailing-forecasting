"""Chronological windows. No decomposition/statistics are fit on future labels."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def load_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        arrays={k:np.array(archive[k],dtype=np.float32) for k in archive.files}
    if 'y' not in arrays or arrays['y'].ndim!=2 or not np.isfinite(arrays['y']).all():
        raise ValueError('NPZ requires finite y[T,M]')
    t,m=arrays['y'].shape
    if 'z' not in arrays:arrays['z']=np.empty((t,0),dtype=np.float32)
    if arrays['z'].ndim!=2 or len(arrays['z'])!=t or np.isinf(arrays['z']).any():
        raise ValueError('z must be [T,C]; NaN denotes unavailable history')
    if 'future' in arrays and (arrays['future'].ndim!=3 or len(arrays['future'])!=t or not np.isfinite(arrays['future']).all()):
        raise ValueError('future must be finite [T,H,F], indexed by forecast origin')
    if 'static' in arrays and (arrays['static'].ndim!=2 or arrays['static'].shape[0]!=m or not np.isfinite(arrays['static']).all()):
        raise ValueError('static must be finite [M,S]')
    return arrays


class Windows(Dataset):
    def __init__(self,arrays,seq_len,pred_len,split,train_end=None,val_end=None,stats=None):
        self.arrays=arrays; self.seq_len=seq_len; self.pred_len=pred_len
        t=len(arrays['y']); train_end=train_end or int(.7*t); val_end=val_end or int(.85*t)
        if not 0<train_end<val_end<t:raise ValueError('Invalid chronological split boundaries')
        lo,hi={'train':(seq_len,train_end),'val':(train_end,val_end),'test':(val_end,t)}[split]
        self.origins=list(range(max(lo,seq_len),hi-pred_len+1))
        if not self.origins:raise ValueError(f'{split} has no complete windows')
        self.stats={} if stats is None else {k:tuple(np.asarray(x) for x in v) for k,v in stats.items()}
        for key in ('y','z'):
            if stats is None:
                a=arrays[key][:train_end]
                # All-missing auxiliary columns use neutral statistics and stay NaN.
                count=np.isfinite(a).sum(0).clip(1)
                mean=np.nansum(a,0)/count
                scale=np.sqrt(np.nansum((a-mean)**2,0)/count).clip(1e-5)
                self.stats[key]=(mean.astype(np.float32),scale.astype(np.float32))
        self.y=(arrays['y']-self.stats['y'][0])/self.stats['y'][1]
        self.z=(arrays['z']-self.stats['z'][0])/self.stats['z'][1]
        if 'future' in arrays and arrays['future'].shape[1]!=pred_len:
            raise ValueError('future horizon must equal pred_len')

    def __len__(self):return len(self.origins)

    def __getitem__(self,index):
        origin=self.origins[index]; a=origin-self.seq_len
        result={'y':self.y[a:origin],'z':self.z[a:origin],
                'label':self.y[origin:origin+self.pred_len]}
        # Row origin-1 is the last observed timestamp; feature values are forecasts
        # issued at this timestamp, not future observations sampled from a series.
        if 'future' in self.arrays:result['future']=self.arrays['future'][origin-1]
        if 'static' in self.arrays:result['static']=self.arrays['static']
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()}
