"""Measure learned DLL residual gates and base-vs-full errors without refitting."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import load_arrays,Windows
from models.forecast import ForecastModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/traffic_effect_20261102');args=p.parse_args()
    root=Path(args.results);records=[];torch.set_num_threads(1)
    for folder in sorted(root.glob('DLL*_seed*')):
        if not folder.is_dir():continue
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        arrays=load_arrays(root/('calendar.npz' if folder.name.startswith('DLL_V4') else 'history_only.npz'))
        data=Windows(arrays,48,5,'test',stats=state['stats'])
        model=ForecastModel(**state['spec'],**state['options']).eval();model.load_state_dict(state['model'])
        base_errors=[];full_errors=[];corrections=[]
        scale=data.stats['y'][1]
        with torch.no_grad():
            for batch in DataLoader(data,batch_size=64):
                pred,details=model(batch['y'],batch['z'],batch.get('future'),batch.get('static'),return_details=True)
                true=batch['label'].numpy()
                base_errors.append(np.abs((details['base'].numpy()-true)*scale))
                full_errors.append(np.abs((pred.numpy()-true)*scale))
                corrections.append(np.abs((pred-details['base']).numpy()*scale))
        records.append(dict(run=folder.name,beta=float(model.core.beta.detach()),gamma=float(model.core.gamma.detach()),
            base_mae=float(np.concatenate(base_errors).mean()),full_mae=float(np.concatenate(full_errors).mean()),
            correction_mean_abs=float(np.concatenate(corrections).mean())))
    (root/'diagnostics.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':main()
