"""Forecast from a saved checkpoint without future target labels."""
import argparse
from pathlib import Path
import numpy as np
import torch
from models.forecast import ForecastModel


def predict(checkpoint, y, z=None, future=None, static=None, device='cpu', details=False):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=ForecastModel(**state['spec'],**state['options']).to(device).eval()
    model.load_state_dict(state['model'])
    ym,ys=(v.numpy() for v in state['stats']['y'])
    zm,zs=(v.numpy() for v in state['stats']['z'])
    def tensor(a):return None if a is None else torch.as_tensor(a,dtype=torch.float32,device=device).unsqueeze(0)
    zn=None if z is None else (z-zm)/zs
    with torch.no_grad():
        output,info=model(tensor((y-ym)/ys),tensor(zn),tensor(future),tensor(static),return_details=True)
    prediction=output[0].cpu().numpy()*ys+ym
    return (prediction,info) if details else prediction


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',default='outputs/prediction.npy');p.add_argument('--device',default='cpu')
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as archive:
        values={k:archive[k] for k in ('y','z','future','static') if k in archive.files}
    prediction=predict(a.checkpoint,device=a.device,**values)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.save(a.output,prediction)
    print(prediction)

if __name__=='__main__':main()
