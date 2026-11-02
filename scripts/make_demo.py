"""Synthetic demonstration only; never a business benchmark."""
from pathlib import Path
import argparse
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',default='data/demo.npz');a=p.parse_args()
rng=np.random.default_rng(47);t=360;h=5
z=rng.normal(size=(t,3)).astype('float32')
y=np.zeros((t,2),dtype='float32')
for i in range(3,t):y[i]=.5*y[i-1]+[z[i-2,0],-z[i-3,1]]+.05*rng.normal(size=2)
future=np.stack([np.stack([np.sin((np.arange(t)+step)/24),np.cos((np.arange(t)+step)/24)],-1) for step in range(1,h+1)],1).astype('float32')
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
np.savez(a.output,y=y,z=z,future=future,static=np.eye(2,dtype='float32'))
print(a.output)
