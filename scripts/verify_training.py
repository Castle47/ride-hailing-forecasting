"""Run bounded end-to-end checks. Results are not accuracy benchmarks."""
from pathlib import Path
import subprocess,sys,json,tempfile,numpy as np
root=Path(__file__).resolve().parents[1]
records=[]
(root/'outputs').mkdir(exist_ok=True)
run=Path(tempfile.mkdtemp(prefix='verification_',dir=root/'outputs'))
for model in ('CEEMDAN','Autoformer','TimeXer','DLLTimeXer'):
    out=str(run/model)
    cmd=[sys.executable,'train.py','--data','data/demo.npz','--model',model,'--output',out,
         '--seq-len','48','--pred-len','5','--epochs','2','--batch-size','2',
         '--max-train-steps','2','--max-eval-steps','2',
         '--config','configs/smoke_ceemdan.json' if model=='CEEMDAN' else 'configs/smoke.json']
    subprocess.run(cmd,cwd=root,check=True)
    before=np.load(root/out/'pred.npy')
    subprocess.run(cmd+['--evaluate'],cwd=root,check=True)
    np.testing.assert_allclose(before,np.load(root/out/'pred.npy'),rtol=0,atol=0)
    result=json.loads((root/out/'metrics.json').read_text())
    records.append(dict(model=model,training_and_reload='passed',samples=result['samples'],
                        prediction_shape=list(before.shape),parameters=result['parameters']))
(root/'docs/verification.json').write_text(json.dumps(dict(synthetic_only=True,checks=records),indent=2))
