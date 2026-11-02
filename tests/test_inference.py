import numpy as np
import torch
from models.DLLTimeXer import Config, Model
from data import Windows
from predict import predict


def test_zero_auxiliary_and_constant_inputs_backward():
    model=Model(Config(seq_len=24,pred_len=3,n_targets=2,n_aux=0,patch_len=4,
                       d_model=8,n_heads=2,e_layers=1,dropout=0.,lags=(0,1),lag_window=4))
    y=torch.ones(2,24,2,requires_grad=True)
    out,details=model(y,return_details=True)
    torch.testing.assert_close(out,details['base'])
    out.square().mean().backward()
    assert torch.isfinite(y.grad).all()


def test_future_context_cannot_change_historical_correlation():
    model=Model(Config(seq_len=24,pred_len=3,n_targets=1,n_aux=1,patch_len=4,
                       d_model=8,n_heads=2,e_layers=1,dropout=0.,future_dim=1,
                       lags=(0,1,3),lag_window=4))
    y,z=torch.randn(1,24,1),torch.randn(1,24,1)
    _,a=model(y,z,torch.zeros(1,3,1),return_details=True)
    _,b=model(y,z,torch.ones(1,3,1)*1e6,return_details=True)
    torch.testing.assert_close(a['correlation'],b['correlation'])


def test_deployment_inference_without_labels_matches_direct_model(tmp_path):
    config=Config(seq_len=24,pred_len=3,n_targets=2,n_aux=1,patch_len=4,
                  d_model=8,n_heads=2,e_layers=1,dropout=0.,lags=(0,1),lag_window=4)
    model=Model(config).eval()
    options={k:v for k,v in vars(config).items() if k not in ['seq_len','pred_len','n_targets','n_aux']}
    spec=dict(name='DLLTimeXer',seq_len=24,pred_len=3,n_targets=2,n_aux=1)
    stats={'y':(torch.tensor([10.,20.]),torch.tensor([2.,3.])),
           'z':(torch.tensor([4.]),torch.tensor([2.]))}
    # The shared adapter nests the independently tested model under core.
    state={'model':{'core.'+k:v for k,v in model.state_dict().items()},'spec':spec,'options':options,'stats':stats}
    checkpoint=tmp_path/'best.pt';torch.save(state,checkpoint)
    y,z=np.random.randn(24,2).astype('float32'),np.random.randn(24,1).astype('float32')
    actual=predict(checkpoint,y,z)
    with torch.no_grad():
        expected=model(torch.tensor((y-[10.,20.])/[2.,3.],dtype=torch.float32)[None],
                       torch.tensor((z-4.)/2.,dtype=torch.float32)[None])[0].numpy()*[2.,3.]+[10.,20.]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
