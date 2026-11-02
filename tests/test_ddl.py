import numpy as np
import pytest
import torch
from models.DLLTimeXer import Config,Model,lagged_correlation
from models.forecast import ForecastModel
from data import Windows


def config(**kwargs):
    options=dict(seq_len=24,pred_len=3,n_targets=2,n_aux=2,patch_len=4,lag_window=4,
                 lags=(0,1,3),corr_window=12,d_model=8,n_heads=2,e_layers=1,dropout=0.)
    options.update(kwargs);return Config(**options)


@pytest.mark.parametrize('variant',range(5))
def test_variants_gradients_attention_and_reload(variant):
    torch.manual_seed(3);c=config(variant=variant,future_dim=2,static_dim=2)
    model=Model(c)
    y=torch.randn(2,24,2);z=torch.randn(2,24,2)
    future=torch.randn(2,3,2,requires_grad=True);static=torch.randn(2,2,2,requires_grad=True)
    pred,info=model(y,z,future,static,return_details=True)
    assert pred.shape==(2,3,2)
    assert info['attention'].shape==(2,2,3,2,1 if variant==0 else 3)
    torch.testing.assert_close(info['attention'].flatten(-2).sum(-1),torch.ones(2,2,3))
    pred.square().mean().backward()
    assert model.patch.weight.grad.abs().sum()>0
    assert model.aux_embed.weight.grad.abs().sum()>0
    if variant==4:
        assert future.grad.abs().sum()>0 and static.grad.abs().sum()>0
    copy=Model(c);copy.load_state_dict(model.state_dict());copy.eval();model.eval()
    torch.testing.assert_close(copy(y,z,future,static),model(y,z,future,static))


def test_zero_gate_and_missing_auxiliary_fall_back_to_target_only():
    c=config(gamma_init=0.)
    model=Model(c).eval();y=torch.randn(2,24,2);z=torch.randn(2,24,2)
    pred,info=model(y,z,return_details=True)
    torch.testing.assert_close(pred,info['base'])
    torch.testing.assert_close(pred,model(y,z*100))
    model.gamma.data.fill_(1.)
    pred,info=model(y,torch.full_like(z,float('nan')),return_details=True)
    torch.testing.assert_close(pred,info['base']);assert info['attention'].sum()==0
    z[0,:,0]=float('nan');pred,info=model(y,z,return_details=True)
    assert torch.isfinite(pred).all();assert info['attention'][0,:,:,0].sum()==0


def test_lag_direction_negative_correlation_and_constant_signals():
    torch.manual_seed(9);z=torch.randn(1,80,1);y=torch.zeros_like(z)
    y[:,3:]=-z[:,:-3]
    r=lagged_correlation(y,z,(0,1,3,6),40)[0,0,0]
    assert r[2]<-.999 and r.abs().argmax()==2
    r=lagged_correlation(torch.ones_like(y),z,(0,3),20)
    assert torch.isfinite(r).all() and r.abs().sum()==0


def test_context_and_prior_change_only_auxiliary_path():
    model=Model(config(future_dim=1,static_dim=1,gamma_init=1.)).eval()
    y,z=torch.randn(2,24,2),torch.randn(2,24,2)
    f,s=torch.zeros(2,3,1),torch.zeros(2,2,1)
    a,ia=model(y,z,f,s,return_details=True)
    b,ib=model(y,z,f+10,s+10,return_details=True)
    torch.testing.assert_close(ia['base'],ib['base']);assert not torch.allclose(a,b)
    _,ip=model(y,z,f,s,spatial_bias=torch.tensor([[10.,-10.],[10.,-10.]]),return_details=True)
    assert not torch.allclose(ia['attention'],ip['attention'])


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):Model(config(lags=(100,)))
    with pytest.raises(ValueError):Model(config(lags=(-1,)))
    with pytest.raises(ValueError):Model(config(future_dim=2))(torch.randn(1,24,2),torch.randn(1,24,2))


def test_windows_use_training_statistics_and_origin_available_context():
    y=np.arange(200,dtype='float32').reshape(100,2)
    a=dict(y=y,z=y.copy(),future=np.arange(300,dtype='float32').reshape(100,3,1))
    train=Windows(a,12,3,'train');val=Windows(a,12,3,'val',stats=train.stats)
    assert train.origins[-1]+3==70 and val.origins[0]==70
    np.testing.assert_allclose(val.stats['y'][0],y[:70].mean(0))
    np.testing.assert_array_equal(val[0]['future'],a['future'][69])
    changed={**a,'y':y.copy()};changed['y'][70:]+=1e6
    second=Windows(changed,12,3,'train')
    np.testing.assert_array_equal(train[0]['y'],second[0]['y'])
    np.testing.assert_array_equal(train.stats['y'][0],second.stats['y'][0])


@pytest.mark.parametrize('name',['CEEMDAN','Autoformer','TimeXer','DDLTimeXer'])
def test_shared_contract_backward(name):
    opts=dict(d_model=8,n_heads=2,e_layers=1,dropout=0.,patch_len=4)
    if name=='CEEMDAN':opts.update(ceemdan_trials=2,ceemdan_max_imf=1,ceemdan_max_sift=5)
    if name=='DDLTimeXer':opts.update(lags=(0,1,3),lag_window=4,corr_window=8)
    model=ForecastModel(name,24,3,2,2,**opts)
    pred=model(torch.randn(2,24,2),torch.randn(2,24,2))
    assert pred.shape==(2,3,2) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
