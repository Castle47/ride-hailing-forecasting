"""Shared input/output contract for the four maintained model families."""
from types import SimpleNamespace
import importlib
import torch
from torch import nn
from models.DLLTimeXer import Config as DDLConfig, Model as DDLModel


class ForecastModel(nn.Module):
    def __init__(self, name, seq_len, pred_len, n_targets, n_aux, **options):
        super().__init__()
        self.name, self.seq_len, self.pred_len = name, seq_len, pred_len
        self.n_targets, self.n_aux = n_targets, n_aux
        if name in {'DDLTimeXer','DLLTimeXer'}:
            self.core = DDLModel(DDLConfig(seq_len=seq_len,pred_len=pred_len,
                                          n_targets=n_targets,n_aux=n_aux,**options))
            return
        # Future/static context belongs exclusively to the DDL extension.
        for key in ('future_dim','static_dim','variant','lags','lag_window','corr_window','gamma_init','beta_init'):
            options.pop(key,None)
        if name not in {'CEEMDAN','Autoformer','TimeXer','PatchTST'}:
            raise ValueError(f'Unknown model: {name}')
        channels = n_targets+n_aux if name in {'TimeXer','Autoformer'} else n_targets
        defaults = dict(seq_len=seq_len,label_len=seq_len//2,pred_len=pred_len,
                        enc_in=channels,dec_in=channels,c_out=channels,
                        task_name='long_term_forecast',features='MS',
                        target_indices=list(range(n_targets)),use_norm=1,
                        d_model=32,n_heads=4,d_ff=128,e_layers=2,d_layers=1,
                        dropout=.1,activation='gelu',factor=1,embed='timeF',freq='h',
                        embed_type=0,output_attention=False,moving_avg=5,
                        patch_len=4,stride=2,padding_patch='end',decomposition=0,
                        individual=0,revin=1,affine=0,subtract_last=0,kernel_size=5,
                        head_dropout=0.,fc_dropout=.05,progressive_decomp='ceemdan',
                        fusion_attention='hybrid',encoder_norm='BatchNorm')
        defaults.update(options)
        if defaults['d_model']%defaults['n_heads'] or defaults['d_model']%2:
            raise ValueError('d_model must be even and divisible by n_heads')
        self.config = SimpleNamespace(**defaults)
        module = 'CEEMDANPatchTST' if name=='CEEMDAN' else name
        self.core = importlib.import_module('models.'+module).Model(self.config)

    def forward(self,y,z=None,future=None,static=None,return_details=False):
        if self.name in {'DDLTimeXer','DLLTimeXer'}:
            return self.core(y,z,future,static,return_details=return_details)
        if y.shape[1:] != (self.seq_len,self.n_targets) or not torch.isfinite(y).all():
            raise ValueError('Invalid target history')
        if self.name in {'TimeXer','Autoformer'} and self.n_aux:
            if z is None or z.shape != (*y.shape[:2],self.n_aux) or not torch.isfinite(z).all():
                raise ValueError('Baseline auxiliary history must be finite and complete')
            x = torch.cat((y,z),-1)
        else:
            x = y
        if self.name in {'CEEMDAN','PatchTST'}:
            out = self.core(x)
        else:
            dec = torch.cat((x[:,-self.config.label_len:],x.new_zeros(len(x),self.pred_len,x.shape[-1])),1)
            out = self.core(x,None,dec,None)
        if isinstance(out,tuple):out=out[0]
        out = out[:,-self.pred_len:,:self.n_targets]
        return (out,{}) if return_details else out
