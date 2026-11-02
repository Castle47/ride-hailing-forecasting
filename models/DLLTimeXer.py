"""Dynamic lead-lag TimeXer; implements chapter 5 (Dynamic Lead–Lag TimeXer)."""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class Config:
    seq_len: int = 48
    pred_len: int = 5
    n_targets: int = 2
    n_aux: int = 3
    patch_len: int = 4
    lag_window: int = 8
    lags: tuple = (0, 1, 2, 3, 4, 6, 9, 12)
    corr_window: int = 16
    d_model: int = 32
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.1
    future_dim: int = 0
    static_dim: int = 0
    variant: int = 4
    gamma_init: float = 0.01
    beta_init: float = 0.


def lagged_correlation(y, z, lags, window):
    """Corr(dy[s], dz[s-lag]); positive lag means auxiliary leads target.

    Missing auxiliary pairs are omitted, constant/insufficient pairs yield zero.
    Only history tensors are accepted. Returns [B, M, C, K].
    """
    dy, dz = y.diff(dim=1), z.diff(dim=1)
    result = []
    for lag in lags:
        n = min(window, dy.shape[1] - lag)
        if n < 2:
            raise ValueError('History too short for correlation window/lags')
        a = dy[:, -n:, :, None]
        b = dz[:, dz.shape[1]-lag-n:dz.shape[1]-lag, None, :]
        valid = torch.isfinite(a) & torch.isfinite(b)
        count = valid.sum(1).clamp_min(1)
        aa, bb = torch.where(valid, a, 0.), torch.where(valid, b, 0.)
        aa = torch.where(valid, a - aa.sum(1, keepdim=True)/count[:, None], 0.)
        bb = torch.where(valid, b - bb.sum(1, keepdim=True)/count[:, None], 0.)
        denom = (aa.square().sum(1) * bb.square().sum(1)).clamp_min(1e-24).sqrt()
        r = (aa*bb).sum(1) / denom.clamp_min(1e-8)
        result.append(torch.where((valid.sum(1) >= 2) & (denom > 1e-8), r, 0.).clamp(-1, 1))
    return torch.stack(result, -1)


class Model(nn.Module):
    """y:[B,L,M], z:[B,L,C], future:[B,H,F], static:[B,M,S].

    V0: full-history variate token, shared horizon query; V1: lag tokens;
    V2: horizon queries; V3: correlation bias; V4: context in query.
    Optional spatial_bias:[M,C] or [B,M,C] is additive attention prior.
    """
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.lags = (0,) if c.variant == 0 else tuple(c.lags)
        if (c.variant not in range(5) or c.n_targets < 1 or c.n_aux < 0
                or c.pred_len < 1 or c.seq_len < 4 or c.patch_len < 1
                or c.patch_len > c.seq_len or c.seq_len % c.patch_len
                or c.n_heads < 1 or c.d_model < 2 or c.d_model % c.n_heads or c.e_layers < 1
                or c.corr_window < 2 or not 0 <= c.dropout < 1
                or c.future_dim < 0 or c.static_dim < 0):
            raise ValueError('Invalid DLL-TimeXer dimensions/configuration')
        if not self.lags or len(set(self.lags)) != len(self.lags) or any(type(i) is not int or i < 0 for i in self.lags):
            raise ValueError('lags must be unique nonnegative integers')
        self.window = c.seq_len if c.variant == 0 else c.lag_window
        if self.window < 1 or max(self.lags)+self.window > c.seq_len or max(self.lags)+3 > c.seq_len:
            raise ValueError('Lag windows/correlation exceed available history')
        d, p = c.d_model, c.seq_len // c.patch_len
        self.patch = nn.Linear(c.patch_len, d)
        self.position = nn.Parameter(torch.randn(1, p+1, d)*.02)
        self.global_token = nn.Parameter(torch.randn(1, 1, d)*.02)
        layer = nn.TransformerEncoderLayer(d, c.n_heads, d*4, c.dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, c.e_layers)
        self.base_head = nn.Linear((p+1)*d, c.pred_len)
        self.var_embedding = nn.Embedding(max(c.n_aux, 1), d)
        self.lag_embedding = nn.Embedding(len(self.lags), d)
        self.aux_embed = nn.Linear(self.window, d)
        self.horizon_embedding = nn.Embedding(c.pred_len, d)
        self.future_encoder = nn.Linear(c.future_dim, d) if c.future_dim else None
        self.static_encoder = nn.Linear(c.static_dim, d) if c.static_dim else None
        self.query = nn.Linear(4*d, d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.bias_net = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, 1))
        self.beta = nn.Parameter(torch.tensor(float(c.beta_init)))
        self.gamma = nn.Parameter(torch.tensor(float(c.gamma_init)))
        self.residual_head = nn.Linear(d, 1)

    def forward(self, y, z=None, future=None, static=None, spatial_bias=None, return_details=False):
        c = self.config
        if y.ndim != 3 or y.shape[1:] != (c.seq_len, c.n_targets) or not torch.isfinite(y).all():
            raise ValueError('y must be finite [B,seq_len,n_targets]')
        b, _, m = y.shape
        if z is None:
            z = y.new_full((b, c.seq_len, c.n_aux), float('nan'))
        if z.shape != (b, c.seq_len, c.n_aux):
            raise ValueError('z shape must be [B,seq_len,n_aux]')
        for value, dim, shape, name in [(future,c.future_dim,(b,c.pred_len,c.future_dim),'future'),
                                       (static,c.static_dim,(b,m,c.static_dim),'static')]:
            if dim and (value is None or value.shape != shape or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite {shape}')
            if not dim and value is not None and value.numel():
                raise ValueError(f'{name} dimension is zero in config')
        mean = y.mean(1, keepdim=True).detach()
        scale = y.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        yn = (y-mean)/scale
        patches = yn.transpose(1,2).unfold(-1,c.patch_len,c.patch_len)
        tokens = self.patch(patches).reshape(b*m,-1,c.d_model)
        tokens = torch.cat((tokens,self.global_token.expand(b*m,-1,-1)),1)
        hidden = self.encoder(tokens+self.position)
        base = self.base_head(hidden.flatten(1)).reshape(b,m,c.pred_len).transpose(1,2)
        global_state = hidden[:,-1].reshape(b,m,c.d_model)
        residual = torch.zeros_like(base)
        weights = y.new_zeros(b,m,c.pred_len,c.n_aux,len(self.lags))
        corr = y.new_zeros(b,m,c.n_aux,len(self.lags))
        if c.n_aux:
            valid = torch.isfinite(z)
            clean = torch.where(valid,z,0.)
            zm = clean.sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
            centered = torch.where(valid,z-zm,0.)
            zs = (centered.square().sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)+1e-5).sqrt()
            zn = centered/zs
            windows = torch.stack([zn[:,c.seq_len-l-self.window:c.seq_len-l].transpose(1,2) for l in self.lags],2)
            available = torch.stack([valid[:,c.seq_len-l-self.window:c.seq_len-l].all(1) for l in self.lags],-1)
            aux = self.aux_embed(windows)
            aux = aux + self.var_embedding.weight[:c.n_aux][None,:,None] + self.lag_embedding.weight[None,None]
            aux = aux.flatten(1,2)
            g = global_state[:,:,None].expand(-1,-1,c.pred_len,-1)
            h = self.horizon_embedding.weight[None,None].expand(b,m,-1,-1) if c.variant>=2 else torch.zeros_like(g)
            f, s = torch.zeros_like(g), torch.zeros_like(g)
            if c.variant>=4:
                if self.future_encoder is not None: f = self.future_encoder(future)[:,None].expand(-1,m,-1,-1)
                if self.static_encoder is not None: s = self.static_encoder(static)[:,:,None].expand(-1,-1,c.pred_len,-1)
            q = self.query(torch.cat((g,h,f,s),-1))
            scores = torch.einsum('bmhd,btd->bmht',q,self.key(aux))/math.sqrt(c.d_model)
            if c.variant>=3:
                corr = lagged_correlation(y,z,self.lags,c.corr_window)
                r = corr[:,:,None].expand(-1,-1,c.pred_len,-1,-1)
                lag = y.new_tensor(self.lags)[None,None,None,None].expand_as(r)/max(c.seq_len-1,1)
                horizon = torch.arange(1,c.pred_len+1,device=y.device,dtype=y.dtype)[None,None,:,None,None].expand_as(r)/c.pred_len
                bias = self.bias_net(torch.stack((r,r.abs(),lag,horizon),-1)).squeeze(-1)
                scores = scores+self.beta*bias.flatten(-2)
            if spatial_bias is not None:
                if spatial_bias.shape == (m,c.n_aux): spatial_bias = spatial_bias[None].expand(b,-1,-1)
                if spatial_bias.shape != (b,m,c.n_aux) or not torch.isfinite(spatial_bias).all():
                    raise ValueError('spatial_bias must be finite [M,C] or [B,M,C]')
                scores = scores+spatial_bias[:,:,None,:,None].expand(-1,-1,c.pred_len,-1,len(self.lags)).flatten(-2)
            mask = available.flatten(1)[:,None,None]
            # Finite sentinel plus renormalization handles fully missing auxiliaries.
            a = scores.masked_fill(~mask,torch.finfo(scores.dtype).min).softmax(-1)*mask
            a = a/a.sum(-1,keepdim=True).clamp_min(1e-8)
            residual = self.residual_head(torch.einsum('bmht,btd->bmhd',a,self.value(aux))).squeeze(-1).transpose(1,2)
            residual = residual*available.any(dim=(1,2))[:,None,None]
            weights = a.reshape(b,m,c.pred_len,c.n_aux,len(self.lags))
        prediction = (base+self.gamma*residual)*scale+mean
        if return_details:
            return prediction,dict(base=base*scale+mean,residual=residual*scale,
                                   attention=weights,correlation=corr,gamma=self.gamma,beta=self.beta)
        return prediction
