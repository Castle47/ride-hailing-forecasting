"""Percentage errors in original units, with explicit zero-target handling."""
import numpy as np


def percentage_metrics(pred, true, epsilon=0.):
    pred, true = np.asarray(pred,dtype=np.float64), np.asarray(true,dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 3 or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError('Expected finite matching [windows,horizon,targets] arrays')
    eps = np.broadcast_to(np.asarray(epsilon,dtype=float),(true.shape[-1],))
    if not np.isfinite(eps).all() or (eps<0).any():raise ValueError('Invalid epsilon')
    error, denominator = np.abs(pred-true), np.abs(true)
    valid=denominator>eps
    def mean_valid(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    per_mape=[];per_wmape=[]
    for j in range(true.shape[-1]):
        mask=valid[...,j]
        per_mape.append(float(100*np.mean(error[...,j][mask]/denominator[...,j][mask])) if mask.any() else None)
        den=denominator[...,j].sum()
        per_wmape.append(float(100*error[...,j].sum()/den) if den>0 else None)
    den=denominator.sum()
    return dict(mape_macro_percent=mean_valid(per_mape),wmape_macro_percent=mean_valid(per_wmape),
                wmape_pooled_percent=float(100*error.sum()/den) if den>0 else None,
                per_target_mape_percent=per_mape,per_target_wmape_percent=per_wmape,
                excluded_per_target=(~valid).sum(axis=(0,1)).tolist(),
                observations_per_target=int(np.prod(true.shape[:-1])))
