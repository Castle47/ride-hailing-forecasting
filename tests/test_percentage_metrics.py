import numpy as np
import pytest
from utils.percentage_metrics import percentage_metrics


def test_zero_targets_excluded_only_from_mape():
    y=np.array([0.,1.,2.]).reshape(1,3,1);p=np.array([1.,2.,1.]).reshape(1,3,1)
    r=percentage_metrics(p,y,1e-8)
    assert r['mape_macro_percent']==75 and r['wmape_pooled_percent']==100
    assert r['excluded_per_target']==[1]


def test_macro_and_pooled_weighting_differ():
    y=np.array([1.,100.]).reshape(1,1,2);p=np.array([2.,110.]).reshape(1,1,2)
    r=percentage_metrics(p,y)
    assert r['mape_macro_percent']==55 and r['wmape_macro_percent']==55
    assert r['wmape_pooled_percent']==pytest.approx(1100/101)


def test_negative_and_all_zero_targets():
    r=percentage_metrics(np.array([-1.,-2.]).reshape(1,2,1),np.array([-2.,-4.]).reshape(1,2,1))
    assert r['mape_macro_percent']==50 and r['wmape_pooled_percent']==50
    r=percentage_metrics(np.ones((1,2,1)),np.zeros((1,2,1)))
    assert r['mape_macro_percent'] is None and r['wmape_pooled_percent'] is None


def test_near_zero_mask_without_denominator_clipping():
    y=np.array([1e-10,1e-3]).reshape(1,2,1);p=np.zeros_like(y)
    r=percentage_metrics(p,y,1e-8)
    assert r['mape_macro_percent']==100 and r['excluded_per_target']==[1]
