"""Post-hoc MAPE/WMAPE of frozen Traffic predictions; no retraining/reselection."""
import argparse,json,hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import load_arrays,Windows
from utils.percentage_metrics import percentage_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/traffic_effect_20261102')
    p.add_argument('--destination',default='docs/traffic_effect_20261102');a=p.parse_args()
    source=Path(a.results);out=Path(a.destination);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    arrays=load_arrays(source/'history_only.npz');y=arrays['y'];train_end=int(len(y)*.7)
    epsilon=np.maximum(1e-8,1e-6*np.mean(np.abs(y[:train_end].astype(np.float64)),axis=0))
    reference=np.load(source/'TimeXer_seed2021/true.npy')
    w=Windows(arrays,48,5,'test');origins=np.array(w.origins)
    np.testing.assert_allclose(reference,np.stack([y[o:o+5] for o in origins]),atol=1e-7,rtol=1e-6)
    records=[]
    for name in ('PatchTST','TimeXer','DLL_V3_history','DLL_V4_calendar'):
        for seed in protocol['seeds']:
            folder=source/f'{name}_seed{seed}'
            np.testing.assert_array_equal(reference,np.load(folder/'true.npy'))
            pred=np.load(folder/'pred.npy')
            scores=percentage_metrics(pred,reference,epsilon)
            horizon=[percentage_metrics(pred[:,h:h+1],reference[:,h:h+1],epsilon) for h in range(5)]
            records.append(dict(model=name,seed=seed,metrics=scores,per_horizon=horizon,
                prediction_sha256=hashlib.sha256((folder/'pred.npy').read_bytes()).hexdigest()))
    baselines={}
    for name,pred in [('Persistence',np.repeat(y[origins-1,None],5,axis=1)),
                      ('DailySeasonal',np.stack([y[o-24:o-19] for o in origins]))]:
        baselines[name]=percentage_metrics(pred,reference,epsilon)
    keys=('mape_macro_percent','wmape_pooled_percent','wmape_macro_percent')
    summary={}
    for name in ('PatchTST','TimeXer','DLL_V3_history','DLL_V4_calendar'):
        runs=[r['metrics'] for r in records if r['model']==name];summary[name]={}
        for key in keys:
            vals=[r[key] for r in runs]
            summary[name][key]=dict(mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)))
    coverage=records[0]['metrics']
    metadata=dict(retrained=False,checkpoint_reselected=False,original_unit_metrics=True,
        mape='100 * mean(abs(pred-true)/abs(true)); excludes abs(true)<=epsilon; equal-weight mean across targets',
        wmape_pooled='100 * sum(abs(pred-true)) / sum(abs(true)); includes zero-target error in numerator',
        wmape_macro='Equal-weight average of per-target WMAPE',
        epsilon_rule='max(1e-8, 1e-6 * training mean absolute target), original units, same mask for every model',
        epsilon=epsilon.tolist(),test_windows=len(reference),horizon=5,targets=protocol['targets'],
        excluded_per_target=coverage['excluded_per_target'],observations_per_target=coverage['observations_per_target'],
        target_min=np.min(reference,axis=(0,1)).tolist(),target_median=np.median(reference,axis=(0,1)).tolist())
    payload=dict(protocol=metadata,summary=summary,baselines=baselines,runs=records)
    (out/'percentage_metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Traffic 测试集的 MAPE / WMAPE','',
        '复用上一轮12份固定预测、相同真值和检查点；没有重训、重新选模或修改标签。神经模型列出3个随机种子的均值±样本标准差。','',
        '| 模型 | MAPE（目标等权） | WMAPE（总体合并） | WMAPE（目标等权） |',
        '|---|---:|---:|---:|']
    for name,r in baselines.items():lines.append('| '+name+' | '+' | '.join(f'{r[k]:.2f}%' for k in keys)+' |')
    for name,r in summary.items():lines.append('| '+name+' | '+' | '.join(f'{r[k]["mean"]:.2f}% ± {r[k]["std"]:.2f}%' for k in keys)+' |')
    lines+=['','## 计算口径','',
        '- MAPE：逐点绝对百分比误差，先逐目标计算，再对4个目标等权平均。',
        '- 总体WMAPE：所有窗口、步长和目标的绝对误差总和 / 真值绝对值总和 ×100%；当前目标都是同类传感器占用率，合并口径按真实总量隐式加权。',
        '- 另列目标等权WMAPE，避免把总体合并与目标平均混为一谈。指标分别按seed计算后平均，不先平均预测。',
        '- 为处理逆标准化的数值残差，仅在MAPE中排除绝对真值不大于训练期固定epsilon的点，不对其余分母截断；WMAPE保留这些点的误差。',
        f'- 每个目标共有{coverage["observations_per_target"]:,}个窗口内预测元素，MAPE各目标排除数为{coverage["excluded_per_target"]}；窗口互有重叠。',
        f'- 原始单位epsilon：{epsilon.tolist()}。',
        '- MAPE对接近零的占用率很敏感，可能和MAE/WMAPE的排名不同，应结合原量纲误差阅读。','',
        '完整seed、逐目标、逐步长结果及预测文件校验和见 `percentage_metrics.json`。这是此前固定Traffic子集的小时预测实验，不是网约车供需测试。','']
    (out/'PERCENTAGE_METRICS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
