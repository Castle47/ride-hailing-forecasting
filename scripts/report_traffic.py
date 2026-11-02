"""Render a report from benchmark_traffic.py results without rerunning training."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/traffic_effect_20261102')
    p.add_argument('--destination',default='docs/traffic_effect_20261102');args=p.parse_args()
    source=Path(args.results);out=Path(args.destination);out.mkdir(parents=True,exist_ok=True)
    data=json.loads((source/'summary.json').read_text());runs=json.loads((source/'runs.json').read_text())
    summaries=data['summary'];baselines=data['baselines'];protocol=data['protocol']
    names=list(baselines)+list(summaries)
    means=[baselines[n]['mae'] if n in baselines else summaries[n]['mae_mean'] for n in names]
    errors=[0 if n in baselines else summaries[n]['mae_std'] for n in names]
    plt.rcParams.update({'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    colors=['#94a3b8','#cbd5e1','#64748b','#2563eb','#0d9488','#ea580c']
    x=np.arange(len(names));axs[0].bar(x,means,yerr=errors,color=colors,capsize=4)
    axs[0].set_xticks(x,names,rotation=30,ha='right');axs[0].set_ylabel('MAE (original occupancy units)')
    axs[0].set_title('Traffic subset: 3 seeds, mean ± sample SD')
    for name in names:
        values=np.array(baselines[name]['mae_per_horizon_target']).mean(-1) if name in baselines else summaries[name]['mae_per_horizon']
        axs[1].plot(range(1,6),values,marker='o',label=name,color=colors[names.index(name)])
    axs[1].set_xticks(range(1,6));axs[1].set_xlabel('Forecast horizon (hours)');axs[1].set_ylabel('MAE')
    axs[1].set_title('Error by forecast horizon');axs[1].legend(fontsize=8)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    table=['| 模型 | MAE（均值 ± SD） | RMSE（均值 ± SD） | 参数量 | 每次完整运行秒数 |',
           '|---|---:|---:|---:|---:|']
    for name in names:
        if name in baselines:
            r=baselines[name];table.append(f'| {name} | {r["mae"]:.6f} | {r["rmse"]:.6f} | 0 | — |')
        else:
            r=summaries[name];table.append(f'| {name} | {r["mae_mean"]:.6f} ± {r["mae_std"]:.6f} | {r["rmse_mean"]:.6f} ± {r["rmse_std"]:.6f} | {r["parameters"]:,} | {r["seconds_mean"]:.1f} |')
    history=summaries['DLL_V3_history'];full=summaries['DLL_V4_calendar'];tx=summaries['TimeXer'];pt=summaries['PatchTST']
    def change(a,b):return (a/b-1)*100
    htable=['| 模型 | 1小时 | 2小时 | 3小时 | 4小时 | 5小时 |','|---|---:|---:|---:|---:|---:|']
    for name in names:
        values=np.array(baselines[name]['mae_per_horizon_target']).mean(-1) if name in baselines else summaries[name]['mae_per_horizon']
        htable.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    diagnostics_path=source/'diagnostics.json'
    diagnostics=json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else []
    lines=['# DLL-TimeXer 初步真实数据效果测试','',
           '日期：2026-11-02。本次不使用合成数据评价效果。代码基线为独立仓库初版 `8ae0bd4`，模型实现未按测试集结果调参。','',
           '## 协议','',
           f'- 数据：本地已有 Traffic（{protocol["rows"]:,} 行、原始862个传感器的小时数据）；固定选0–3为4个目标，4–15为12个辅助变量。不是全862传感器基准，也不是网约车数据。',
           '- 划分：按时间70%训练、15%验证、15%测试；标准化只拟合训练段。所有模型预测完全相同的测试窗口，标签不跨分区。此划分与部分论文采用的70/10/20不同，不直接比较论文数字。',
           f'- 任务：历史48小时 → 未来1–5小时，共{data["test_windows"]:,}个测试窗口。误差使用原始占用率量纲，跨目标/步长均匀平均。',
           f'- 训练：每个模型完整训练{protocol["epochs"]}轮；随机种子{protocol["seeds"]}；batch=64，Adam，lr=0.001，CPU单线程。每轮完整验证，按验证MSE选最佳checkpoint，最终评估完整测试集。没有使用max-steps截断。',
           '- 宽度16、2个heads、1层、dropout=0.1、patch=4。PatchTST stride=2，其余patch不重叠。DLL有8个lag，每个辅助窗口8步，相关性窗口16步。参数量与计算量不完全相等。',
           '- DLL_V3_history与TimeXer使用相同目标/辅助历史，不加未来/静态上下文；DLL_V4_calendar额外使用预测时刻已知的未来小时/星期sin-cos特征。没有使用未来实测传感器值或静态区域属性。',
           '- Persistence重复最后一个观测值；DailySeasonal重复24小时前对应位置，所有基线输入均在历史窗口内。','',
           '## 结果','',*table,'',
           '均值/SD跨3个随机种子计算；每次运行秒数包含训练、验证与最终测试，不是单样本推理时延。不同seed的预测误差有相关性，3个seed不能作为充分显著性检验。','',
           '![误差对比](comparison.png)','',
           '## 数值比较','',
           f'- DLL历史版相对TimeXer：MAE变化 {change(history["mae_mean"],tx["mae_mean"]):+.2f}%，RMSE变化 {change(history["rmse_mean"],tx["rmse_mean"]):+.2f}%。负值表示误差降低。',
           f'- DLL日历版相对TimeXer：MAE变化 {change(full["mae_mean"],tx["mae_mean"]):+.2f}%，RMSE变化 {change(full["rmse_mean"],tx["rmse_mean"]):+.2f}%。',
           f'- DLL日历版相对PatchTST：MAE变化 {change(full["mae_mean"],pt["mae_mean"]):+.2f}%，RMSE变化 {change(full["rmse_mean"],pt["rmse_mean"]):+.2f}%。',
           f'- DLL日历版相对DLL历史版：MAE变化 {change(full["mae_mean"],history["mae_mean"]):+.2f}%。两者同时改变上下文开关和输入，不能将差异解释为某一个模块的单独贡献。','',
           '## 按步长MAE','',*htable,'',
           '## 修正分支诊断','',
           '日历版3个seed的gamma从初始0.01训练到约0.157–0.164，辅助修正分支确实参与输出。历史版和日历版的相关性beta也有更新，但这不能单独证明该模块改善了泛化。',
           '对日历版已训练checkpoint把修正量移除后，基础分支MAE分别约0.02663、0.03640、0.05544，均高于完整模型。因此“可以退化到基础分支”目前是功能保证，不是缺失外生输入时的准确率保证。共同训练后基础分支可能与辅助分支相互补偿；这不是重新训练的target-only消融。后续可在验证集上研究基础分支辅助监督/缺失训练，不能用测试结果直接调参。',
           '完整诊断数值见 `diagnostics.json`。本次不修改模型以追赶测试集指标。','',
           '## 解释边界','',
           '本轮回答“模型能否训练、在固定真实数据任务上的初步效果如何”，不证明业务收益或因果关系。固定前16个传感器没有保证地理关系或最大可预测相关；不依据测试结果重新挑选传感器。相邻预测窗口重叠，不把每个窗口当独立样本做显著性结论。','',
           '尚需真实网约车数据、更多传感器子集、验证集超参数搜索、V0–V4逐模块消融、不同训练预算和高峰/异常场景评估。不能仅凭本轮结果断言DLL在其他任务上优于或劣于TimeXer。','',
           '## 复现','',
           '```bash',
           'python scripts/benchmark_traffic.py --csv /path/to/traffic.csv --output outputs/traffic_effect_20261102',
           'python scripts/report_traffic.py --results outputs/traffic_effect_20261102',
           '```','',
           '源码和精简报告提交仓库；数据、权重和完整日志保留本地。`summary.json`、`protocol.json` 和 `runs.json` 包含实际记录，可追溯每个seed及验证选中的epoch。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    for name in ['summary.json','protocol.json','runs.json']:shutil.copy2(source/name,out/name)
    if diagnostics_path.exists(): shutil.copy2(diagnostics_path,out/'diagnostics.json')
    print(out/'REPORT.md')

if __name__=='__main__':main()
