# Ride-hailing Forecasting

网约车供需多目标短窗口预测研究代码。独立项目，不依赖 TSLib，也不包含其完整模型库。

## 模型

| 训练名称 | 内容 | 输入 |
|---|---|---|
| `CEEMDAN` | CEEMDAN × Autoformer 周期聚合 × PatchTST 渐进式分解骨干 | 目标历史 |
| `Autoformer` | 原始 Autoformer 架构的项目适配，保留趋势/季节分解及 Encoder–Decoder | 目标与辅助历史 |
| `TimeXer` | 项目多目标扩展，选定目标共享外生上下文，选定目标不进入外生分支 | 目标与辅助历史 |
| `DLLTimeXer` | 第五章 Dynamic Lead–Lag TimeXer：Lag Tokens、逐目标/步长 Query、相关性偏置、上下文融合、残差门控 | 目标/辅助历史、未来已知特征、静态特征 |
| `PatchTST` | CEEMDAN 骨干依赖及 target-only 对照 | 目标历史 |

Autoformer、TimeXer 和 PatchTST 是已有开源方法，不将其原始架构声称为原创。本项目维护 CEEMDAN 融合、TimeXer 多目标适配、DLL-TimeXer 扩展及训练/评估工具；详见 [来源与许可](THIRD_PARTY_NOTICES.md)。

## 环境和运行

Python 3.11；当前验证环境记录在 `requirements.lock.txt`。CPU 即可运行功能测试。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
source .venv/bin/activate
python -m pytest -q
python scripts/make_demo.py
python train.py --data data/demo.npz --model DLLTimeXer --config configs/dll.json --output outputs/dll
```

用相同训练入口切换模型：

```bash
python train.py --data data/demo.npz --model Autoformer --output outputs/autoformer
python train.py --data data/demo.npz --model TimeXer --output outputs/timexer
python train.py --data data/demo.npz --model CEEMDAN --config configs/smoke_ceemdan.json --output outputs/ceemdan --epochs 1 --max-train-steps 2 --max-eval-steps 2
```

CEEMDAN 在每一编码层执行真实分解，CPU 成本高；最后一条是受限功能检查，不是完整训练。`--max-train-steps` / `--max-eval-steps` 默认 0，表示完整遍历。已有 checkpoint 的输出目录不能被新训练静默覆盖。

```bash
# 单独重载 best.pt 测试；结构和归一化统计从 checkpoint 恢复。
python train.py --data data/demo.npz --output outputs/dll --evaluate
# 四模型训练、保存和逐元素一致的重载验证。
python scripts/verify_training.py
```

每次输出验证集最优的 `best.pt`、训练历史、配置、原始量纲预测/真值及 MAE、RMSE、逐步长/逐目标误差。验证集用于模型选择，测试集仅在最终重载后评估。默认 70% / 15% / 15% 时间切分，标准化统计仅拟合训练期，标签不跨分区。

## 数据协议

按时间升序排列的 NPZ，数值型数组：

| 键 | 形状 | 含义 |
|---|---|---|
| `y` | `[T,M]` | M 个预测目标，必须有限 |
| `z` | `[T,C]`，可选 | 历史辅助变量；DLL 允许 NaN，其他基线要求完整 |
| `future` | `[T,H,F]`，可选 | **在每个预测发起时刻已经可得**的未来 H 步特征 |
| `static` | `[M,S]`，可选 | 每个目标的静态数值特征 |

窗口历史为 `y[o-L:o]`，预测 `y[o:o+H]`，读取 `future[o-1]`。例如历史最后时刻 t，`future[t,h]` 是时刻 t 已知的 t+h+1 特征。日历可提前构造，天气必须是当时发布的预报，不能放入未来真实天气或未来真实订单。程序可以保证切片边界，无法判断上游特征是否真实可得，这需要数据生产方保证。

第一版使用数值上下文编码器。类别字段请预先按训练期确定映射做 one-hot；当前未实现独立类别 Embedding，也未将 future/static 接入 Autoformer/TimeXer 基线。比较包含上下文的 DLL 与这些基线时应报告输入差异；消融可设置 `variant` 进行等输入比较。

## DLL-TimeXer

实现与第五章的映射见 [DLL 设计](docs/DLL_TIMEXER.md)。正式名称为 DLL-TimeXer；`DDLTimeXer` 是早期拼写兼容别名。

```python
import torch
from models.DLLTimeXer import Config, Model

model = Model(Config(seq_len=48, pred_len=5, n_targets=2, n_aux=3,
                     future_dim=2, static_dim=4))
y = torch.randn(8, 48, 2)
z = torch.randn(8, 48, 3)
future = torch.randn(8, 5, 2)
static = torch.randn(8, 2, 4)
forecast, details = model(y, z, future, static, return_details=True)
# forecast: [8,5,2]
# details['attention']: [8,2,5,3,8] = batch/target/horizon/variable/lag
```

`gamma=0` 时严格退化为 target-only 基础预测；默认 0.01 允许修正分支一开始获得梯度。相关性门控 `beta` 默认 0。完全缺失的辅助输入退化到基础分支，窗口内不完整的辅助 lag token 被屏蔽。注意力和相关性是预测关联，不代表因果关系。

无未来标签的单窗口推理：

```bash
python predict.py --checkpoint outputs/dll/best.pt --input data/request.npz --output outputs/prediction.npy
```

`request.npz` 使用 `y[L,M]`、可选 `z[L,C]`、`future[H,F]`、`static[M,S]`。维度必须与 checkpoint 一致；缩放参数和模型结构自动恢复。Python `predict.predict(..., details=True)` 可读取注意力。

## 验证范围

见 [验证记录](docs/VERIFICATION.md) 与机器可读的 `docs/verification.json`。测试和合成数据实验验证计算与工程流程，不宣称业务收益，不代表 DLL 已优于其他模型。尚需真实网约车数据、多随机种子、同输入预算对照，以及场景分组、延迟和外生变量错位/延迟到达实验。

## 真实数据初步效果测试

已完成 Traffic 固定16传感器子集的3随机种子对比，历史48小时预测未来1–5小时。DLL日历版平均MAE较TimeXer低约2.2%，但相同历史输入的DLL版本反而高约3.9%；尚不能确认lag/相关性模块的独立收益。设置、逐步误差及降级风险见 [完整报告](docs/traffic_effect_20261102/REPORT.md)。该结果不代表网约车数据或全862传感器基准。
