# DLL-TimeXer：第五章实现映射

依据用户提供的《基于 Transformer 框架的网约车供需预测模型演进之路》第五章（2026-11-02读取）。本文只保存必要的工程规格，不上传原始 Obsidian 文档或附件。

1. **目标分支**：每个目标独立 Patching，共享 Transformer Encoder，在 Patch 后附加可学习 Global Token。基础 Flatten Head 仅依赖目标历史，输出 H 步。目标内实例归一化，最终回到输入量纲。
2. **Lag Tokens**：对每个辅助变量 j、每个非负 lag l，抽取 `[t-l-Wp+1:t-l]`。线性投影加变量和 lag embedding。默认 lags 为 `[0,1,2,3,4,6,9,12]`，窗口8，必须在历史内，不向未来 padding。
3. **Horizon Queries**：目标 Global Token、horizon embedding、未来已知数值特征编码、目标静态特征编码拼接，映射成每个 (m,h) 的 Query。Cross-Attention 权重显式保留 `[B,M,H,C,K]`。
4. **Dynamic Correlation Bias**：历史内计算 `Corr(diff(y)[s], diff(z)[s-l])`，正 lag 表示辅助领先。只使用历史、去中心的变化量；常数/有效对不足时置零。MLP 输入 `[r,abs(r),l/(L-1),h/H]`，分数为内容点积除 sqrt(d) 加 `beta*MLP`。beta 可训练，初始0。
5. **Context-aware Fusion**：第一版仅通过 Query 融合未来与静态特征，不额外调制 bias MLP，与5.5的首版建议一致。数值编码器支持 one-hot 类别；不存在默认类别语义。
6. **Residual Forecast**：辅助 attention value 聚合经线性头预测修正量，最终 `base + gamma*residual`，gamma 可训练，默认0.01，也可初始化0。
7. **缺失与降级**：辅助窗口有缺失则屏蔽该 token；相关性忽略无效对。全辅助缺失时权重及修正量为0，不产生 NaN。无辅助变量也可独立输出 base。
8. **空间先验**：forward 的 `spatial_bias[M,C]` 或 `[B,M,C]` 参数作为可选加法分数先验广播到各 horizon/lag；不自动构图、不据未来数据估计。训练CLI不自动加载空间先验，需通过 Python 接口传入已验证的数据。

## 消融

`Config.variant` / JSON `variant`：

| 版本 | 路径 |
|---|---|
| 0 | 整条辅助历史一个 variate token；共享 horizon Query，逐步输出由目标 base head 区分 |
| 1 | Lag Tokens；horizon Query仍共享 |
| 2 | 加 horizon embedding |
| 3 | 加动态相关偏置 |
| 4 | 加未来/静态上下文 |

版本0是明确实现的普通外生融合对照，不声称与官方 TimeXer 逐层结构完全一致。独立 `TimeXer` 模型提供项目原始扩展基线。Target-only 对照可令 n_aux=0 或 gamma=0。空间先验是在版本4上的可选Python扩展，未宣称完成真实路网实验。

## 计算成本和解释边界

辅助 token 数 C×K，交叉注意力规模约 B×M×H×C×K。当前未使用稀疏或近似 attention。CEEMDAN 与 DLL 是不同实验模型，DLL 默认不叠加 CEEMDAN。数据关系、lag权重和注意力不应被解读为因果传播。
