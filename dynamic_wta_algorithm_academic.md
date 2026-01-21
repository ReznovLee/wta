# 受约束掩码自回归 Transformer 强化学习用于动态 WTA：算法设计与训练流程

## 摘要

本文在“受时间约束的动态武器–目标分配与递进式多武器配置”模型基础上，给出一个前沿且确定的强化学习算法设计，用于直接解决动态 WTA。算法采用掩码自回归的双集 Transformer 策略与分解式评论家，严格执行硬约束（时间窗与可达性、武器步内并发、占用与冷却、弹药、容量与并发下界），并通过原始–对偶拉格朗日项处理必要的软冲突。训练流程为两阶段：离线受约束决策变压器预训练（CDT），随后在线受约束策略优化（CSPO）微调，配合 KL 正则与 PID–拉格朗日乘子更新。算法在集合–变规模场景下保持鲁棒，适配 50–300 目标规模的测试。

## 1. 问题设定与符号（与模型一致）

决策发生在离散时间 $t_k = k\,\Delta t$，$k=0,1,\dots,K$。武器集合为 $\mathcal{W}$，第 $k$ 步在场目标集合为 $\mathcal{T}_k$。每个目标 $i\in\mathcal{T}_k$ 具有类型 $\mathrm{type}_i\in\{\mathrm{BM},\mathrm{CM},\mathrm{AC}\}$、价值 $v_i>0$、以及 EKF 预测状态 $\widehat{s}_i(t;\mathrm{EKF})$。对每一对 $(w,i)$ 的可行拦截时间窗为 $\tau_{w,i,k}\in [t_k,\,t_k+H]$，拦截点 $p_{w,i,k}\in\mathbb{R}^3$ 必须满足可达性掩码 $M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\in\{0,1\}$。单武器杀伤概率为 $P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\in[0,1]$。武器具有单位发射成本 $c_w>0$、时间权重 $c_t\ge 0$、机动/几何成本 $c_{\mathrm{move}}(w,i,k)\ge 0$、弹药上限 $\mathrm{ammo}_w\in\mathbb{Z}_{\ge 0}$，以及冷却时间 $t_{\mathrm{ready}}(w)\ge 0$。初始并发偏好：$n_{\mathrm{init}}(\mathrm{BM})\in\{2,3\}$，$n_{\mathrm{init}}(\mathrm{CM})=n_{\mathrm{init}}(\mathrm{AC})=1$，并发上限 $n_{\max}\in\mathbb{Z}_{\ge 1}$。

第 $k$ 步的决策变量为二值指派 $x_{w,i,k}\in\{0,1\}$、目标并发 $m_{i,k}\in\mathbb{Z}_{\ge 0}$、拦截时间 $\tau_{w,i,k}$（及拦截点 $p_{w,i,k}$）。失败计数器 $z_{i,k}\in\mathbb{Z}_{\ge 0}$，武器占用标志 $e_{w,k}\in\{0,1\}$。占用由飞行与冷却驱动，定义准备完成步：$$ k^{\mathrm{ready}}_{w,i,k} = k + \left\lceil \frac{\tau_{w,i,k}-t_k}{\Delta t} \right\rceil + \left\lceil \frac{t_{\mathrm{ready}}(w)}{\Delta t} \right\rceil . $$并发递进下界：$$ n^{\min}_{i,k} = \min\{ n_{\max},\ n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k} \} . $$当前可用武器集合：$$ \mathcal{W}^{\mathrm{avail}}_k = \{ w\in\mathcal{W}\mid e_{w,k}=0,\ \text{且弹药未耗尽} \} . $$

成功概率聚合（独立假设）：$$ P^{\mathrm{succ}}_{i,k} = 1 - \prod_{w\in\mathcal{W}} \big( 1 - x_{w,i,k}\, P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k}) \big) . $$

## 2. MDP 定义与动作因子化

状态 $s_k$ 包含目标集、武器集与全局特征的确定性向量化：
- 目标特征（每个 $i\in\mathcal{T}_k$）：$[v_i,\ \mathrm{type}_i\ \text{嵌入},\ z_{i,k},\ n^{\min}_{i,k},\ \widehat{s}_i(t;\mathrm{EKF})\ \text{摘要},\ \text{时间窗与候选统计}]$。
- 武器特征（每个 $w\in\mathcal{W}$）：$[c_w,\ c_t,\ t_{\mathrm{ready}}(w),\ \mathrm{ammo}_w\ \text{剩余},\ e_{w,k},\ k^{\mathrm{ready}}\ \text{若占用},\ \text{平台状态摘要}]$。
- 全局特征：$[t_k,\ \Delta t,\ n_{\max},\ |\mathcal{W}^{\mathrm{avail}}_k|,\ \text{历史失败与资源使用摘要}]$。

动作 $a_k$ 由三个确定阶段组成，全部在硬掩码下解码：
1) 并发选择阶段：为每个目标选择 $m_{i,k}\in[n^{\min}_{i,k},\ n_{\max}]$，随后由预算器裁剪确保 $$ \sum_{i\in\mathcal{T}_k} m_{i,k} \le |\mathcal{W}^{\mathrm{avail}}_k| . $$
2) 指派阶段（自回归指针）：对每个目标 $i$ 按顺序选择 $m_{i,k}$ 个武器；每次选择在掩码下进行，掩码禁止步内同武器重复、禁止 $e_{w,k}=1$、禁止弹药耗尽武器。
3) 时间选择阶段：对每条选中边 $(w,i)$ 在离散集合 $\mathcal{T}_k^{\mathrm{disc}}=\{ t_k + b\,\Delta t\mid b=1,\dots,B \}\cap[t_k,\ t_k+H]$ 中选择 $\tau_{w,i,k}$；掩码确保 $M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1$，拦截点由 $$ p_{w,i,k} = \mathrm{SolveIntercept}(w,\ \widehat{s}_i(t;\mathrm{EKF}),\ \tau_{w,i,k}) $$ 得到。

占用更新规则：若 $x_{w,i,k}=1$，则在 $k'<k^{\mathrm{ready}}_{w,i,k}$ 期间武器不可用，$k\ge k^{\mathrm{ready}}_{w,i,k}$ 后自动恢复可用（若弹药未耗尽）。失败递进：若拦截失败，则在 $k'\approx \lceil \tau_{w,i,k}/\Delta t \rceil$ 插回目标并更新 $$ z_{i,k'} = z_{i,k} + 1,\quad n^{\min}_{i,k'} = \min\{ n_{\max},\ n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k'} \} . $$

## 3. 约束处理（硬掩码优先、软冲突拉格朗日）

硬约束在动作解码过程中逐步执行：
- 武器侧同步并发：同一步对同一武器的多边选择被掩码禁止，满足 $$ \sum_{i\in\mathcal{T}_k} x_{w,i,k} \le 1 . $$
- 目标侧并发等式：指派阶段自回归选择 $m_{i,k}$ 次保证 $$ \sum_{w\in\mathcal{W}} x_{w,i,k} = m_{i,k} . $$
- 时间窗与可达性：时间候选掩码确保 $$ x_{w,i,k}=1 \Rightarrow \tau_{w,i,k}\in[t_k,\ t_k+H],\ \ M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1 . $$
- 弹药上限：全局屏蔽 $\mathrm{ammo}_w=0$ 的武器，满足 $$ \sum_{k=0}^{K} \sum_{i\in\mathcal{T}_k} x_{w,i,k} \le \mathrm{ammo}_w . $$
- 占用与冷却：指派前屏蔽 $e_{w,k}=1$ 的武器，并在选择后更新 $k^{\mathrm{ready}}_{w,i,k}$，满足 $$ x_{w,i,k}=1 \Rightarrow \sum_{k'=k+1}^{\min\{k^{\mathrm{ready}}_{w,i,k},\ K\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} = 0 . $$
- 容量上限：在并发阶段由预算器裁剪，满足 $$ \sum_{i\in\mathcal{T}_k} m_{i,k} \le |\mathcal{W}^{\mathrm{avail}}_k| . $$

软约束仅针对并发下界与容量冲突的残差，用拉格朗日乘子 $\mu\ge 0$ 施加罚项并在线更新：$$ L_{\mathrm{soft}} = \mu \cdot \max\big(0,\ \sum_{i\in\mathcal{T}_k} n^{\min}_{i,k} - |\mathcal{W}^{\mathrm{avail}}_k| \big) . $$乘子采用 PID 更新，使平均违反率趋近 0。

## 4. 策略与评论家网络（确定结构）

底座网络采用双集–交叉注意力的集合 Transformer：
- 目标编码器：2–4 层 Transformer（前馈维度 $d_{\mathrm{ff}}$，注意力头数 $h$），嵌入 $[v_i,\ \mathrm{type}_i,\ z_{i,k},\ n^{\min}_{i,k},\ \text{EKF摘要},\ \text{候选统计}]$。
- 武器编码器：2–4 层 Transformer，嵌入 $[c_w,\ c_t,\ t_{\mathrm{ready}},\ \mathrm{ammo},\ e_{w,k},\ k^{\mathrm{ready}},\ \text{平台摘要}]$。
- 交叉注意力融合：目标为查询，武器为键/值，再反向一层，得到目标上下文 $h_i$ 与边级上下文 $h_{w,i}$。

解码头：
- 并发头：对每个目标输出 $m_{i,k}\in[n^{\min}_{i,k},\ n_{\max}]$ 的分类分布，随后由预算器裁剪确保总并发不超过 $|\mathcal{W}^{\mathrm{avail}}_k|$。
- 指派指针头：对每个目标自回归选择 $m_{i,k}$ 个武器，掩码严格禁止不可行动作（步内重复、占用、弹药耗尽）。
- 时间头：为已选边选择 $\tau_{w,i,k}\in\mathcal{T}_k^{\mathrm{disc}}$，掩码确保时间窗与可达性。

分解式评论家：共享底座输出全局价值 $V(s_k)$ 与边级优势分量 $A_{w,i,k}$，总体优势为 $$ A(s_k,a_k) = \sum_{i\in\mathcal{T}_k} A_i + \sum_{(w,i)\in X_k} A_{w,i,k} . $$

## 5. 训练目标与优化（确定流程）

主步奖励采用加权差式：$$ r_k = \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k} - \lambda \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\, ( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) ) - L_{\mathrm{soft}} . $$

训练流程分两阶段：
- 离线预训练（CDT）：将动作序列化为令牌流（先 $m_{i,k}$，再每个 $i$ 的指派序列与时间序列），以回报–到来（RTG）为条件进行 teacher forcing 训练；解码过程严格使用掩码，损失为掩码交叉熵与 RTG 配置的序列损失。
- 在线微调（CSPO）：在硬掩码环境中进行受约束策略优化（如 PPO/A2C 变体），目标为最大化期望累计奖励；加入 KL 正则到离线策略以抑制分布移位：$$ J_{\mathrm{KL}} = \beta_{\mathrm{KL}}\, \mathrm{KL}(\pi\,\|\,\pi_{\mathrm{offline}}) , $$并对拉格朗日乘子采用 PID 更新，确保软约束残差收敛。

优势估计采用 GAE，策略与价值网络共享底座但分头训练；优化器使用自适应学习率（如 AdamW），训练时对掩码后的合法动作进行损失计算，避免不可行动作影响梯度。

## 6. 数据集构建与生成（确定方案）

离线数据由以下确定流程生成：
1) 可行域随机（掩码安全）：在掩码合法集合中进行温度控制的均匀采样，生成广覆盖轨迹。
2) 多启发式族（带噪声）：按固定的价值–成本比排序、失败递进优先、弹药节约与占用敏感规则生成轨迹，并在掩码合法集内加入小幅噪声以增加多样性。
3) 小规模近最优注入：对小规模实例进行束搜索与局部交换改进，得到更高回报片段混入数据集。
4) 离线 RL 自举：用初版策略生成新轨迹，筛选高 RTG 样本并加入数据池。

所有轨迹严格满足硬约束，记录 $(s_k, a_k, r_k, \text{masks}, \text{RTG})$ 以供 CDT 训练；数据覆盖 50–300 目标的多规模场景，保证策略对变规模鲁棒。

## 7. 规模鲁棒性与复杂度控制（确定措施）

为适配 50–300 目标规模，采用以下确定措施：
- 集合式编码：不使用固定长度位置编码，改用类型嵌入与相对几何编码，确保对变规模鲁棒。
- 掩码稀疏化与 Top-$k$ 候选：对每个目标仅在 Top-$k$ 武器候选上计算注意力与指派（$k$ 由场景设定，如 $k\in[20,50]$），显著降低 $O(|\mathcal{T}_k|\cdot|\mathcal{W}|)$ 开销。
- 预算器：并发阶段采用确定的容量裁剪策略（优先高价值/高失败计数）。
- 课程学习：训练时混合多规模场景，逐步提升大规模权重。

## 8. 实现要点与接口（确定规范）

环境接口（src/enviroment/wta_env.py）：
- get_action_masks(s_k)：返回并发范围、容量预算、武器步内可用掩码、占用与弹药掩码、时间候选掩码（基于 $H$ 与 $M$）。
- solve_intercept(w, \widehat{s}_i, \tau)：返回 $p$ 与可达性确认；失败则屏蔽该候选。
- 更新占用与冷却：当 $x_{w,i,k}=1$ 时设置 $e_{w,k'}=1$ 直至 $k'\ge k^{\mathrm{ready}}_{w,i,k}$；弹药递减与恢复逻辑与模型一致。

模型接口（src/models/transformer/）：
- 目标/武器编码器、交叉注意力层、并发头、指派指针头、时间头、分解式评论家；所有头在掩码下进行前向与损失计算。

训练脚本（scripts/train.py）：
- 实现 CDT 预训练与 CSPO 微调的确定管线；包含 KL 正则、PID–拉格朗日乘子更新、评估与日志。

## 9. 评估指标与验证（确定设置）

评估覆盖 50、100、150、200、250、300 目标规模，报告：
- 价值加权成功总和（与 $J$ 一致）、平均并发满足率（目标侧等式）、约束违反率（应接近 0）。
- 弹药使用效率、剩余率与早期消耗率；失败计数 $z$ 的下降速度；对高价值/弹道目标的优先成功率。
- 在线稳定性：平均 KL、软约束残差与乘子收敛情况。

## 10. 总结

本文提出的受约束掩码自回归 Transformer 强化学习（WTA–ConTra）在严格硬掩码与原始–对偶拉格朗日框架下运行，动作因子化与占用–冷却更新与模型完全一致。两阶段训练（CDT 预训练 + CSPO 微调）在保证约束一致性的同时提升策略质量；集合式编码与规模控制措施确保策略对 50–300 目标的鲁棒性。该算法为动态 WTA 提供一个前沿且工程可落地的强化学习方案。