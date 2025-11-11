# 受时间约束的动态武器–目标分配与递进式多武器配置：优化模型与强化学习嵌入

## 摘要

我们研究一个动态、受时间限制的武器–目标分配（WTA）问题：首波拦截偏好要求对非弹道类目标采用单武器拦截，而对弹道导弹采用多武器拦截；若拦截失败，则在后续决策步中递进式提高并行武器数量。环境施加时间窗与运动学可达性约束；武器在飞行与冷却期间不可用，且弹药有限。在该可行域内，学习型策略选择并发水平、二值指派以及拦截时间，以最大化价值–成本的权衡。我们提出一个严格的优化建模，统一了偏好规则、占用与容量约束以及成功概率聚合；同时给出一个受约束强化学习（RL）嵌入，它在硬掩码动作空间中工作，并通过拉格朗日罚处理软约束。

## 问题设定与记号

决策发生在离散时间 $t_k = k\,\Delta t$，$k=0,1,\dots,K$。记 $\mathcal{W}$ 为武器集合，$\mathcal{T}_k$ 为第 $k$ 步的在场目标集合。每个目标 $i\in\mathcal{T}_k$ 具有类型 $\mathrm{type}_i\in\{\mathrm{BM},\mathrm{CM},\mathrm{AC}\}$（弹道导弹、巡航导弹、飞机），价值 $v_i>0$，以及基于 EKF 的预测状态 $\widehat{s}_i(t;\mathrm{EKF})$。对第 $k$ 步的每一对 $(w,i)$，可行拦截时间窗为 $\tau_{w,i,k}\in [t_k,\,t_k+H]$，拦截点 $p_{w,i,k}\in\mathbb{R}^3$ 必须满足可达性掩码 $M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\in\{0,1\}$。单武器杀伤概率估计记为 $P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\in[0,1]$。武器具有单位发射成本 $c_w>0$、时间权重 $c_t\ge 0$、机动/几何成本 $c_{\mathrm{move}}(w,i,k)\ge 0$、弹药上限 $\mathrm{ammo}_w\in\mathbb{Z}_{\ge 0}$，以及冷却时间 $t_{\mathrm{ready}}(w)\ge 0$。初始并发偏好为弹道导弹 $n_{\mathrm{init}}(\mathrm{BM})\in\{2,3\}$，其他目标 $n_{\mathrm{init}}(\mathrm{CM})=n_{\mathrm{init}}(\mathrm{AC})=1$，并有全局并发上限 $n_{\max}\in\mathbb{Z}_{\ge 1}$。

第 $k$ 步的决策变量为二值指派 $x_{w,i,k}\in\{0,1\}$、每个目标的并发水平 $m_{i,k}\in\mathbb{Z}_{\ge 0}$，以及拦截时间 $\tau_{w,i,k}$（及其对应拦截点 $p_{w,i,k}$）。环境维护每个目标的失败计数器 $z_{i,k}\in\mathbb{Z}_{\ge 0}$，以及每个武器的占用标志 $e_{w,k}\in\{0,1\}$（$e_{w,k}=1$ 表示不可用）。占用由飞行时间与冷却驱动；定义离散占用长度：$$ L_{w,i,k} \triangleq \left\lceil \frac{\tau_{w,i,k}-t_k}{\Delta t} \right\rceil + \left\lceil \frac{t_{\mathrm{ready}}(w)}{\Delta t} \right\rceil . $$

递进式并发规则由下界给出：$$ m_{i,k} \ge n^{\min}_{i,k} \triangleq \min\big\{\, n_{\max},\; n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k} \,\big\} , $$它对非弹道目标强制首波单武器拦截、对弹道导弹强制多武器拦截；若拦截失败，在下一决策步 $z$ 增加并提高最小并发。当前可用武器集合为：$$ \mathcal{W}^{\mathrm{avail}}_k \triangleq \big\{ w\in\mathcal{W}\;\big|\; e_{w,k}=0,\ \text{且弹药未耗尽} \big\} . $$

## 优化模型

采用加权差形式的价值–成本目标，并在独立假设下聚合成功概率。整体规划为：$$ \max\; J \;=\; \sum_{k=0}^{K} \left[ \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k} \; - \; \lambda \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\, \big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \right] ,\quad \lambda>0, $$

满足武器侧的同步并发约束：$$ \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; 1,\quad \forall w\in\mathcal{W},\ \forall k, $$以及目标侧约束：$$ \sum_{w\in\mathcal{W}} x_{w,i,k} \;=\; m_{i,k},\qquad 0\le m_{i,k}\le n_{\max},\quad m_{i,k}\in\mathbb{Z}_{\ge 0},\quad \forall i\in\mathcal{T}_k,\ \forall k, $$并发的递进式下界：$$ m_{i,k} \;\ge\; n^{\min}_{i,k} ,\quad \forall i,k. $$

时间限制与可达性要求：$$ x_{w,i,k}=1 \Rightarrow \tau_{w,i,k}\in[t_k,\,t_k+H],\qquad x_{w,i,k}=1 \Rightarrow M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1, $$弹药上限：$$ \sum_{k=0}^{K} \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; \mathrm{ammo}_w,\quad \forall w\in\mathcal{W}, $$占用约束（飞行+冷却期间不可重用，冷却结束后恢复可用）：$$ k^{\mathrm{ready}}_{w,i,k} \triangleq k + \left\lceil \frac{\tau_{w,i,k}-t_k}{\Delta t} \right\rceil + \left\lceil \frac{t_{\mathrm{ready}}(w)}{\Delta t} \right\rceil, $$ $$ x_{w,i,k}=1 \Rightarrow \sum_{k'=k+1}^{\,\min\{k^{\mathrm{ready}}_{w,i,k},\,K\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} = 0. $$

为保证每步的可行性，目标请求的总并发不能超过可用武器数量：$$ \sum_{i\in\mathcal{T}_k} m_{i,k} \;\le\; \big|\mathcal{W}^{\mathrm{avail}}_k\big|,\qquad \forall k. $$在独立假设下的“至少一次成功”概率为：$$ P^{\mathrm{succ}}_{i,k} \;=\; 1 - \prod_{w\in\mathcal{W}} \Big( 1 - x_{w,i,k}\, P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k}) \Big) . $$拦截几何与 EKF 的耦合：$$ p_{w,i,k} \;=\; \mathrm{SolveIntercept}\big( w,\ \widehat{s}_i(t;\mathrm{EKF}),\ \tau_{w,i,k} \big) ,\qquad M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1. $$

在拦截时间判定结果后，若失败，则将目标在下一决策步 $k'\approx \lceil \tau/\Delta t \rceil$ 重新插入，并更新计数器与最小并发下界：$$ z_{i,k'} \;=\; z_{i,k} + 1,\qquad n^{\min}_{i,k'} = \min\{ n_{\max},\ n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k'} \} . $$而成功则将 $i$ 从后续集合 $\mathcal{T}_\cdot$ 中移除。为奖励塑形或消融，也可使用比值型目标：$$ J_{\mathrm{ratio}} \;=\; \sum_{k=0}^{K} \frac{ \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k} }{ \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\,\big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) + \epsilon },\quad \epsilon>0. $$

## 约束的作用与解释

- 武器侧同步并发约束（同一步仅能指派一次）：$$ \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; 1,\quad \forall w,\forall k. $$
  作用：保证每个武器在同一决策步不被同时分配给多个目标，符合“单次发射/单资源占用”的工程现实。与占用约束共同作用，前者管控“同一步并发”，后者扩展到“跨步占用期”。实现要点：在动作空间解码前，对同一 $w$ 的多条边进行硬掩码，防止重复分配。

- 目标侧并发约束（精确并发数量）：$$ \sum_{w\in\mathcal{W}} x_{w,i,k} \;=\; m_{i,k},\quad 0\le m_{i,k}\le n_{\max}. $$
  作用：确保目标 $i$ 在第 $k$ 步恰好获得 $m_{i,k}$ 个武器，用于实现对弹道与非弹道目标的拦截强度控制。等式“=”体现“需求即供给”的并发语义；若需允许“至多”并发，可改为“\le”，并在奖励或拉格朗日项中惩罚未满足的并发需求。

- 并发递进下界（失败后提高并发）：$$ m_{i,k} \;\ge\; n^{\min}_{i,k} \triangleq \min\{ n_{\max},\ n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k} \}. $$
  作用：编码首波拦截偏好与“失败增配”策略，避免对高威胁目标的低强度反复尝试。与容量约束存在冲突时（总下界超出可用武器数），需采用优先级调度、剪裁并发或拉格朗日软化以维持系统可行性。

- 时间窗与可达性（物理可行性过滤）：$$ x_{w,i,k}=1 \Rightarrow \tau_{w,i,k}\in[t_k,\,t_k+H],\quad x_{w,i,k}=1 \Rightarrow M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1. $$
  作用：保证拦截动作发生在可规划的时间范围内，且拦截几何（由 EKF 预测与解算器确定）在运动学上可达。实现要点：先用时间离散化产生候选 $\tau$，再用可达性掩码 $M$ 对 $(w,i,\tau)$ 边进行硬过滤，防止策略选择不可达动作。

- 弹药上限（资源消耗约束）：$$ \sum_{k=0}^{K} \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; \mathrm{ammo}_w,\quad \forall w. $$
  作用：限制武器总发射次数不超过其弹药容量。该约束在长期规划中防止“早期过度消耗”导致后期无弹可用；在 RL 中可作为硬掩码（当弹药耗尽时移除相关边），也可作为稀疏惩罚项以鼓励节约。

- 占用约束（飞行+冷却期间不可重用，冷却结束后恢复可用）：$$ k^{\mathrm{ready}}_{w,i,k} \triangleq k + \left\lceil \frac{\tau_{w,i,k}-t_k}{\Delta t} \right\rceil + \left\lceil \frac{t_{\mathrm{ready}}(w)}{\Delta t} \right\rceil, \quad x_{w,i,k}=1 \Rightarrow \sum_{k'=k+1}^{\min\{k^{\mathrm{ready}}_{w,i,k},\,K\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} = 0. $$
  作用：在飞行与冷却期间禁止该武器再次被分配，冷却结束后武器自动恢复可用（若弹药未耗尽），避免“永久停用”的不合理情形。$k^{\mathrm{ready}}_{w,i,k}$ 由拦截时间与冷却时长决定，可在早期基线中近似为常数以降低计算复杂度。实现要点：维护每武器的“可用至时间戳”，在动作解码时对跨步边进行掩码，并在 $k\ge k^{\mathrm{ready}}_{w,i,k}$ 时将武器重新加入可用集合。

- 容量约束（步内总并发不超可用资源）：$$ \sum_{i\in\mathcal{T}_k} m_{i,k} \;\le\; \big|\mathcal{W}^{\mathrm{avail}}_k\big|. $$
  作用：确保在第 $k$ 步请求的总并发不超过当前可用武器数量，从而保证存在可行的指派矩阵（如匈牙利或可微指派层能找到匹配）。当该约束与并发下界冲突时，应进行需求裁剪或采用拉格朗日策略以平衡可行性与效能。

约束之间的协同与优先级：武器侧同步并发与占用约束共同保证“时域独占”；目标侧并发与递进下界共同编码“任务强度”；时间窗与可达性提供“物理可行性过滤”；弹药与容量约束提供“资源边界”。在工程实现中，建议以“硬掩码优先、软惩罚兜底”的策略，先剔除显然不可行的动作，再用拉格朗日或奖励项平衡容量–效能之间的权衡。

## 受约束强化学习嵌入

在上述可行域内，策略 $\pi_\theta$ 选择 $ (m_{i,k})_{i\in\mathcal{T}_k} $、指派矩阵 $x_{w,i,k}$ 以及拦截时间 $\tau_{w,i,k}$。实践中，环境会基于时间窗、可达性、占用与弹药提供硬掩码，将无效的 $(w,i)$ 边在动作解码前剪除。偏好规则通过并发下界 $m_{i,k}\ge n^{\min}_{i,k}$ 固化；并发上界 $m_{i,k}\le n_{\max}$ 与容量约束 $\sum_i m_{i,k}\le |\mathcal{W}^{\mathrm{avail}}_k|$ 共同维持可行性。

用于训练的加权差式步奖励可写为：$$ r_k \;=\; \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k} \; - \; \lambda \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\, \big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \; - \; \beta_{\mathrm{infeas}}\, \mathbb{I}\{\text{violations}\} $$或采用比值变体；其中 $\beta_{\mathrm{infeas}}$ 在软约束方案（如原始–对偶拉格朗日）中惩罚残余不可行性。结构化动作解码——可采用两阶段（先选 $m$ 后指派与时序）或可微指派层——可与序列模型（如决策变压器）和离线 RL（如隐式 Q 学习）结合，以利用历史并稳定训练。部署时，环境依据结果更新 $z_{i,k}$、占用队列与目标集合，确保优化约束与 RL 执行一致。

## 实用离散化与实现

为降低计算负担，可在时间窗内对拦截时间离散化为 $\tau_{w,i,k}\in\mathcal{T}_k^{\mathrm{disc}}=\{\, t_k + b\,\Delta t\mid b=1,\dots,B\,\}\cap [t_k,\,t_k+H]$，并在早期基线中将占用近似为武器特定常数长度 $L_w^{\mathrm{const}}$。这些离散化在缩小动作空间的同时保持了时间限制与占用语义，且与上文的连续时间完整模型保持一致。本节的全部符号与约束与完整优化程序中的定义完全一致，便于从基线实现向完整模型平滑过渡。

## 总结

该建模紧密整合了首波拦截偏好、失败后的递进式并发、时间窗、运动学可达性、占用与弹药约束以及全局容量可行性。在独立假设下聚合成功概率；目标函数变体平衡价值与资源成本。受约束 RL 嵌入在硬掩码的可行域内运行，并可结合拉格朗日罚处理残余软约束。该统一模型既适用于仿真驱动的学习，也适用于通过时间分桶的混合整数近似，并与基于 EKF 的拦截几何和现有环境接口保持一致。