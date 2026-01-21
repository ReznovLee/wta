# 动态时间受限武器目标分配（WTA）优化模型与强化学习框架（含逐次增配与弹道导弹首轮多武器）

本文件给出满足以下运行偏好的可优化模型与强化学习嵌入接口，并用 LaTeX 公式表达：
- 首轮对一般目标采用单武器；若失败，下一次决策逐次增加并发武器数（2、3、…）。
- 对弹道导弹（BM）首轮直接采用 2 或 3 枚并发武器。
- 时间受限（time restricted）与几何可达性约束，指派后武器在拦截与冷却期间不再可用。
- 忽略编队/区域约束（可留作后续拓展）。
- 由强化学习在可行域内自动权衡效费比，学习何时增配及增配到何种水平。

---

## 1. 集合、索引与时间

- 离散决策时间：
  $$
  t_k = k\,\Delta t,\quad k=0,1,\dots,K.
  $$
- 武器集合：$\mathcal{W}$；时间步 $k$ 的在场目标集合：$\mathcal{T}_k$。
- 目标类型：$\mathrm{type}_i \in \{\mathrm{BM},\ \mathrm{CM},\ \mathrm{AC}\}$（弹道导弹、巡航导弹、飞机）。

## 2. 参数（由环境与 EKF/拦截几何提供或配置）

- 目标价值（威胁效应）：$v_i>0$。
- 武器发射成本：$c_w>0$；时间成本权重：$c_t\ge 0$；机动/姿态成本估计：$c_{\mathrm{move}}(w,i,k)\ge 0$。
- 时间受限规划视界：$H>0$，拦截时间窗：
  $$
  \tau_{w,i,k}\in [t_k,\ t_k+H].
  $$
- 可达性遮罩（几何与性能约束）：$M_{w,i,k}(\tau,p)\in\{0,1\}$。
- 单武器拦截成功概率：$P^{\mathrm{kill}}_{w,i,k}(\tau,p)\in[0,1]$。
- 弹药上限：$\mathrm{ammo}_w\in\mathbb{Z}_{\ge 0}$；准备/冷却时间：$t_{\mathrm{ready}}(w)\ge 0$。
- 初始并发武器偏好：
  $$
  n_{\mathrm{init}}(\mathrm{BM})\in\{2,3\},\quad n_{\mathrm{init}}(\mathrm{CM})=1,\quad n_{\mathrm{init}}(\mathrm{AC})=1.
  $$
- 并发武器上限：$n_{\max}\in\mathbb{Z}_{\ge 1}$。

## 3. 变量（决策与状态）

- 指派二元变量：
  $$
  x_{w,i,k}\in\{0,1\}\quad\text{表示步 }k\text{ 将武器 }w\text{ 指派到目标 }i.
  $$
- 并发武器数（由策略选择）：
  $$
  m_{i,k}\in\mathbb{Z}_{\ge 0}\quad\text{表示步 }k\text{ 对目标 }i\text{ 的并发武器数量}.
  $$
- 逐次增配标记（累计失败次数）：
  $$
  z_{i,k}\in\mathbb{Z}_{\ge 0}\quad\text{表示目标 }i\text{ 截至步 }k\text{ 的累计拦截失败次数}.
  $$
- 预计拦截时间与拦截点：$\tau_{w,i,k}\ge 0$, $\ p_{w,i,k}\in\mathbb{R}^3$。
- 武器占用状态（由占用队列派生）：$e_{w,k}\in\{0,1\}$ 表示步 $k$ 时武器是否处于占用状态（$e_{w,k}=1$ 为占用，不可用；$e_{w,k}=0$ 为可用）。

## 4. 并发与逐次增配规则（关键机制）

- 对任意目标 $i$ 在步 $k$，并发武器数与指派满足：
  $$
  \sum_{w\in\mathcal{W}} x_{w,i,k}\;=\; m_{i,k}.
  $$
- 最小并发武器数由“初始偏好 + 累计失败次数”决定：
  $$
  m_{i,k}\;\ge\; n^{\min}_{i,k}\;\triangleq\;\min\Big\{\,n_{\max},\;n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k}\,\Big\}.
  $$
  说明：
  - 非弹道目标首轮（$z_{i,k}=0$）至少 $m_{i,k}\ge 1$，失败后 $z$ 增加促使下一次至少 $2,3,\dots$。
  - 弹道目标首轮即 $m_{i,k}\ge 2$ 或 $3$（由 $n_{\mathrm{init}}(\mathrm{BM})$ 决定），失败后继续增配。
- 强化学习的自由度：策略可在 $m_{i,k}\in\{n^{\min}_{i,k},\dots,n_{\max}\}$ 内优化选择，并同时优化 $\tau_{w,i,k}$。

## 5. 时间受限与可达性

- 时间窗约束：若 $x_{w,i,k}=1$，则
  $$
  \tau_{w,i,k}\in[t_k,\ t_k+H].
  $$
- 可达性约束：
  $$
  x_{w,i,k}=1\ \Rightarrow\ M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1.
  $$
- 指派后占用与“从可用集合删除”：若在步 $k$ 指派 $w\to i$，则武器在拦截发生与冷却完成前不能重复指派。令
  $$
  L_{w,i,k}\;\triangleq\;\left\lceil\frac{\tau_{w,i,k}-t_k}{\Delta t}\right\rceil + \left\lceil\frac{t_{\mathrm{ready}}(w)}{\Delta t}\right\rceil,
  $$
  则有（近似离散化表达）
  $$
  x_{w,i,k} + \sum_{k'=k+1}^{\,\min\{k+L_{w,i,k},\,K\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} \;\le\; 1,\quad \forall w.
  $$

## 6. 同步并发约束（武器侧与目标侧）

- 每步每武器仅拦截一个目标：
  $$
  \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; 1,\quad \forall w,\;k.
  $$
- 每步每目标的并发数量受 $m_{i,k}$ 控制：
  $$
  \sum_{w\in\mathcal{W}} x_{w,i,k} \;=\; m_{i,k},\quad 0\le m_{i,k}\le n_{\max},\quad \forall i\in\mathcal{T}_k.
  $$
- 弹药约束：
  $$
  \sum_{k=0}^{K} \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; \mathrm{ammo}_w,\quad \forall w.
  $$

## 7. 并发成功概率的合成（独立假设）

- 目标 $i$ 在步 $k$ 的总体成功概率（至少一枚成功）：
  $$
  P^{\mathrm{succ}}_{i,k}\;=\;1\; -\; \prod_{w\in\mathcal{W}}\Big(1 - x_{w,i,k}\cdot P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\Big).
  $$

## 8. 效费比目标（两种常用形式）

### 8.1 直接比值最大化（逐步或整体）

$$
\max\;\;\sum_{k=0}^{K}\;\frac{\displaystyle\sum_{i\in\mathcal{T}_k} v_i\cdot P^{\mathrm{succ}}_{i,k}}{\displaystyle\sum_{w\in\mathcal{W}}\sum_{i\in\mathcal{T}_k} x_{w,i,k}\cdot\big(c_w + c_t\cdot \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k)\big)\; +\; \epsilon},\quad \epsilon>0.
$$

该形式直接体现“效费比”，适合作为强化学习的逐步奖励，但在整数规划中较难线性化。

### 8.2 加权差形式（便于优化与近似比值）

$$
\max\;\;\sum_{k=0}^{K}\left[\sum_{i\in\mathcal{T}_k} v_i\cdot P^{\mathrm{succ}}_{i,k}\; -\; \lambda\cdot \sum_{w,i} x_{w,i,k}\cdot\big(c_w + c_t\cdot \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k)\big)\right],\quad \lambda>0.
$$

$\lambda$ 可通过线下标定或在线拉格朗日更新自适应。

## 9. 时间受限失败后的逐次增配更新（动态规则）

- 在拦截时刻 $\tau$ 判定目标 $i$ 成功/失败。若失败，则将目标在 $\tau$ 的状态（位置、速度、EKF 协方差等）重新加入集合 $\mathcal{T}_{k'}$（其中 $t_{k'}\approx \tau$ 的下一决策步），并增记失败次数：
  $$
  z_{i,k'} \;=\; z_{i,k}+1.
  $$
- 下一步的最小并发武器数更新：
  $$
  n^{\min}_{i,k'} \;=\; \min\Big\{\,n_{\max},\;n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k'}\,\Big\}.
  $$
- 若成功拦截，则目标 $i$ 从后续 $\mathcal{T}_{\cdot}$ 中移除。

## 10. 可行性与拦截点计算（与 AntiWeapon/EKF 接口）

- 当 $x_{w,i,k}=1$ 时，拦截点与时间由几何约束求解：
  $$
  p_{w,i,k} \;=\; \mathrm{SolveIntercept}\big(w,\ \widehat{s}_i(t;\mathrm{EKF}),\ \tau_{w,i,k}\big),
  $$
  其中 $\widehat{s}_i(t;\mathrm{EKF})$ 为目标 $i$ 的 EKF 预测状态。必须满足：
  $$
  M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1,\quad P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},p_{w,i,k})\in[0,1].
  $$

## 11. 强化学习嵌入点（偏好由策略生成）

- 在上述约束下，策略选择 $\{m_{i,k},\ x_{w,i,k},\ \tau_{w,i,k}\}$ 以最大化效费比，并满足时间窗与可达性：
  $$
  r_k\;=\;\frac{\sum_{i} v_i\cdot P^{\mathrm{succ}}_{i,k}}{\sum_{w,i} x_{w,i,k}\cdot\big(c_w + c_t\cdot \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k)\big) + \epsilon}\; -\; \beta_{\mathrm{infeas}}\cdot \mathbb{I}\{\text{不可达/越窗}\},
  $$
  或采用加权差形式奖励：
  $$
  r_k\;=\;\sum_{i} v_i\cdot P^{\mathrm{succ}}_{i,k}\; -\; \lambda\cdot \sum_{w,i} x_{w,i,k}\cdot\big(c_w + c_t\cdot \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k)\big)\; -\; \beta_{\mathrm{infeas}}\cdot \mathbb{I}\{\text{不可达/越窗}\}.
  $$

- 逐次增配偏好通过 $m_{i,k}\ge n^{\min}_{i,k}$ 体现，RL 可在 $[n^{\min}_{i,k},\ n_{\max}]$ 内自适应增配并优化 $\tau$。

## 12. 简化说明与工程可落地性

- 占用约束包含 $\tau$ 的非线性依赖，适合在仿真与 RL 环境中通过“占用队列”维护；在纯 MIP 中可用时间桶近似（固定飞行时间）线性化。
- 忽略编队/区域约束后，模型重点在时间窗、可达性与逐次增配；与现有 EKF（/src/utils/filter.py）与 AntiWeapon（/src/models/air_equipment_motion.py）接口天然契合。
- 弹道导弹初始 $n_{\mathrm{init}}(\mathrm{BM})$ 可由场景配置设为 2 或 3；策略在满足 $m_{i,k}\ge n_{\mathrm{init}}(\mathrm{BM})$ 的同时可进一步加配（不超过 $n_{\max}$）。

---

### 备注：与决策变压器（DT）、IQL 与可微指派的结合（简述）

- 状态以图/集合形式编码（目标/武器节点 + 边特征），动作由两部分组成：并发指派矩阵与拦截时间。可用 Gumbel–Sinkhorn 产生近似双随机指派（满足“一武器一目标”的同时性约束），并对不可达边强制遮罩。
- 用决策变压器（Decision Transformer）建模长视界、时间受限的序列结构（输入历史 $(s,a,r)$ 与 $RTG_t$），输出当前的并发指派与 $\tau$ 分布参数；离线阶段采用隐式 Q 学习（IQL）进行优势加权策略拟合，在线阶段以拉格朗日（Primal–Dual）方式对关键约束进行安全细化。
- 奖励可采用上述比值或加权差形式，并加入违规惩罚项；逐次增配通过 $z_{i,k}$ 与 $n^{\min}_{i,k}$ 驱动，策略在可行域内自动权衡“成本上升 vs 成功率上升”。

---

## 13. 完整数学模型（汇总：Time-Restricted Dynamic WTA with Progressive Multi-Weapon Allocation）

为便于实现与论文呈现，下面给出一个可直接引用的完整模型（采用加权差目标作为主目标；比值形式可作为奖励替代或用于消融）。

### 13.1 目标函数（主目标：加权差形式）

$$
\max\;\; J \;=\; \sum_{k=0}^{K} \Bigg[\; \sum_{i\in \mathcal{T}_k} v_i \cdot P^{\mathrm{succ}}_{i,k} \;\; - \;\; \lambda\, \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\, \big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \; \Bigg] .
$$

其中 $\lambda>0$ 为权衡系数（可在线拉格朗日更新），$P^{\mathrm{succ}}_{i,k}$ 的定义见 13.2 中的第 8 条约束。

### 13.2 约束条件（核心）

1) 同步并发（武器侧）：

$$
\sum_{i\in\mathcal{T}_k} x_{w,i,k} \; \le \; 1,\quad \forall w\in\mathcal{W},\; k=0,\dots,K.
$$

2) 同步并发（目标侧）与并发选择：

$$
\sum_{w\in\mathcal{W}} x_{w,i,k} \;=\; m_{i,k},\quad 0\;\le\; m_{i,k} \;\le\; n_{\max},\quad m_{i,k}\in \mathbb{Z}_{\ge 0},\quad \forall i\in\mathcal{T}_k,\;\forall k.
$$

3) 逐次增配下界（体现偏好）：

$$
m_{i,k} \;\ge\; n^{\min}_{i,k} \;\triangleq\; \min\Big\{\, n_{\max},\; n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k} \,\Big\},\quad \forall i,k.
$$

4) 时间窗（time restricted）：

$$
x_{w,i,k}=1 \;\Rightarrow\; \tau_{w,i,k} \in [t_k,\, t_k+H],\quad \forall w,i,k.
$$

5) 几何可达性：

$$
x_{w,i,k}=1 \;\Rightarrow\; M_{w,i,k}(\tau_{w,i,k},\, p_{w,i,k}) = 1,\quad \forall w,i,k.
$$

6) 弹药约束：

$$
\sum_{k=0}^{K} \sum_{i\in\mathcal{T}_k} x_{w,i,k} \;\le\; \mathrm{ammo}_w,\quad \forall w\in\mathcal{W}.
$$

7) 占用与冷却（从可用集合删除的离散化近似）：

令

$$
L_{w,i,k} \;\triangleq\; \left\lceil \frac{\tau_{w,i,k} - t_k}{\Delta t} \right\rceil + \left\lceil \frac{t_{\mathrm{ready}}(w)}{\Delta t} \right\rceil ,
$$

则有

$$
x_{w,i,k} \; + \sum_{k'=k+1}^{\, \min\{\,k+L_{w,i,k},\,K\,\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} \; \le \; 1,\quad \forall w\in\mathcal{W},\; \forall k.
$$

8) 并发成功概率（独立假设下的至少一枚成功）：

$$
P^{\mathrm{succ}}_{i,k} \;=\; 1 \; - \; \prod_{w\in\mathcal{W}} \Big( 1 - x_{w,i,k}\, P^{\mathrm{kill}}_{w,i,k}(\tau_{w,i,k},\, p_{w,i,k}) \Big),\quad \forall i\in\mathcal{T}_k,\; \forall k.
$$

9) 逐次增配的失败累计（动态更新）：

在拦截时刻 $\tau_{w,i,k}$ 判定成功/失败。若失败，则令下一决策步 $k'$（满足 $t_{k'}\approx \tau_{w,i,k}$ 的最近决策时刻）

$$
z_{i,k'} \;=\; z_{i,k} + 1,\qquad \text{并据此更新 } n^{\min}_{i,k'} = \min\{ n_{\max},\, n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k'} \} .
$$

若成功拦截，则目标 $i$ 从后续 $\mathcal{T}_{\cdot}$ 中移除。

10) 资源总量与可用武器容量约束（统一可行性）：

为避免因弹药和占用导致“目标侧并发等式不可满足”，需要总量约束：令 $\mathcal{W}^{\mathrm{avail}}_k \triangleq \{\, w\in\mathcal{W}\mid e_{w,k}=0,\ \text{且弹药未耗尽}\,\}$，则

$$
\sum_{i\in\mathcal{T}_k} m_{i,k} \;\le\; \big|\, \mathcal{W}^{\mathrm{avail}}_k \,\big|,\quad \forall k.
$$

工程实现中，$e_{w,k}$ 由占用队列与冷却时间递推得到，“弹药未耗尽”由弹药计数与约束 6) 保证。

### 13.3 可选的比值型目标（用于奖励或消融）

$$
\max\;\; J_{\mathrm{ratio}} \;=\; \sum_{k=0}^{K}\; \frac{\displaystyle \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k}}{\displaystyle \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\,\big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \; +\; \epsilon },\quad \epsilon>0.
$$

---

## 14. 符号说明（Sets, Indices, Parameters, Variables）

### 14.1 集合与索引

- $\mathcal{W}$：武器集合；$w\in\mathcal{W}$ 为武器索引。
- $\mathcal{T}_k$：步 $k$ 的在场目标集合；$i\in\mathcal{T}_k$ 为目标索引。
- $k\in\{0,1,\dots,K\}$：决策步索引；$t_k=k\,\Delta t$ 为对应物理时间。

### 14.2 参数

- $v_i$：目标 $i$ 的价值（威胁效应）。
- $c_w$：武器 $w$ 的发射/资源成本。
- $c_t$：时间成本权重（反映响应延迟代价）。
- $c_{\mathrm{move}}(w,i,k)$：几何/机动引致的额外成本估计（可选）。
- $H$：时间受限窗口长度；$\tau\in[t_k,\,t_k+H]$。
- $M_{w,i,k}(\tau,p)\in\{0,1\}$：几何可达性遮罩（由 AntiWeapon + 物理约束给出）。
- $P^{\mathrm{kill}}_{w,i,k}(\tau,p)\in[0,1]$：单武器对目标的拦截成功概率估计（由几何、类型、速度等决定）。
- $\mathrm{ammo}_w$：武器 $w$ 的弹药/可出动次数上限。
- $t_{\mathrm{ready}}(w)$：武器 $w$ 的准备/冷却时间。
- $n_{\mathrm{init}}(\mathrm{type}_i)$：目标类型的初始并发武器偏好（BM 为 2 或 3；CM/AC 为 1）。
- $n_{\max}$：并发武器上限。
- $\lambda>0$：目标函数中的成本权衡系数。
- $\epsilon>0$：比值目标中的数值稳定项。
- 物理与预测接口：$\widehat{s}_i(t;\mathrm{EKF})$ 为目标 $i$ 的 EKF 预测状态；$\mathrm{SolveIntercept}(\cdot)$ 为求解拦截点与时间的几何过程。

### 14.3 决策变量与状态变量

- $x_{w,i,k}\in\{0,1\}$：是否在步 $k$ 将武器 $w$ 指派到目标 $i$。
- $m_{i,k}\in\mathbb{Z}_{\ge 0}$：步 $k$ 对目标 $i$ 的并发武器数量（由策略选择）。
- $z_{i,k}\in\mathbb{Z}_{\ge 0}$：目标 $i$ 截至步 $k$ 的累计拦截失败次数（环境/策略驱动更新）。
- $\tau_{w,i,k}\ge 0$：预计拦截时间（满足时间窗）。
- $p_{w,i,k}\in\mathbb{R}^3$：拦截点坐标（满足几何可达性）。
- $L_{w,i,k}\in\mathbb{Z}_{\ge 0}$：占用长度（拦截飞行时间 + 冷却时间的离散化近似）。

### 14.4 派生量

- $P^{\mathrm{succ}}_{i,k}$：并发成功概率，见式 (13.6)。
- $n^{\min}_{i,k}$：逐次增配下界，见式 (13.3)。

---

> 注：若用于纯 MIP/MIQP 优化，可将时间窗与占用近似为固定时间桶，并以线性化的可达性近似替代 $M(\cdot)$；若用于 RL 仿真，则保留上述非线性接口，环境在执行期负责状态更新与判定（含失败累计 $z_{i,k}$ 的递推与目标集合的动态维护）。

---

## 15. 简化版模型（工程基线，便于快速落地）

为便于两周内完成从“基线到全功能”的迭代，这里给出两个层级的简化版数学模型，保持与前文符号完全一致、含义统一。

### 15.1 基线 A（最简可用）：固定并发下界、拦截时间离散化、占用近似常数

- 并发武器数固定为逐次增配下界：

$$
m_{i,k} \;=\; n^{\min}_{i,k} \;=\; \min\Big\{ n_{\max},\ n_{\mathrm{init}}(\mathrm{type}_i) + z_{i,k} \Big\} .
$$

- 拦截时间采用离散集合（降低搜索维度）：

$$
\tau_{w,i,k} \in \mathcal{T}_k^{\mathrm{disc}} \;\triangleq\; \big\{\, t_k + b\,\Delta t\ \big|\ b=1,2,\dots,B\,\big\} \cap [\,t_k,\ t_k+H\,] .
$$

- 占用与冷却采用常数近似：给定武器常数占用长度 $L_w^{\mathrm{const}}\in\mathbb{Z}_{\ge 0}$，若 $x_{w,i,k}=1$，则在后续 $L_w^{\mathrm{const}}$ 个时间步视为占用，不可再次指派。

目标函数（沿用加权差形式）：

$$
\max\; J_A \;=\; \sum_{k=0}^{K} \Bigg[\; \sum_{i\in\mathcal{T}_k} v_i\, P^{\mathrm{succ}}_{i,k} \; - \; \lambda \sum_{w\in\mathcal{W}} \sum_{i\in\mathcal{T}_k} x_{w,i,k}\, \big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \Bigg] .
$$

核心约束（与 13.2 保持一致，但作离散与常数近似）：

- 武器侧并发：$\sum_{i\in\mathcal{T}_k} x_{w,i,k} \le 1$。
- 目标侧并发：$\sum_{w\in\mathcal{W}} x_{w,i,k} = m_{i,k}\ (\text{此处 } m_{i,k} \text{ 固定为 } n^{\min}_{i,k})$。
- 时间窗（离散）：$x_{w,i,k}=1 \Rightarrow \tau_{w,i,k}\in \mathcal{T}_k^{\mathrm{disc}}$。
- 可达性遮罩：$x_{w,i,k}=1 \Rightarrow M_{w,i,k}(\tau_{w,i,k},p_{w,i,k})=1$。
- 弹药：$\sum_{k} \sum_{i} x_{w,i,k} \le \mathrm{ammo}_w$。
- 占用近似：$x_{w,i,k} + \sum_{k'=k+1}^{\min\{k+L_w^{\mathrm{const}},K\}} \sum_{i'\in\mathcal{T}_{k'}} x_{w,i',k'} \le 1$。
- 资源总量：$\sum_{i\in\mathcal{T}_k} m_{i,k} \le \big|\mathcal{W}^{\mathrm{avail}}_k\big|$。
- 并发成功概率：$P^{\mathrm{succ}}_{i,k}$ 同 13.2-8）。
- 失败累计与增配：$z_{i,k}$ 的更新同 13.2-9）。

说明：基线 A 不学习“额外并发”，只执行“首轮单武器/多武器 + 失败逐次增配”的偏好；拦截时间离散化显著降低搜索与训练难度。

### 15.2 基线 B（两阶段轻量版）：并发选择离散化 + 拦截时间离散化

- 并发武器数在离散集合中选择：

$$
m_{i,k} \in \{\, n^{\min}_{i,k},\ n^{\min}_{i,k}+1,\ \dots,\ \min\{n_{\max},\ n^{\min}_{i,k}+M\} \,\},\quad M\in\mathbb{Z}_{\ge 0} .
$$

- 拦截时间同基线 A：$\tau_{w,i,k}\in \mathcal{T}_k^{\mathrm{disc}}$。

- 目标函数：$J_B$ 与 13.1 的主目标一致（加权差形式）。

- 约束集合：完全沿用 13.2（含资源总量约束 10)），仅将 $m_{i,k}$ 视为可选整数，并受 $m_{i,k}\ge n^{\min}_{i,k}$、$m_{i,k}\le n_{\max}$ 控制。

说明：基线 B 在保持工程简洁的同时，赋予策略对“何时增配到大于下界”的决策自由度；结合时间离散化，训练稳定且易于调参。

可选的分步奖励（用于 RL 训练，与 11 节一致）：

$$
r_k^{(\mathrm{diff})} \;=\; \sum_{i} v_i\, P^{\mathrm{succ}}_{i,k} \; - \; \lambda \sum_{w,i} x_{w,i,k}\, \big( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) \big) \; - \; \beta_{\mathrm{infeas}}\, \mathbb{I}\{\text{不可达/越窗}\},
$$

或采用比值型：

$$
r_k^{(\mathrm{ratio})} \;=\; \frac{\sum_{i} v_i\, P^{\mathrm{succ}}_{i,k}}{\sum_{w,i} x_{w,i,k}\, ( c_w + c_t\, \tau_{w,i,k} + c_{\mathrm{move}}(w,i,k) ) + \epsilon} \; - \; \beta_{\mathrm{infeas}}\, \mathbb{I}\{\text{不可达/越窗}\} .
$$

---

一致性与无歧义声明：

- 本节所有符号与含义严格复用 1–14 节的定义；其中并发成功概率、逐次增配规则、可达性与时间窗约束完全一致。
- 新增的“资源总量约束 10)”用于统一可行性，避免出现目标侧并发等式在资源不足时不可满足的情形；其实现依赖占用状态 $e_{w,k}$ 与弹药计数，二者已在 3 节与 13.2-6)/7) 中定义。
- 基线 A/B 的“离散化”仅针对拦截时间与并发选择的搜索空间，不改变物理意义和约束逻辑；若需要恢复连续时间与可微指派，可直接使用 13 节完整模型。