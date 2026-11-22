# EEG Foundation Model 的语言对齐合并：研究方案与 Codex 实施规范

**版本：** v1.0 · 2026-11-13  
**项目：** LEAF 后续研究  
**暂定论文题目：** *Specialize, Then Consolidate: Alignment-Preserving Merging of EEG Foundation Models*  
**文档性质：** 可执行的研究计划。现有 POC 结果、数学性质、待检验假设和计划实验分别标注；本文不代表新方法已经有效或新颖性已经成立。

---

## 0. 给 Codex 的执行摘要

### 0.1 本轮要实现什么

在现有 LEAF language-aligned post-training / downstream CE 框架上，实现一个**保留固定文本原型读出、同时合并 EEG backbone 与共享语义接口、最终导出单一静态模型**的研究流水线。

首轮核心候选暂称 **CPSC：Contrast-Preserving Semantic Canonicalisation**，中文为“保持类别对比分数的语义坐标校准”。流程如下：

1. 复用从同一完整初始化 checkpoint 微调得到的各数据集专家。
2. 利用每个任务的固定文本原型，找出决定该任务类别分数差的语义子空间。
3. 在它的正交补中，对专家最终语义输出做带恒等偏好的正交校准，以共同 base 的输出为参照。
4. 将校准精确折叠进专家最后一个语义投影线性层。
5. 对变换后的完整专家执行现有的分组参数合并，联合优化 backbone 和 interface 的合并系数。
6. 导出普通 checkpoint；推理只保留一个 backbone、一个共享 interface，以及固定任务文本元数据。

**首先检验：在每个专家自身任务的预测保持不变时，仅调整这些输出坐标，是否能改善最终参数合并？** 这是一个具体、可否证的假设，不是已证实的 EEG 机制。

### 0.2 默认范围与优先级

| 阶段 | 内容 | 默认执行范围 |
|---|---|---|
| P0 | 审计本地 POC、checkpoint、数据和读出协议 | 必做，复用已有资产 |
| P1 | 实现 CPSC、基线、缓存、合并、导出及必要正确性检查 | 必做 |
| P2 | 在现有 LEAF + Q-Former 上完成 seed 0 静态试验；有效后扩展三 seed | 第一轮研究主体 |
| P3 | 一个轻量共享接口、一个第二 backbone，以及论文级对照 | 静态证据成立后扩展 |
| P4 | 固定记忆预算的 sequential expert consolidation | 后续扩展，单独报告 |

本轮默认主方法是 `cpsc_block_ce`；`cpsc_block_semkd` 是预设扩展。不要在首轮同时换 backbone、connector、文本编码器和训练目标。

### 0.3 执行原则

- 先阅读实际仓库的 `AGENTS.md`、`git status` 和相关代码。保留本地未提交修改，旧 checkpoint 与 `runs/poc16` 只读。
- 优先复用已有专家、Joint 和校准索引；不要默认重新训练 48 个专家。
- 本文列出的新文件、配置字段和 CLI 是**待实现接口**，不能假定已经存在。
- 不新增 dataset-specific classifier；共享投影到文本空间的 projector 属于 semantic interface。
- 不将不同架构的模型权重直接相加；每一次参数合并都要求相同架构、参数命名、shape 和共同初始化。
- 所有需要数据支持的结论来自真实运行结果。缺数据、缺 checkpoint 或本地代码不匹配时，报告具体缺口并完成其余可执行部分；不生成模拟性能结果。
- 首轮交付代码、正确性检查、seed 0 报告和可复现实验命令。后续阶段按本文的证据门槛扩展，不以大规模架构搜索替代主假设检验。

---

## 1. 研究问题、现有证据与贡献边界

### 1.1 统一研究问题

不同 EEG 数据集上的监督微调，会同时改变 EEG 表征和解释这些表征的语义接口。本项目研究：

> 如何将这些共同初始化、共同文本读出空间下的专家，合并成一个模型，并在能力整合过程中保持任务相关的 EEG–language 对应关系？

backbone merging 和 language-aligned consolidation 是这个问题的两个组成部分。持续学习用于检验顺序接收专家时的整合能力，先不把它写成另一套独立方法。

### 1.2 已有 POC 证明了什么

依据 `POC16_Three_Seed_Report(2).md`，当前已完成 16 数据集、seed 0/1/2 的实验。主指标为官方 `c_direct_inference.py` 的 L2 source macro balanced accuracy；以下数值是已有结果，不是本方案新实验。[P1]

| 模型 | Macro BAcc，均值 ± sample SD | 说明 |
|---|---:|---|
| Base | 0.5407 | 同一个固定初始化 |
| Uniform | 0.5292 ± 0.0017 | 专家参数增量均值 |
| Global | 0.5393 ± 0.0019 | 16 个可学习系数 |
| Block | 0.5684 ± 0.0034 | 33 × 16 = 528 个可学习系数 |
| Joint | 0.6015 ± 0.0026 | 单模型多数据集联合训练 |
| Specialist reference | 0.6316 ± 0.0014 | 每任务使用自己的专家，共 16 个模型 |

Block 比 Global 好，但与 Joint 的配对差值仍为 −3.30 ± 0.31 个百分点。这个结果支持继续研究细粒度合并；**它没有证明缺口来自语义坐标漂移，也没有证明 Q-Former 是瓶颈**。Specialist 是多模型参考，不是理论上界。[P1]

当前 POC 已经同时合并 Tower 和 Q-Former，并用固定文本原型 CE 优化系数。因此，下列内容本身不构成新增贡献：拆分 backbone/interface；联合优化两部分系数；将现有 CE 改称 language-guided merging。

### 1.3 文献定位

| 相关工作 | 已经覆盖的部分 | 本项目必须额外证明的部分 |
|---|---|---|
| LEAF [R1–R2] | 指令条件化 EEG 表征、固定文本原型、共享语义接口 | 专家整合与持续整合是否有效 |
| Expert Merging [R3] | 用专家 hidden states / logits 优化层级合并系数 | 固定文本读出的特定几何结构是否提供额外收益 |
| MERA [R4] | 多模态持续学习中的组件错位、合并后再对齐 | 保持任务分数的预合并变换，及单共享接口的行为 |
| PivotMerge [R5] | projector 的 post-alignment merging、对齐引导的层权重 | 在 backbone 与接口均发生变化的 EEG 专家中，新的约束是否必要 |
| MAny [R6] | projection / low-rank 两部分的持续整合 | 单一固定容量模型的整合效果与成本，而非依赖增长的专家模块 |

**候选贡献：**利用固定文本原型下的预测等价性，在不改变源专家自身类别概率的条件下，主动改变其可合并性。该性质可给出简单证明；能否改善 EEG 合并、能否推广到其他接口、与已有重参数化/对齐方法有何增量，需要实验和进一步相关工作核查。不能仅凭该几何性质宣称 ICML/ICLR 级创新已经成立。

---

## 2. 固定任务设定

### 2.1 模型与读出

令完整模型参数为 $\Theta=(\theta,\phi)$，其中：

$$
h=B_\theta(x),\qquad
v=A_\phi(h,u_d),\qquad
e_\Theta(x,u_d)=\frac{v}{\|v\|_2},
$$

$$
p_{dc}=\operatorname{normalize}(E_{\rm text}(t_{dc})),\qquad
\ell_{dc}=\tau p_{dc}^{\top}e_\Theta(x,u_d).
$$

- $B_\theta$：EEG backbone。
- $A_\phi$：所有任务共享的语义接口；LEAF 中包括 FiLM、queries、Q-Former layers、相关位置编码和最终语义投影。
- $E_{\rm text}$：冻结的文本编码器；首轮继承 MPNet 文本缓存。
- $P_d\in\mathbb R^{C_d\times D}$：按数据集原始 label ID 顺序排列的归一化候选原型。
- $u_d$：任务说明与全部候选类别构成的条件；不使用当前样本真实标签。
- $\tau=1$：首轮继承现有 cosine logits；不同时引入可学习 temperature。

首轮监督目标为 `cross_entropy(logits, y)`。用户提供的 LEAF 稿 III-B Eq.13 描述正类 cosine alignment loss，而当前 `d_downstream_qformer_ce.py` / POC 使用原型 CE；本项目明确继承**当前 CE 实现**，不把二者混写。[R1–R2, P1]

### 2.2 公共初始化与专家

$$
\Theta_k=\operatorname{FineTune}(\Theta_0,\mathcal D_k^{\rm train}),\qquad
\Delta_k=\Theta_k-\Theta_0.
$$

首轮 $\Theta_0$ 为完整的 `leaf-v1.0-instruct-mpnet-base.ckpt`，包含 backbone 与 interface。要求 `strict=True` 加载；不能用 Tower-only checkpoint 留下随机 Q-Former。

**Backbone-agnostic 的定义：**同一个整合算法可以分别用于 LEAF、CBraMod 等架构；每次合并仅发生在同一架构的一组专家内部。跨架构直接参数相加不在本方案范围内。

### 2.3 数据、任务与语义边界

- 每个 batch 来自一个数据集，使用该数据集的 $P_d$、$u_d$ 和 label map。
- 不把不同数据集的整数 label ID 拼成一个全局 CE。
- 相同语言说明可能对应多个数据集；语言条件不等于 dataset ID，不引入可学习 dataset embedding。
- 共享标签名不意味着两数据集具有相同 EEG 分布，也不意味着相同神经机制。
- 合并、校准和验证都保留现有 subject/session split；不能重新做 trial-level 随机划分造成泄漏。
- 从专家集合中留出一个数据集，只能称为“未参与本轮专家合并”；若初始化已接触该数据集，不能宣称严格未见任务泛化。

---

## 3. 首轮候选方法：CPSC + Joint Block Merging

### 3.1 待检验假设

**H1：**固定类别原型只通过有限个原型差方向约束类别分数差；专家在其他输出方向上的不同变化，可能影响参数合并。

**H2：**将这部分输出坐标向共同 base 校准，在保持源专家自身预测的同时，可能改善合并后模型。

**H3：**这个效果在控制额外数据拟合、计算预算和普通蒸馏后仍然存在，并且不完全依赖 Q-Former。

H1 中关于分类器的几何事实成立；“实际存在有害漂移”“校准能改善合并”和 H2/H3 均须验证。不要推断正交补完全没有训练梯度，也不要将它直接解释为无用或无语义信息。

### 3.2 决定类别分数差的子空间

对任务 $d$，定义：

$$
\bar p_d=\frac1{C_d}\sum_c p_{dc},\qquad
V_d=\operatorname{span}\{p_{dc}-\bar p_d\}_{c=1}^{C_d}.
$$

取正交基 $Q_d\in\mathbb R^{D\times r_d}$ 和正交补基 $N_d\in\mathbb R^{D\times(D-r_d)}$：

$$
Q_d^\top Q_d=I,\quad N_d^\top N_d=I,\quad
Q_dQ_d^\top+N_dN_d^\top=I,\quad r_d\le C_d-1.
$$

因为：

$$
\ell_{dc}-\ell_{dc'}=\tau e^\top(p_{dc}-p_{dc'}),
$$

若一个正交矩阵 $R$ 在 $V_d$ 上恒等，那么将归一化输出 $e$ 变成 $Re$ 后，本任务所有类别分数差保持不变。因此固定温度下的 softmax、CE、margin 和预测保持不变；**原始 logits 可以一起增加相同的样本相关偏移，不要求逐项相等**。

这个性质针对固定 $P_d$，并不保证其他任务的标签集合或任意开放文本查询保持。它对任意输入与指令成立，条件是变换位于最终语义输出处且读出的 $P_d$ 不变；用于拟合变换的样本不是性质成立的必要条件。

### 3.3 校准数据与正交 Procrustes

对专家 $k$ 所属任务 $d=k$，使用其 train split 中既有的分层校准子集 $\mathcal C_k$，最多 512 条。使用相同输入、合法训练指令和候选原型，缓存：

$$
Z_k=[e_{\Theta_k}(x_i,u_k)]_{i=1}^{n_k}\in\mathbb R^{D\times n_k},\qquad
Z_0=[e_{\Theta_0}(x_i,u_k)]_{i=1}^{n_k}.
$$

取 $m=D-r_k$，并计算：

$$
Y_k=N_k^\top Z_k,\qquad Y_0=N_k^\top Z_0.
$$

求解带恒等偏好的正交对齐：

$$
R_{\perp,k}=\arg\min_{R^\top R=I}
\frac1{n_k}\|RY_k-Y_0\|_F^2+\mu_k\|R-I\|_F^2.
$$

令：

$$
C_0=Y_0Y_k^\top/n_k,\qquad
s_k=\|C_0\|_F/\sqrt m,\qquad
\mu_k=\eta s_k,
$$

$$
C_0+\mu_k I=U\Sigma V^\top,\qquad
R_{\perp,k}=UV^\top,
$$

$$
R_k=Q_kQ_k^\top+N_kR_{\perp,k}N_k^\top.
$$

默认 $\eta=1$。$\eta\in\{0.1,10\}$ 是预设敏感性检查，不是默认逐任务调参；若统一选取一个 $\eta$，只依据验证集，并记录全部搜索成本。

**实现约定：**

- 数学公式采用 embedding 列向量；代码缓存一般为 `[n_samples, text_dim]`，旋转对应 `z_new = z @ R.T`。
- 只中心化文本原型。默认不中心化 EEG 输出样本，也不单独归一化 $Y_k$ 的残差。
- 允许反射，不要求 `det(R)=+1`。
- 原型子空间和 Procrustes SVD 使用 CPU FP64；建议秩阈值 `rtol=1e-10, atol=1e-12`，同时记录实际秩和奇异值。
- $m=0$ 时，先于任何除以 $\sqrt m$ 的操作返回恒等；$s_k\le 10^{-12}$ 时也返回恒等。记录明确原因，不能随机旋转无可识别信号的方向。
- 512 条样本下正交补通常秩不足。恒等正则减少任意变化，但不能宣称消除了过拟合或保证唯一解。
- 不用 `(1-a) * I + a * R` 插值代替正则：该矩阵通常不再正交，会破坏保持性质。
- 所有专家先独立完成校准，base checkpoint 本身不改变。

### 3.4 折叠进最终语义投影

若接口最后一步为 $v=W_k h+b_k$，之后仅有 L2 归一化，则：

$$
W'_k=R_kW_k,\qquad b'_k=R_kb_k.
$$

由正交性，`normalize(R @ v) = R @ normalize(v)`，所以可以在不增加运行时模块的情况下实现该变换。

已核对的公开 LEAF commit 中，最后线性层为：

```text
qformer.output_embed.2.weight
qformer.output_embed.2.bias
```

该线性层之前有 flatten/dropout，之后由 `LEAF.forward` 做 L2 normalisation，满足折叠条件。[R2] **本地分支必须再次检查实际前向结构**；若末层之后还有 LayerNorm、非线性或其他输出路径，不能只按同名参数强行折叠。

折叠后的完整专家记为 $\Theta'_k$。随后重新计算：

$$
\Delta'_k=\Theta'_k-\Theta_0.
$$

特别注意：末层增量是 $R_kW_k-W_0$，不是 $R_k(W_k-W_0)$。只有专家末端投影被变换，其他专家参数沿用原值；原 checkpoint 始终只读。

折叠函数返回独立模型/独立 state，或在修改前保存可独立比较的原输出。不得原地修改 `expert` 后把同一对象作为 original 与 aligned 做对照，否则不变性检查可能虚假通过。

### 3.5 联合分组参数合并

继续使用现有 POC 的完整模型分组：

$$
\Theta_M^{(g)}=\Theta_0^{(g)}+
\sum_{k=1}^{K}\alpha_{k,g}\Delta_k'^{(g)}.
$$

- LEAF 首轮复用 33 组：input、mask outer、12 mask layers、next outer、12 next layers、condition、4 Q-Former layers、output。
- $\alpha_{k,g}$ 是**无约束实数**，允许负值，系数和无需为 1；初始化 `1/K`。
- 所有活动 backbone/interface 组共同通过完整复合函数前向优化。
- 主方法 `cpsc_block_ce` 仅优化系数，以原型 CE 为目标：

$$
\mathcal L_{\rm CE}=\frac1K\sum_d
\mathbb E_{(x,y)\sim\mathcal C_d}
\operatorname{CE}(P_de_{\Theta_M}(x,u_d),y).
$$

数据集等权，按 shuffled dataset rounds 实现；不按样本量让大数据集占据更多更新。

**解释边界：**原专家自身预测保持，不意味着合并模型预测保持；末端校准也不等于修复了全部 backbone–interface 错位。这些正是实验需要回答的部分。

### 3.6 预设功能蒸馏变体

为了区分几何校准与一般专家行为蒸馏，加入：

$$
\mathcal L_{\rm semKD}=\frac1K\sum_d\mathbb E_{x\sim\mathcal C_d}
\left[1-e_{\Theta_M}(x,u_d)^\top\tilde e_d(x)\right],
\qquad \mathcal L=\mathcal L_{\rm CE}+\lambda_{\rm KD}\mathcal L_{\rm semKD}.
$$

| 方法 | Teacher target | 默认系数 |
|---|---|---:|
| `block_semkd` | 原专家输出 $e_{\Theta_d}$ | $\lambda_{\rm KD}=1$ |
| `cpsc_block_semkd` | 校准后输出 $R_de_{\Theta_d}$ | $\lambda_{\rm KD}=1$ |

Teacher 必须 stop-gradient；每个输入只使用所属任务专家。不能要求所有专家在不相关 EEG/指令上互相一致。

可补 `block_logitkd`：在各自候选类别集合上计算 `KL(teacher || merged)`，固定 KD temperature 为 1。由于 CPSC 保持本任务 softmax，这一 logit target 在变换前后相同。**不要把普通 CE+KD 本身写成方法创新。**

### 3.7 可直接照做的伪代码

```python
# 已完成数据与 checkpoint 审计；所有原始模型、样本 ID、文本缓存固定。
base = load_complete_checkpoint(strict=True).eval()

for task in task_manifest:
    expert = load_expert(task, strict=True).eval()
    P, instruction = load_frozen_text(task)
    sample_ids = load_existing_calibration_ids(task)
    z_base, z_expert = cache_outputs(base, expert, sample_ids, instruction)
    Q, N = prototype_contrast_basis(P, dtype=float64)
    R = identity_regularized_procrustes(Q, N, z_expert, z_base, eta=1.0)
    aligned_expert = fold_into_final_linear(expert, R, inplace=False)
    verify_expert_invariance(expert, aligned_expert, R, task)
    save_alignment_and_teacher_cache(task, R, z_expert, sample_ids)
    del expert, aligned_expert  # 可以保存变换后的末层，避免复制整套专家文件。

delta_bank = build_active_deltas(base, experts, final_layer_transforms=R_bank)
alpha = initialize_signed_coefficients(K, groups, value=1.0 / K)
evaluate_step_zero_and_initialize_best_checkpoint()

for step, (task, batch) in equal_dataset_rounds(calibration_loaders):
    params = differentiable_merge(base, delta_bank, alpha)
    embedding = functional_forward(params, base_buffers, batch.x, task.instruction)
    logits = normalize_fp32(embedding) @ task.prototypes.T
    loss = cross_entropy(logits, batch.y)
    if method_uses_semkd:
        teacher = cached_teacher(task, batch.sample_ids, aligned=uses_cpsc)
        loss += lambda_kd * cosine_distillation(embedding, teacher)
    optimize_only_alpha(loss)
    if is_validation_step(step):
        select_by_validation_macro_bacc_then_ce()

merged = fold_selected_coefficients_to_plain_checkpoint()
verify_functional_vs_reloaded_checkpoint(merged)
evaluate_frozen_checkpoint_with_official_protocol(merged)
```

---

## 4. 实验矩阵与公平比较

### 4.1 P2 首轮矩阵

| ID / 方法 | 新增操作 | 用途 | 首轮安排 |
|---|---|---|---|
| `base`, `specialists`, `joint` | 无 | 原始参照 | 校验后复用 |
| `uniform`, `global_ce`, `block_ce` | 现有合并流程 | 已有基线 | 校验后复用；至少核对 seed 0 |
| `group_scale_ce` | 33 个组尺度参数 | 判断 Block 收益是否主要来自尺度调整 | 必做 |
| `block_semkd` | CE + 原专家完整输出 KD | 一般功能蒸馏基线 | 必做 |
| `cpsc_block_ce` | CPSC 后做原始 Block-CE | 隔离预测等价变换的效果 | 主候选，必做 |
| `cpsc_block_semkd` | CPSC + 校准坐标中的 KD | 判断功能保持是否带来额外收益 | 必做 |
| `cpsc_uniform` | CPSC 后直接平均 | 低成本检验初始可合并性变化 | 建议同时导出 |

GroupScale 明确定义为：

$$
\Theta_M^{(g)}=\Theta_0^{(g)}+
\lambda_g\left(\frac1K\sum_k\Delta_k^{(g)}\right),\qquad \lambda_g=1\text{ 初始化}.
$$

它与 Block 使用相同校准数据、批次顺序和 1,600 次更新，是尚待运行的控制。不要由原 POC 的系数和直接推出其结果。

### 4.2 当主候选有效时，补齐的关键控制

1. **全空间 Procrustes：**使用相同 teacher 样本、恒等正则和求解流程，但不保护 $V_d$。方法名 `fullrot_block_ce`。同时报告变换前后专家自身性能；它不具有任务分数保持保证。
2. **普通 logits KD：**补 `block_logitkd`，排除完整 embedding KD 的某个实现选择造成弱基线。
3. **相同总计算预算：**记录 CPSC 缓存/SVD/合并优化的全部耗时；给原始 Block-CE 近似相同总计算预算增加更新步数，并固定验证比较次数。使用独立 benchmark batch 估算单步耗时，提前固定步数，不根据测试结果调整。
4. **正则敏感性：**统一比较 $\eta=0.1,1,10$。若根据验证集选择主结果，对 KD 权重提供相近数量的预设候选并记录搜索总成本。
5. **校准样本数：**128 与 512 两档足够完成首轮敏感性检查；重新拟合全部校准依赖项，不能用 512 样本拟合 R 后把实验称作 128 样本方法。

CPSC 的正交矩阵由数据拟合，约有 $m(m-1)/2$ 个正交自由度。**只能说最终模型架构、部署参数量及合并系数数量不变，不能说总拟合自由度完全相同或完全 training-free。** 缓存、SVD、搜索和额外 teacher forward 都计入成本。

### 4.3 共享 interface repair：另列训练变体

如果纯参数合并仍有明显缺口，增加三个严格对应的控制：

- Base + interface-only CE repair。
- Block-CE + interface-only CE repair。
- CPSC-Block-CE + interface-only CE repair。

默认冻结 backbone，只训练同一共享 interface；复用同一 calibration pool、batch schedule、400 总步数、AdamW `lr=1e-4, weight_decay=1e-3`，每 40 步验证并按相同规则选择。三者使用完全相同预算和超参数。该设置是新的工程起点，不是已有 POC 配置。

这三项属于“合并/初始化 + 参数再训练”，与只优化合并系数的方法分表或明确标记。不能将额外 interface 训练带来的收益全部归因于 CPSC。

### 4.4 论文级外部基线

在进入完整论文实验前，补 TIES-Merging [R7] 和至少一个较接近的现代功能对齐方法，例如 Expert Merging [R3]；RegMean++ [R8] 可作为另一种依赖关系建模对照。

先固定官方代码版本和默认算法定义，再适配同一 EEG readout、校准集与指标。仅实现一项普通 KD 不等于复现了 Expert Merging 全方法；若采用简化版，命名必须显式包含 `adapted` 或 `simplified`，并列出差异。外部基线不应阻塞 P1 正确性验证，但完整论文结论不能只比较自己增加的弱变体。

---

## 5. 复用现有 POC 的具体配置

### 5.1 数据集清单

按实际 POC manifest 保留以下 16 项及其顺序，不用 LEAF 稿件的另一份“16 datasets”列表替换：[P1]

```yaml
datasets:
  - MI_OpenBMI
  - MI_BCIC_IV2a
  - MI_ShanghaiU
  - MI_BCIC_Upperlimb
  - MI_HighGamma
  - MI_Cho2017
  - MI_Shin2017A
  - MI_PhysioNet
  - EMO_FACED
  - EMO_SEED_3_seg4
  - EMO_SEED_4_seg4
  - EMO_SEED_5_seg4
  - EMO_SEED_7_seg4
  - SSVEP_OpenBMI
  - Workload
  - ADHD_AliMotie
```

MI_Weibo2014 仅为既有补充评测，不自动升级为严格未见数据的主要泛化证据。

### 5.2 已有训练与合并设置

| 项目 | 首轮设置 |
|---|---|
| 完整初始化 | `checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt` |
| 模型配置 | `configs/LEAF_mpnet.yaml` |
| 专家训练 | 每数据集 3,000 optimizer steps；首轮复用 |
| Joint 训练 | 3,000 rounds × 16 = 48,000 optimizer steps；首轮复用 |
| 专家/Joint optimiser | AdamW；tokenizer `1e-3`，其余活动参数 `1e-4`；WD `1e-3` |
| 专家/Joint schedule | 5% warmup；cosine 到峰值的 0.1 倍 |
| Batch / precision | 128；bf16 mixed；余弦读出及 CE 使用 FP32 |
| BN | running statistics 固定在 base；原训练中 affine 参数可训练 |
| 校准子集 | 每数据集 train 中最多 512 条，沿用现有分层索引 |
| 合并优化 | Adam，系数 LR `1e-2`，1,600 总步数，数据集等频 |
| 合并初始化 | Global/Block `1/K`；GroupScale `1` |
| 验证 | 更新前 step 0 及之后每 160 步，最大 macro BAcc；相同则最小 macro CE |
| Seed | 正式 `0,1,2`；先做 seed 0 |

以上是后续三 seed 报告的实际设置；不回退到早期 POC plan 的 5 数据集 / 1,000-step / 500-step 配置。[P1–P2]

统一将 step 0 纳入本轮可学习合并的选择范围；CPSC 的 step 0 是变换后专家的 Uniform。P0 核对旧实现是否已包含相同规则，若不同则在本轮基线中统一重跑并记录协议差异。无法恢复原 calibration IDs 时，新建固定索引并对所有可学习基线统一重跑，不能将旧表直接当作同校准条件的对照。

### 5.3 训练、验证和官方测试协议

- 校准拟合、teacher 缓存和 merge CE 使用现有 training readout。
- validation selection 复用 POC 的 validation readout 和宏平均规则。
- 主测试使用官方 `c_direct_inference.py`，`--level 2 --seed 42`；L0/L1 作为补充。
- 官方指令选择须对完整 `tasks.yaml` 按既有顺序抽样，再选数据集子集；不能让 `--ds` 子集改变选中的指令。
- 训练和官方评测可能使用不同任务说明，但候选标签原型一致。CPSC 的分数保持条件取决于原型差空间；应在这两种指令下都做数值核查。
- 若另做 label paraphrase，原型空间已改变，必须重新说明哪些原型差被保护；不能沿用“精确保持”的结论。
- 官方脚本只输出的指标与补充脚本指标分清。新增 CE/margin 分析保持同一指令与原型，并单列其协议标识。
- 所有 checkpoint/超参数选择依据 validation；固定方案后运行 test。现有 POC 测试结果已知这一事实应保留在实验日志中。

---

## 6. 工程实现要求

### 6.1 本地资产审计

POC 报告记载的仓库为 `/home/jingzhou/BFM-merge/LEAF`、分支 `poc-model-merging`，当时含未提交工作。这个路径只是定位线索；实际执行以当前 workspace 为准。[P1]

P0 输出 `audit.json`，至少包含：

- 当前 repo root、commit、dirty 状态，以及实际使用的 POC 代码文件。
- Base / experts / Joint 的真实路径、完整 SHA-256、seed、数据集、初始化来源和参数签名。
- 文本缓存、配置文件、label 顺序、指令字符串及其 hash。
- train/val/test 样本 ID；没有原始 ID 时生成可追溯的稳定 ID，并保留 subject/session 信息。
- 已有 calibration IDs、33 组映射、inactive 参数列表、buffer 列表。
- 缺失资产和当前可完成的实验子集。

公开仓库只能作为代码参照，不能假设其中已经包含本地 `merge_poc.py`、`eval_poc.py` 或 `runs/poc16`。公共代码参照 commit 为 `1e03d69a4a43fd1b17a4fdaf3bddc34d4d4c3f4a`。[R2]

### 6.2 最小新增模块

尽量在现有实现上扩展；若相同职责已存在，复用并记录差异，不复制第二套 trainer。

| 建议文件 | 职责 |
|---|---|
| `merge_aligned.py` | audit/cache/align/merge/evaluate/summarise 的薄 CLI |
| `aligned_merge/geometry.py` | 原型子空间、正则 Procrustes、末层折叠及数值诊断 |
| `aligned_merge/teachers.py` | 按 sample ID 缓存 base / expert 输出，管理坐标标识 |
| `aligned_merge/merger.py` | 封装或复用现有 functional merge、GroupScale 和损失 |
| `aligned_merge/adapters.py` | LEAF adapter；P3 再增加其他模型与接口 |
| `configs/merge_aligned/poc16.yaml` | 数据、已有资产、方法、预算和评测配置 |
| `tests/test_aligned_merge.py` | 影响数学正确性和实验有效性的必要检查 |
| `analysis/aligned_merge_report.py` | 汇总真实结果、配对差值、成本和对齐诊断 |

新的代码组织可以顺应本地仓库结构；以上是职责约束，不要求为目录名称重构整个项目。

### 6.3 Adapter 合同

每个 adapter 明确提供：

```python
encode_eeg(x, metadata) -> tokens[B, N, D_backbone], valid_mask[B, N]
encode_semantics(tokens, valid_mask, instruction) -> raw_output[B, D_text]
forward_normalized(x, instruction, metadata) -> embedding[B, D_text]
active_parameter_groups() -> dict[group_name, list[parameter_name]]
foldable_final_linear() -> weight_name, optional_bias_name
preprocessing_spec() -> sampling_rate, units, channel_order, patching, masks
```

LEAF adapter 应包装现有 forward，而非为了满足接口改写其数值行为。每个活动参数属于且仅属于一组。最终线性层的折叠条件由实际结构验证；不支持时明确报错，不偷偷增加额外推理模块。

### 6.4 参数、buffer 与梯度

- 对活动 `named_parameters()` 构造差值；**不要直接对整个 state_dict 做减法**。
- buffer 从 base 原样复制。因果 attention mask 含合法 `-inf`，不能相减生成 NaN。
- 原 POC 未参与 downstream forward 的 9 个张量继续留在 base；从本地活动列表确认，而非只凭数量匹配。
- 合并优化默认 `model.eval()`，关闭 dropout、固定 BN；梯度仍须流向合并系数。
- 用 `torch.func.functional_call` 或现有等价可微实现。优化循环内禁止用 `.data`、`load_state_dict` 或 `no_grad()` 安装合并参数。
- Teacher 缓存与最终评测可以 `no_grad()` / `inference_mode()`；学生优化前向不可以。
- 系数和参数增量保留 FP32；forward 可 bf16 autocast；最终 readout/loss 用 FP32。
- 最终导出普通完整 state_dict；必须可以不加载专家或 R bank 而独立预测。

### 6.5 缓存与结果账本

Teacher 缓存至少包含 `dataset, sample_id, expert_hash, base_hash, instruction_hash, prototypes_hash, preprocessing_hash, original_embedding`；CPSC 另存 `R, eta, basis_rank, coordinate_frame`。变换后的 target 可按需由原 embedding 与 R 计算。

缓存前向使用确定性预处理、`eval()` 和固定指令，不做随机 EEG augmentation。任何影响输入或输出的配置变化都应触发 cache mismatch；不能只根据文件名复用。

推荐每个实验输出：

```text
manifest.json
config_resolved.yaml
calibration_ids.json
groups.json
alignment_diagnostics.json
coefficients.pt
merged.ckpt
validation_history.csv
official_test.csv
per_dataset_metrics.csv
compute.json
report.md
```

R 和 teacher cache 可以作为实验资产保留，但不属于部署模型。`compute.json` 分别记录专家训练、缓存、分解、系数优化、repair、验证、测试的 wall time / GPU time / peak memory，不仅报告合并优化阶段。

### 6.6 配置草案

下面是待实现 schema 的参考。资产路径由 P0 解析生成 manifest；不能凭空构造不存在的 checkpoint 路径。

```yaml
experiment: eeg_language_aligned_merge_v1
phase: static
seed: 0
model:
  adapter: leaf_qformer
  config: configs/LEAF_mpnet.yaml
  base_checkpoint: checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
  text_encoder_frozen: true
  readout: fixed_text_cosine
  logit_scale: 1.0
assets:
  existing_poc_root: runs/poc16
  resolved_manifest: runs/aligned_merge_v1/inputs/assets.json
  preserve_existing_outputs: true
calibration:
  source_split: train
  max_samples_per_dataset: 512
  reuse_existing_indices: true
  augmentation: none
alignment:
  mode: contrast_preserving_procrustes
  reference: common_base
  eta: 1.0
  svd_dtype: float64
  rank_rtol: 1.0e-10
  rank_atol: 1.0e-12
  zero_covariance_threshold: 1.0e-12
merge:
  method: cpsc_block_ce
  grouping: existing_poc33
  coefficients: signed_unconstrained
  init: uniform
  optimizer: adam
  learning_rate: 0.01
  weight_decay: 0.0
  steps: 1600
  batch_size: 128
  dataset_sampling: equal_frequency_rounds
  semantic_kd_weight: 0.0
  student_mode: eval_with_grad
  buffer_source: base
validation:
  interval: 160
  selection: [macro_bacc_max, macro_ce_min]
evaluation:
  primary_protocol: official_direct_inference
  instruction_level: 2
  instruction_seed: 42
  extra_levels: [0, 1]
output:
  root: runs/aligned_merge_v1
  refuse_overwrite: true
```

`cpsc_block_semkd` 自动切换到对齐后的 teacher frame 且 `semantic_kd_weight=1.0`；`block_semkd` 使用原始 teacher frame。运行前将解析后的最终配置写入 manifest，避免 method 名称与实际 loss 不一致。

### 6.7 建议 CLI

以下命令为 Codex 需要实现的接口示例，不是当前已可运行的命令：

```bash
python merge_aligned.py audit --config configs/merge_aligned/poc16.yaml --seed 0
python merge_aligned.py cache --config configs/merge_aligned/poc16.yaml --seed 0
python merge_aligned.py align --config configs/merge_aligned/poc16.yaml --seed 0
python merge_aligned.py merge --config configs/merge_aligned/poc16.yaml --seed 0 --method block_ce
python merge_aligned.py merge --config configs/merge_aligned/poc16.yaml --seed 0 --method group_scale_ce
python merge_aligned.py merge --config configs/merge_aligned/poc16.yaml --seed 0 --method block_semkd
python merge_aligned.py merge --config configs/merge_aligned/poc16.yaml --seed 0 --method cpsc_block_ce
python merge_aligned.py merge --config configs/merge_aligned/poc16.yaml --seed 0 --method cpsc_block_semkd
python merge_aligned.py evaluate --config configs/merge_aligned/poc16.yaml --seed 0
python merge_aligned.py summarise --run-root runs/aligned_merge_v1
```

支持 `--dry-run` 列出真实输入、输出和预估运行数量。新增实验使用独立目录；不覆盖 `runs/poc16`，不自动删除或重写已有结果。

---

## 7. 必须通过的正确性检查

这里的检查针对数学性质、梯度、模型导出和数据有效性；不要求为每个低影响辅助函数增加形式化测试。

| 检查 | 方法与通过条件 |
|---|---|
| 参数与分组 | 同构、同 shape；每个活动参数仅出现一次；未用参数/buffers 保留 base |
| 原型空间 | FP64 检查 $Q^\top Q$、$N^\top N$、正交补和对中心化原型的重构；记录 rank |
| CPSC 正交性 | FP64 检查 $R^\top R\approx I$、$RQ\approx Q$，最大绝对误差目标 `1e-8` |
| 输出范数 | 比较未归一化 $v$ 与 $Rv$ 的范数；记录接近零的原输出，检查归一化退化 |
| 专家折叠 | 原模型输出右乘 `R.T` 与折叠后模型输出一致；不只检查权重公式 |
| 源任务预测保持 | 比较中心化 logits、softmax、CE、margin；FP32 前向误差目标 `1e-4` 以内，报告实测值 |
| 初始合并等价 | 原始 Global/Block 的 `1/K` 与 Uniform 一致；GroupScale 的 `1` 与 Uniform 一致；CPSC 对应检查其自己的变换后 Uniform |
| 合并梯度 | 至少一个真实 batch 上，backbone 和 interface 的活动系数均获得有限梯度；全量分组覆盖另行核查 |
| 导出等价 | FP32 functional merge 与重载的普通 checkpoint 比较 logits；目标 `max_abs_diff <= 1e-5`，超出则检查后报告 |
| 数据边界 | calibration 全部来自 train，与 val/test ID 无重叠；保留 subject/session 约束 |
| 条件与 label | 只使用合法 instruction + 全部候选类别；prototype 行序与 numeric label ID 完全一致 |
| 独立部署 | 移除专家路径/缓存依赖后，单 checkpoint + 固定文本元数据能完成评测 |

数值检查关闭 autocast 与 TF32，模型处于 `eval()`；纯矩阵性质用 FP64。近乎平局的样本可能因浮点舍入改变 argmax，应结合 margin 与误差界单列，而不是静默忽略或伪称逐样本严格一致。

CPSC 的性质检查使用训练校准样本和 validation 样本；测试集只在方法固定后检查，不以测试误差调整 SVD 阈值。应在训练指令及官方 L2 指令下核对；两者保持同一候选原型集合。

---

## 8. 评估、分析与推进门槛

### 8.1 主指标

- 官方 L2 的 source macro BAcc；先逐数据集计算，再等权平均。
- 三训练 seed 的均值、sample SD；方法差值在相同 seed 内配对计算。
- 每数据集 BAcc，以及相对 Base、Block、Joint 的变化。
- 最差任务变化、下降任务数量、当前高损失任务的变化，防止宏平均掩盖局部退化。
- 总计算成本、部署参数量、训练期额外资产和内存。

不能把 16 个异质数据集当作 16 次独立训练重复，也不由三 seed 自动得出强显著性结论。若做 subject-level 统计，应保留真实 subject IDs，并明确跨任务宏平均与被试内不确定性的区别。

### 8.2 必要机制分析

1. **先确认专家不变性：**变换前后本任务概率和性能保持到数值误差范围。
2. **再比较合并结果：**原始 Block 与 CPSC-Block 的配对性能差异。
3. **输出对齐变化：**校准集和 validation 上分别报告 $\|R_k e_k-e_0\|^2$ 的正交补部分；训练集改善不等于泛化改善。
4. **参数变化：**末层实际增量范数、与 base/其他专家增量的角度，以及 merge 之后的范数。不要把 $\sum_k\alpha_{k,g}$ 解释为参数移动了多少倍。
5. **对应控制：**完整输出 KD、全空间对齐、计算预算控制是否解释了全部收益。

可用小型 2×2 交叉组合 $(B_0,A_0),(B_0,A_M),(B_M,A_0),(B_M,A_M)$ 辅助分析 co-adaptation；不为每个组合额外调参。非加性只能说明组件交互，不能单独证明“语义漂移导致性能下降”。此分析不应成为实现主候选前的大规模诊断门槛。

### 8.3 阶段门槛

| 门槛 | 决策规则 |
|---|---|
| 工程通过 | 第 7 节检查通过；否则先修正实现，不跑完整 sweep |
| Seed 0 pilot | 完成第 4.1 节矩阵；先看 validation。实用推进参考：主候选较 Block 提升至少 0.5 pp，且没有任务下降超过 3 pp；这是资源管理阈值，不是统计结论 |
| 敏感性检查 | 若默认 $\eta=1$ 无效，只执行预设 `0.1/10` 检查并保留全部结果；不要按测试集逐任务搜索 |
| 三 seed | pilot 支持继续后，统一配置扩展到 `0,1,2`，固定后做正式 test |
| 方法增量成立 | 需对强基线与公平预算有稳定收益；若只优于 Uniform，或与普通 KD 相当，应收窄结论 |
| 扩展 | 静态结果明确后再做第二 interface/backbone 和 CL；不把扩展实验用于掩盖主结果阴性 |

这些门槛决定后续计算投入；不把“超过 Joint”设为工程验收条件。Joint 是重要参照，保留有用专家能力、提升成本效率或减少持续整合遗忘也可能形成结果，但需清晰限定结论。

---

## 9. P3：接口与 backbone 泛化

### 9.1 先保留 Q-Former 的原因

现有 checkpoint、专家和完整 POC 已经覆盖 LEAF + Q-Former。直接复用可以隔离合并方法带来的变化。Q-Former 不是方法的必要条件；CPSC 要求的是固定文本原型与可折叠的最终线性语义输出。

### 9.2 第二接口：instruction-conditioned attention pooling

实现一个轻量、共享的接口对照，不以“新 connector”作为首轮论文贡献：

$$
q=W_q u_d,\qquad K=HW_K,\qquad V=HW_V,
$$

$$
a=\operatorname{softmax}(qK^\top/\sqrt{d_a}+\mathrm{mask}),\qquad
v=W_2\operatorname{GELU}(W_1(aV)+b_1)+b_2.
$$

起始配置：单 query，$d_a=256$，共享 projector `256 -> D_text -> D_text`，最后 Linear 后只做 L2 normalisation。所有任务共享参数，task conditioning 仍来自合法文本输入；不加 dataset embedding、router 或 task-specific head。

建立新的公共初始化 `(B0, A0_pool)` 后，所有专家从这同一个 checkpoint 出发。不能直接合并不同随机初始化的接口，也不能把 Q-Former 专家与 pooling 专家权重混在一起。

静态设置下，一个可执行起点是冻结预训练 backbone，用所有 source **train splits** 等频训练共享接口 3,000 总步数，AdamW `lr=1e-3, weight_decay=1e-3`，batch 128；选择规则与数据曝光单列。随后再从该公共 checkpoint 训练同预算专家和 Joint。这个新初始化过程须计入成本，不与原 LEAF 公共 checkpoint 假定完全等价。

### 9.3 第二 backbone

默认优先 CBraMod；如果本地已有更完整的 LaBraM adapter、checkpoint 和数据适配，改用 LaBraM，二者择一。CBraMod 官方代码给出了移除输出投影、提取 EEG 表征的接入方式。[R9]

接入时核对原生 sampling rate、signal units、channel order、patch length、padding mask 与预训练权重加载。不要为了统一接口，盲目套用 LEAF 的 tokenizer 或全部预处理。

推荐最小配置集合：

| 配置 | 验证目标 |
|---|---|
| LEAF + 原 Q-Former | 主实验 |
| LEAF + conditional pooling | 是否依赖 Q-Former |
| CBraMod/LaBraM + conditional pooling | 是否依赖 LEAF backbone |

每个配置内部使用相同初始化、相同专家预算比较合并方法。跨配置绝对成绩受初始化和参数量影响，不能据此独立归因于某个组件。

### 9.4 其他项目论文的使用边界

- EEG-PRIME：提供接口设计参考，但其 task-specific FT 包含额外 MLP 分类头；直接复用其结果不满足本项目读出约束。其 dataset embedding、subject adversarial loss、LQM 等也不是“只替换 connector”的单一变量。[P3]
- EEG-SPELL：稿件 lexical supervision 来自 Whisper 音频特征，下游为 CLS + MLP；不是现成的 class-text prototype 接口。[P4]
- NeuroLM：采用 EEG tokenizer 与 causal LLM 的答案 token 生成；首轮接入会改变输出与训练范式，暂作相关工作而非直接替换 backbone 的基线。[P5]

---

## 10. P4：固定预算的持续专家整合

### 10.1 协议定义

先实现 **common-base specialist arrival / continual consolidation**：专家始终从同一公共 $\Theta_0$ 训练，但按指定顺序送达整合器。整合器在阶段 $t$ 仅访问：

- 固定 base $\Theta_0$。
- 上一阶段的单一模型 $\Theta_{t-1}$。
- 当前到达的专家 $\Theta_{e,t}$ 及当前 calibration data。
- 有总预算限制的旧任务 replay memory。

这不同于从上一合并模型继续微调出新专家，也不同于 task-unknown class-incremental learning。测试允许任务说明与该任务候选标签集，但不提供真实类别。

### 10.2 递归合并

只对**新到达的单任务专家**执行 CPSC，参照固定 base，得到 $\Theta'_{e,t}$。不要在当前任务的保护空间下旋转已经承载多个旧任务的 $\Theta_{t-1}$。

对 $t\ge2$：

$$
\Theta_t^{(g)}=\Theta_0^{(g)}+
a_{t,g}(\Theta_{t-1}^{(g)}-\Theta_0^{(g)})+
b_{t,g}(\Theta'_{e,t}^{(g)}-\Theta_0^{(g)}).
$$

初始化 $a_{t,g}=(t-1)/t,\ b_{t,g}=1/t$，优化 2G 个 signed coefficients。上一模型已学习过系数后，这个初始化不等于所有历史专家的精确均值。第一阶段直接采用当前专家；CPSC 变体采用变换后的专家，其本任务概率保持。

每阶段折叠为一个模型，当前阶段完成后不保留历史 expert bank。部署模型参数量固定；训练中 base、旧模型、当前专家的数量也是常数。

### 10.3 Replay 与数据边界

默认预算：**最多 1,024 条原始 EEG，且 EEG payload 不超过 1 GiB**；两个限制同时满足，记录实际 byte 数。另记录标签、sample IDs、文本元数据及当前阶段临时 teacher 缓存大小。

- 旧记忆在已见数据集间尽可能均分，再在数据集内按类别分层；不足配额按固定规则分配。
- 更新 memory 只访问上一 memory 与当前最多 512 条 calibration pool；不能重新访问旧数据集完整 train split 选样。
- 缩减旧配额时按固定 hash 顺序裁剪；样本选择与 byte cap 的规则在第一个阶段前确定。
- 当前 calibration 在阶段结束后释放，只有被选入 memory 的样本保留。
- 旧 teacher target 由更新前的 $\Theta_{t-1}$ 在旧 memory 上生成；当前 teacher target 为 CPSC 后当前专家输出。
- 旧 teacher、当前 teacher 与 student 使用相同任务指令和候选原型定义；临时缓存绑定 instruction/prototype hash，不能混用不同条件下生成的输出。
- 标签/文本原型是增长的任务元数据，单独计量；不能把无限增长的缓存隐藏在“固定内存”结论里。
- 首轮允许 raw replay；不宣称 data-free。Feature-only memory 无法直接提供旧 EEG 经 backbone 的梯度约束，应另设协议。

### 10.4 阶段损失与预算

从当前校准集与旧 memory 各取一个 minibatch；旧 batch 先均匀选一个已见任务，再采样该任务样本，避免大任务支配。

$$
\mathcal L_t=\tfrac12\mathcal L_{\rm CE}^{\rm current}
+\tfrac12\mathcal L_{\rm CE}^{\rm replay}
+\lambda_{\rm KD}\left(\tfrac12\mathcal L_{\rm semKD}^{\rm current}
+\tfrac12\mathcal L_{\rm semKD}^{\rm replay}\right).
$$

默认 $\lambda_{\rm KD}=1$、每阶段 400 次 coefficient updates、Adam LR `1e-2`；当前/旧 batch 各 64 条。第一阶段没有 replay，不使用半权重公式，直接评测当前专家。该设置是待验证的 CL 起点，不是已有结果。

首版使用预先固定的步数及最后一步模型，不保留所有旧 validation 用于 early stopping。若另做验证缓冲，必须计入相同总记忆预算。每阶段 test 由只读 evaluator 运行，不向更新器提供调参反馈。

### 10.5 最小 CL 对照与顺序

| 方法 | 含义 |
|---|---|
| Current expert replacement | 直接用当前专家替换旧模型 |
| Recursive Uniform | 原始专家的固定均值递归 |
| Recursive Block-CE | 不做 CPSC、不做 KD；相同 replay / 更新预算 |
| Recursive Block-SemKD | 不做 CPSC；相同 replay 和 teacher 生成规则，用于隔离几何校准 |
| CPSC recursive consolidation | CPSC + 相同 CE/KD 与 replay |
| Sequential fine-tuning + replay | 在上一模型上训练；记录与预训练专家路线不同的成本来源 |

“相同 teacher 生成规则”指：旧 teacher 为各方法自身的更新前模型，当前 teacher 为该方法实际使用的原始或校准后专家；不要求不同方法共享同一份旧 teacher 张量。

`Sequential fine-tuning + replay` 默认仅使用当前最多 512 条 calibration 与同预算 replay，当前/旧 batch 各 64 条、400 次更新，训练全部活动 backbone/interface 参数。LEAF 配置用 AdamW：tokenizer `1e-3`、其余 `1e-4`、WD `1e-3`，固定学习率、固定 BN running statistics，使用最后一步模型；CE 的 current/replay 权重与递归方法相同。它从上一模型训练，不额外读取当前专家参数。第一阶段统一从当前专家开始，比较之后的持续整合。补充相同预算的 CE + SemKD 版本时，teacher 为该方法更新前模型和当前专家，$\lambda_{\rm KD}=1$，显式命名 `sequential_ft_semkd_replay`；此版本需读取当前专家并计入其成本。

先用 1 个专家训练 seed × 3 个固定顺序做 pilot；正式使用 3 expert seeds × 相同 3 orders。顺序由 manifest 上的 NumPy `Generator(PCG64(order_seed)).permutation(K)` 生成，`order_seed=[17,29,43]`，保存完整列表；不按结果挑顺序。

### 10.6 指标

令 $R_{t,j}$ 为阶段 $t$ 模型在已见任务 $j$ 上的 BAcc：

$$
A_t=\frac1t\sum_{j=1}^{t}R_{t,j},\qquad
\bar A=\frac1T\sum_{t=1}^{T}A_t,
$$

$$
F_T=\frac1{T-1}\sum_{j<T}
\max\left(0,\max_{s=j,\ldots,T-1}R_{s,j}-R_{T,j}\right).
$$

报告 final $A_T$、$\bar A$、$F_T$、current-task $R_{t,t}$、完整矩阵、顺序敏感性与成本。Seed 与 order 分开呈现，不把所有相关运行简单当作独立随机重复。

### 10.7 关于“新任务”的限制

现有 LEAF language-aligned 初始化可能已见相关任务。用它做顺序整合，可以评价**专家到达与整合时的遗忘**；不能直接声称模型第一次接触这些任务。

若要做严格 future-task-unseen CL，需要另建不使用未来任务监督的公共 alignment checkpoint，并审计预训练/post-training 暴露。P3 中“用所有 source train 建立共享 interface”的静态初始化不能直接用于这一严格声明。CL 超参数也不得通过当前流的未来任务 validation/test 选择；若借用过这些数据，必须明确为 task-aware development 协议。

---

## 11. 实施里程碑与最终交付

### P0：可复用资产与协议锁定

- [ ] 本地代码、dirty 状态、checkpoint、数据和文本缓存审计完成。
- [ ] 与 POC 的 33 组映射、calibration IDs、训练/官方读出对应。
- [ ] Base 和一个已有专家的复评与原报告在相同协议下相符。
- [ ] 新实验目录与 manifest 建立，原结果未修改。

### P1：数学与工程可执行

- [ ] CPSC 求解、末层折叠和目标坐标缓存实现。
- [ ] Block / GroupScale / 普通 KD / CPSC 的配置可独立运行。
- [ ] 数值保持、梯度路径、导出等价、数据边界检查通过。
- [ ] 一个实际数据集上完成小预算端到端 smoke run；小预算数值不作为论文结果。

### P2：静态研究结论

- [ ] seed 0 的完整必做矩阵与真实指标、计算成本完成。
- [ ] 按验证结果执行预设推进门槛并记录决策。
- [ ] 若推进，固定配置完成三 seed；补齐关键公平对照。
- [ ] 报告明确区分已有 POC、新实验、阴性结果和解释性假设。

### P3 / P4：后续扩展

- [ ] 一个第二 interface 和一个第二 backbone；分别建立公共初始化。
- [ ] 补充论文级外部基线。
- [ ] 固定预算 CL、完整性能矩阵与初始化暴露说明。

**Codex 每轮交付说明应包含：**改了哪些文件；实际运行了哪些实验；结果位于哪里；是否通过正确性检查；哪些里程碑尚未执行及具体原因；下一条可运行命令。没有执行的实验标注 `planned` 或 `not_run`。

---

## 12. 研究风险与预设解释

| 观察结果 | 可支持的解释 | 后续处理 |
|---|---|---|
| 源专家概率保持，但 merge 无改善 | 末端预测等价坐标并非主要障碍，或 base 参照无效 | 保留阴性结果；以普通 KD/repair 作为基线继续评估，不夸大几何作用 |
| CPSC 与普通 KD 相当 | 新几何步骤没有证明独立增量 | 收窄方法声明，分析成本与稳健性，不把重复功能包装成新贡献 |
| 全空间旋转优于受限旋转 | 允许改变源专家功能可能更有效 | 明确牺牲了预测保持；重新审视论文问题，而非隐藏对照 |
| 仅校准集对齐改善 | 高维拟合可能过拟合 | 检查 validation 对齐、恒等正则与样本敏感性 |
| 末层 task vector 大幅增大 | 输出校准可能破坏参数可合并性 | 报告真实范数/夹角，不能仅展示 embedding 更相似 |
| 只有 repair 后提高 | 改善可能依赖额外 interface 参数训练 | 以相同 repair 预算的 Base / Block 控制判断增量 |
| CL 有效但静态弱 | 顺序整合与一次性整合的作用不同 | 分别报告，不用一种设置的结果替另一种作结论 |

最终论文应围绕最明确、可复现的证据收敛。暂定题目可以保留；只有当方法和实验确实支持时，才将“alignment-preserving”进一步具体化为已证实的能力保持机制。

---

## 13. 参考依据

### 项目内部资料

- **[P1]** `POC16_Three_Seed_Report(2).md`，16 datasets / 3 seeds，运行日期 2026-11-04。本文实际 POC 参数与指标以此为准。
- **[P2]** `LEAF_Model_Merging_POC_Plan(2).md`，早期 POC 计划；与 P1 冲突时，以实际运行和 P1 为准。
- **[P3]** 用户提供 `Zhang 2026-EEG-Prime.md`，重点核对 III-C3 与 III-D；不额外推定其发表状态。
- **[P4]** 用户提供 `Guo-2026-EEG-SPELL.md`，重点核对 III-C1 与 III-D；不额外推定其发表状态。
- **[P5]** 用户提供 `Jiang-2025-NeuroLM.md`，重点核对 §2.1、§2.3；官方项目：[NeuroLM](https://github.com/935963004/NeuroLM)。
- **[P6]** 用户提供 `Jiang 2026 Leaf.md`，重点核对 III-B / III-C；稿件方法描述与实际代码有差异时分别记录，不静默替换。

### 公开原始来源

- **[R1]** *LEAF: Language-EEG Aligned Foundation Model for Brain-Computer Interfaces*. [arXiv:2509.24302](https://arxiv.org/abs/2509.24302)。
- **[R2]** [LEAF 官方仓库](https://github.com/Jiang-Muyun/LEAF)，代码核对版本 `1e03d69a4a43fd1b17a4fdaf3bddc34d4d4c3f4a`：[LEAF.py](https://github.com/Jiang-Muyun/LEAF/blob/1e03d69a4a43fd1b17a4fdaf3bddc34d4d4c3f4a/LEAF.py)、[d_downstream_qformer_ce.py](https://github.com/Jiang-Muyun/LEAF/blob/1e03d69a4a43fd1b17a4fdaf3bddc34d4d4c3f4a/d_downstream_qformer_ce.py)、[c_direct_inference.py](https://github.com/Jiang-Muyun/LEAF/blob/1e03d69a4a43fd1b17a4fdaf3bddc34d4d4c3f4a/c_direct_inference.py)。
- **[R3]** *Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking*. ICLR 2026. [正式论文页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/24dbe8babeca9f372712bbf19dd3584e-Abstract-Conference.html)。
- **[R4]** *Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs*. EMNLP 2025. [ACL Anthology](https://aclanthology.org/2025.emnlp-main.665/). DOI: [10.18653/v1/2025.emnlp-main.665](https://doi.org/10.18653/v1/2025.emnlp-main.665)。
- **[R5]** *PivotMerge: Bridging Heterogeneous Multimodal Pre-training via Post-Alignment Model Merging*. 2026 arXiv preprint. [arXiv:2604.22823](https://arxiv.org/abs/2604.22823)。
- **[R6]** *MAny: Merge Anything for Multimodal Continual Instruction Tuning*. 2026 arXiv preprint. [arXiv:2604.14016](https://arxiv.org/abs/2604.14016)。
- **[R7]** *TIES-Merging: Resolving Interference When Merging Models*. NeurIPS 2023. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)。
- **[R8]** *RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging*. arXiv v3 标注 TMLR accepted. [arXiv:2508.03121](https://arxiv.org/abs/2508.03121)。
- **[R9]** *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*. [官方代码与接入说明](https://github.com/wjq-learning/CBraMod)。LaBraM 替代适配参考：[官方仓库](https://github.com/935963004/LaBraM)。

文献状态按本方案核对时的原始页面记录。原调研笔记中无法核实的题名或接收状态不作为本方案依据。
