# LEAF Model Merging — Research POC 实施方案

## 1. 目标与范围

固定完整 LEAF 的 EEG Tower + Q-Former + 文本原型余弦分类路径，比较：独立专家、联合微调、均匀参数合并、全局可学习合并、block-wise 可学习合并。

首轮回答两个问题：可学习合并能否超过同一初始化下的联合训练？更细粒度的合并是否优于全局合并？先得到实验现象，不把 block-wise baseline 本身当作论文创新。

所有实现以 `d_downstream_qformer_ce.py` 为基础。不要使用 `d_downstream_add_mlp.py`，不要添加 MLP、线性分类头或数据集专属可训练 head。保留现有 Q-Former 及其 output_embed；固定 instruction 和 label prototypes。现有 `PrototypeClassifier` 是无新增可训练参数的读出封装，不是新增分类头。

## 2. 固定首轮实验设置

| 项目 | 首轮设置 |
|---|---|
| 初始化 | 同一个完整 instruction-tuned LEAF MPNet checkpoint |
| 模型配置 | configs/LEAF_mpnet.yaml |
| 数据集 | MI_OpenBMI、MI_Cho2017、MI_PhysioNet、EMO_SEED_3_seg4、SSVEP_OpenBMI |
| 输入 | 复用本地 EEG 缓存及文本 embedding 缓存，不改预处理和通道映射 |
| 数据划分 | 复用 load_dataset() 返回的 train / val / test |
| 精度 / batch size | bf16-mixed / 128 |
| Seed | 0 |
| 专家训练 | 每个数据集 1,000 个更新步 |
| 联合训练 | 5,000 个更新步，每个数据集恰好参与 1,000 步 |
| 优化器 | AdamW；tokenizer LR=1e-3，其余活动参数 LR=1e-4；weight decay=1e-3 |
| LR schedule | 前 5% 步数 warmup，随后 cosine decay 至峰值的 0.1 倍 |
| 模型选择 | 验证集 dataset-macro balanced accuracy；平局时取 macro CE 较小者 |
| 合并校准集 | 每个原 train 固定分层抽取至多 512 条，不足则全部使用 |
| 合并优化 | Adam，LR=1e-2，weight decay=0，500 个总更新步 |

1,000 步只是首轮搜索预算，不假设已收敛。训练曲线仍明显改善时，优先统一增加专家和联合训练预算，不引入新模块。

Calibration 是 train 的子集，专家和联合训练均可使用完整 train。这样合并没有额外的标注数据访问优势；val 只负责模型选择，test 不参与优化。保存 calibration indices 即可，不建立新的缓存系统。

## 3. 统一前向和损失

直接复用 `prepare_text()` 的类别顺序、instruction 构造与原型归一化。每个数据集 d 保存固定的 instruction u_d 和原型矩阵 P_d。

```python
embedding, _ = leaf(x, instruction[d].expand(len(x), -1))
logits = F.normalize(embedding.float(), dim=-1) @ prototypes[d].T
loss = F.cross_entropy(logits, y)
```

专家、联合训练、合并系数优化和最终评估全部使用相同计算。原型矩阵的行必须与当前数据集的整数标签一一对应。不要把不同数据集的类别索引直接拼接，不使用全局标签并集 CE，也不新增温度或文本编码器微调。

联合训练只有一个 LEAF 实例。每个 batch 只来自一个数据集，只切换 instruction 和 prototypes，不切换可训练模型参数。

## 4. 训练脚本：直接扩展 d_downstream_qformer_ce.py

在研究分支中做小范围修改，保留现有模型加载、prepare_text、make_loader、评价指标及 checkpoint 保存逻辑。

### 接口

让 `--ds` 接受单个名称或逗号分隔的多个名称。新增 `--steps-per-dataset`；设置该参数后，以步数而不是 epochs 控制训练。

### 数据调度

把一次 round 定义为每个数据集各取一个 batch，数据集顺序每轮随机打乱。一个 batch 对应一次 optimizer.step()，不要先累计全部数据集梯度再更新。

```python
for round_id in range(steps_per_dataset):
    for dataset in shuffled(dataset_names):
        x, y = next_batch(dataset)  # loader 用尽后重建 iterator
        loss = dataset_ce(leaf, x, y, dataset)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

单数据集时，这就是专家训练；五数据集时，每轮五步，1,000 轮共 5,000 步。总专家更新步数和联合训练更新步数相同，各数据集的 batch 暴露次数也相同。合并额外的 500 步另行记录，不声称整个流程严格 FLOPs 匹配。

每 100 个 round 在原 val 上评价，保存最佳 checkpoint。单专家按自己的 val BAcc 选择；联合模型按五个数据集 val BAcc 的等权平均选择。不用 source test 或外部数据集选择 checkpoint。验证后恢复训练模式，并再次固定 BN statistics。把原脚本结尾的 test 调用移至 eval_poc.py；训练和合并阶段只运行 train / val，最后统一评估 test。

### BatchNorm 的统一设置

LEAF tokenizer 使用 SyncBatchNorm。首轮统一固定其 running statistics 为初始化 checkpoint 的值，仿射参数仍参与训练。在每次 `model.train()` 后，将 SyncBatchNorm 单独设为 eval：

```python
for module in leaf.modules():
    if isinstance(module, torch.nn.SyncBatchNorm):
        module.eval()
```

这项修改同时用于专家和联合模型；其余 dropout 保持正常训练行为。该设置是本 POC 的明确选择，不是原脚本默认行为。首轮不增加 BN 重估或数据集专属 BN。

## 5. 合并脚本：merge_poc.py

### 合并公式

对第 k 个专家定义 delta_k = theta_k - theta_0。对参数组 g：

    theta_M[g] = theta_0[g] + sum_k alpha[k, g] * delta_k[g]

首轮仅实现三种模式：

| 模式 | 系数 |
|---|---|
| uniform | 固定 alpha_k = 1/K |
| global | 每个专家一个可学习系数 |
| block | 每个专家、每个结构化参数组一个可学习系数 |

所有可学习系数初始化为 1/K，使 global / block 的初始模型均等于 uniform。直接优化实数 alpha，不加 softmax、不加额外 gate、正则或 mask；允许负值和总和不为 1，输出时称为合并系数而不是概率。

### Block 分组

为当前 LEAF 写一个简单的 `group_for_name(name)`，按参数名前缀分组：

| 组 | 规则 |
|---|---|
| 输入组 | tower.tokenizer + tower.pos_embedding |
| 双向分支外围组 | tower.mask_inProj + tower.mask_transformer.norm |
| 因果分支外围组 | tower.next_inProj + tower.next_transformer.norm |
| 双向 Transformer | tower.mask_transformer.layers 中每个 block 单独一组 |
| 因果 Transformer | tower.next_transformer.layers 中每个 block 单独一组 |
| Q-Former 条件组 | qformer.base_queries + qformer.pos_embedding + qformer.film |
| Q-Former blocks | qformer.layers 中每个 block 单独一组 |
| 输出组 | qformer.final_norm + qformer.output_embed |

以默认两条 12 层分支、4 层 Q-Former 为例，共 33 组，5 个专家对应 165 个可学习系数。层内 attention、FFN、bias、LayerNorm 共用该 block 的系数。本轮不做 attention-head-wise、neuron-wise 或 element-wise 合并。

`tower.mask_token`、`tower.mask_outProj`、`tower.next_outProj` 不参与分类 forward，直接保留 theta_0 对应参数。

### 参数与 buffers

只对选中的 `named_parameters()` 计算 task vectors。所有 buffers 复制 theta_0；不要把完整 state_dict 逐项相减。尤其不能对包含 -inf 的 causal_mask 做减法。上面的 BN 固定策略保证各模型 running statistics 一致。

### 可微合并

theta_0 和所有 delta 都作为常量，只优化 alpha。用 `torch.func.functional_call` 执行合成参数的前向，不要在优化循环中通过 load_state_dict 或 .data.copy_ 安装权重。

```python
merged_params = compose_params(base_params, task_vectors, alpha, group_ids)
embedding, _ = torch.func.functional_call(
    leaf, (merged_params, base_buffers),
    (x, instruction[d].expand(len(x), -1)),
)
logits = F.normalize(embedding.float(), dim=-1) @ prototypes[d].T
loss = F.cross_entropy(logits, y)
loss.backward()
alpha_optimizer.step()
```

合并优化时 `leaf.eval()` 关闭 dropout，但不要使用 no_grad / inference_mode 包住上述计算。数据集等频调度，每步只用一个数据集的 calibration batch，共 500 个总更新步。

训练前将 uniform 初始化作为 step 0 验证结果；每 50 步验证一次。按 source val macro-BAcc 保存最佳 alpha。训练结束后将最佳 alpha 静态折叠成普通 LEAF state_dict，保存 merged.ckpt。推理时不保留专家列表或合并计算，也不做 ensemble 或 router。

## 6. 评估脚本：eval_poc.py

复用原脚本的 `prepare_text()`、`PrototypeClassifier` 和 `evaluate()`。对于每个待评估的普通 LEAF checkpoint，只切换数据集对应的固定文本读出。

统一评价 Base LEAF、各独立专家、Joint、Uniform、Global、Block。主结果报告各 source dataset 的 test BAcc、source macro-BAcc 和相对 Joint 的百分点差值；保留现有 Kappa / AUROC 输出，不增加新评价体系。

独立专家可以在每个对应数据集上组成 specialist reference，但必须标为多个模型，不可当作一个部署模型的成绩。不要在 test 上挑选 best expert。

外部评估沿用已讨论的 MI_Dreyer2023、MI_Weibo2014：本地有缓存和明确标签映射就接入；不为首轮 POC 新增下载与预处理任务。外部数据不用于 alpha 更新或 checkpoint 选择。它们只能支持相应 MI 外部迁移结论；是否严格未见，还需依据初始化 checkpoint 的实际训练数据记录判断，不能仅凭未参与本轮微调而宣称 zero-shot。

## 7. 文件范围与目标运行接口

只修改/新增以下 Python 文件，均放仓库根目录以方便复用：

```text
LEAF/
├── d_downstream_qformer_ce.py   # 扩展为单/多数据集训练
├── merge_poc.py                # uniform / global / block
├── eval_poc.py                 # 汇总普通 LEAF checkpoints 的结果
└── runs/poc/                   # 运行产物
```

沿用已有 model/env YAML。实验常量放命令行或简单 shell 文件中，不增加配置管理框架。

下面是需要实现的目标接口，不是仓库当前已经支持的全部命令：

```bash
BASE=checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt
DATASETS=MI_OpenBMI,MI_Cho2017,MI_PhysioNet,EMO_SEED_3_seg4,SSVEP_OpenBMI

# 独立专家；每次都加载同一个 BASE。
for DS in ${DATASETS//,/ }; do
  python d_downstream_qformer_ce.py \
    --ckpt "$BASE" --ds "$DS" --steps-per-dataset 1000 \
    --bs 128 --seed 0 --gpu 0 --outdir "runs/poc/experts/$DS"
done

# 联合训练：1,000 rounds × 5 datasets = 5,000 optimizer steps。
python d_downstream_qformer_ce.py \
  --ckpt "$BASE" --ds "$DATASETS" --steps-per-dataset 1000 \
  --bs 128 --seed 0 --gpu 0 --outdir runs/poc/joint

# 三种合并共享相同专家和 calibration indices；uniform 不优化。
for METHOD in uniform global block; do
  python merge_poc.py \
    --base "$BASE" --experts-dir runs/poc/experts \
    --datasets "$DATASETS" --method "$METHOD" \
    --calibration-per-dataset 512 --steps 500 --lr 1e-2 \
    --bs 128 --seed 0 --gpu 0 --outdir "runs/poc/merge_$METHOD"
done

python eval_poc.py \
  --base "$BASE" --run-dir runs/poc --datasets "$DATASETS" \
  --bs 128 --gpu 0 --output runs/poc/results.csv
```

实际初始化 checkpoint 路径以本地文件为准。

## 8. 输出与推进标准

保留普通 best/merged checkpoint、实际参数配置、训练/验证曲线、最佳 alpha 和 calibration indices。最终生成一个 results.csv，以及一个 block × expert 系数表/热图。记录各阶段实际更新步数和运行时间即可，不开发 profiling 系统。

训练前不开展多轮 smoke tests 或单元测试工程。直接启动实际训练，沿用现有日志排查运行错误。代码中保留严格 checkpoint 加载和必要的维度/标签对应检查即可；不要增加自动恢复、自动缩 batch、重试、跨版本兼容或异常兜底。

| 结果 | 下一步 |
|---|---|
| Global 超过 Uniform，Block 进一步提升 | 有细粒度合并信号，复跑 seed 1、2 |
| Merge 的 source 表现提高，外部 MI 不提高 | 暂时只支持已知任务整合，不支持外部泛化提升 |
| Joint 更好 | 先看专家是否学到增益及训练曲线，不立即叠加复杂算法 |

本轮暂不实现 Fisher、TIES、子空间分解、稀疏 mask、元学习、LoRA、动态路由、蒸馏、DDP/FSDP、完整消融矩阵或生产部署。

## 参考实现

- LEAF/d_downstream_qformer_ce.py: https://github.com/Jiang-Muyun/LEAF/blob/main/d_downstream_qformer_ce.py
- LEAF/LEAF.py: https://github.com/Jiang-Muyun/LEAF/blob/main/LEAF.py
- LEAF/load_datasets.py: https://github.com/Jiang-Muyun/LEAF/blob/main/load_datasets.py
- LEAF/configs/LEAF_mpnet.yaml: https://github.com/Jiang-Muyun/LEAF/blob/main/configs/LEAF_mpnet.yaml
- PyTorch functional_call: https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html

方案依据上述公开代码制定；尚未在用户本地缓存上执行训练。
