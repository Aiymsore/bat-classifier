# 伪标签反哺（Cluster 核心区 → 物种模型自训练闭环）Spec（v4）

本文档描述 ⑥ 批量整理与专家标注之后的**模型反哺闭环**：用专家审核结论 +
Cluster 结构，把少量专家标签安全地扩展为一批**伪标签（pseudo label）**训练样本，
带权并入已确认物种模型的训练集，迭代提升模型覆盖度。

核心思想：

```text
专家只审代表样本（每簇 ~20 条）
+ 聚类把"已审结论"扩展到簇内高纯核心区（不是全簇！）
+ recording-level OOF 模型概率做第二道闸门（杜绝同录音自我确认）
+ 伪标签永不冒充专家标签，降权训练、独立留出评估
+ 采纳前与旧模型在专家留出集上对比，不退步才换模型
```

> **本 spec 只做设计约定，不含实现。** 实现前请先通读
> [species_model.md](species_model.md)（物种模型与模型对照）。

## 修订记录

| 版本 | 变更 |
|---|---|
| v1 | 初版：距离半径 + 纯度 + 模型置信三闸门；固定比例否决 |
| v2 | ① 簇最少已审 5→10；② 自动反哺收紧为 HIGH_CONSISTENCY only；③ 新增 I9：评分概率必须 recording-level OOF / cross-fit；④ 明确三种模型角色（Production / OOF Scoring / Evaluation）；⑤ nearest-first 填充改为 Safe Pool + 两阶段多样性抽样；⑥ 新增 top1-top2 margin 闸门（含冗余性推导）；⑦ 预留 per-species 阈值；⑧ 采纳增加稳定性报告（诊断项）；⑨ rejection_reason / 出处链 / 测试计划全量扩充 |
| **v3** | ① **HIGH_CONSISTENCY 改由 OOF 预测判定**：Cluster 一致性 = OOF 预测 vs 专家主导标签，Production 模型**完全退出**一切伪标签准入决策（含概率闸门），无 OOF → 零传播（I11）；② **新增 ⑥「专家数据划分」**：专家真值按 `source_recording` 拆为 Expert Train Pool / Evaluation Set（冻结考试卷），`expert_split.json` 版本化冻结，评估表（Support/Precision/Recall/F1/Macro-F1，缺真值 N/A）；③ **新增局部专家邻域检查**（K=5、半径=core_radius、最小邻居 3、一致率 ≥ 0.80），防大簇内部异质小区域误传播；④ core radius 明确为 Train Pool 一致证据的 P95 + radius_fallback；⑤ Evaluation Set **一条都不参与**伪标签传播依据（reviewed 计数/纯度/radius/局部邻域/propagated_from/OOF 训练）；⑥ 冻结后 Evaluation 录音上的新标注 parked（录音隔离优先）；⑦ 落地：`expert_split.py` / `oof_scoring.py` / `curation/pseudo_labels.py` / 评估与对比 API / UI 子面板 / Test 1–10 |
| **v4（本版）** | ① **Evaluation 生命周期重定义为 Fixed Gold Evaluation（外部锚点）**：三类数据严格分离（Expert Train Pool / Fixed Gold Evaluation / Pseudo Training Pool）；评估集的录音、pulse、真值一旦建成 v1 **永不修改、永不自动扩充**；能进入 Evaluation 的**只有独立人工专家真值**（模型判断 / pseudo label / HIGH_CONSISTENCY / 高概率脉冲 / 人工只"确认模型预测"一律禁止），伪标签**永远是伪标签**（`label_source=pseudo`），唯一升级专家真值路径 = 人工专家重新真实审核该 pulse；未来更新 = 用新完整录音 + 独立人工审核另建 **Evaluation v2**（v1/v2 独立存储绝不覆盖，v2 建议盲审）；② **新增 EVAL_CONTAMINATED 三重校验**：`expert_train ∩ evaluation = pseudo_training ∩ evaluation = oof_training ∩ evaluation = ∅`，任一非空立即终止并报 `EVAL_CONTAMINATED` + 具体 `source_recording`，不自动修；③ **新旧模型对比改为同一 Gold Evaluation 双重重新评估**（禁止拿旧模型缓存指标对比新模型现算指标）；④ **修复代表样本偏置**：Cluster 一致性改为 **Cluster-wide OOF consistency**——覆盖整个可合法 OOF 评分的 Train-side Cluster（OOF top1 == dominant 的全簇比例），representatives 只提供 Expert Truth；新增 `cluster_oof_coverage` 与 `PSEUDO_MIN_OOF_CLUSTER_COVERAGE = 0.80`，覆盖不足 → `INSUFFICIENT_OOF_COVERAGE` 禁止传播，**绝不 fallback Production**；⑤ 明确**两层判断分离**（Cluster-level OOF consistency vs Candidate-level OOF scoring，绝不混为一个 probability）；⑥ 每簇输出 §4.1 全字段（total/eligible/coverage/agree/disagree/agreement/status）；⑦ 落地：`gold_eval_v{N}.json` 归档 API / `evaluate_payload_on_gold_evaluation` / `compare_models_on_gold_evaluation` / cluster-wide OOF 门控 / Test 1–10 重排 |

---

## 0. 背景与定位

### 0.1 要解决的问题

⑥ 流程的产出瓶颈在专家标注：每簇默认只抽 20 条代表样本，专家审完后
`expert_training_dataset.csv` 往往只有几十到几百条。用它训练的
`CalibratedClassifierCV + RandomForest` 在代表样本之外的大量未审脉冲上
覆盖不足。反哺闭环的目标是：**在不污染专家真值的前提下**，把高置信、
低噪声的未审脉冲以伪标签形式带权加入训练，扩大有效训练规模。

v2 的额外目标：进一步压低五类风险——confirmation bias、同录音泄漏式
自我确认、小样本簇误传播、大量高度重复 pseudo 样本、模型概率虚高
导致的错误反哺。**原则不变：宁可第一版少产 pseudo，也不扩大错误标签。**

### 0.2 非目标（明确不做）

- **不做**无监督标签传播到全簇；混合簇（mixed）一条都不传播；
- **不做**用伪标签评估模型（指标只认专家标签）；
- **不改** `expert_training_dataset.csv` 的语义与生成逻辑；
- **不改** `expert_label` 字段的含义（专家真值，只增不改）；
- **不替代**主动学习：MODERATE / LOW / MIXED / INSUFFICIENT 簇
  进入人工补审队列，而不是自动打标；
- **不做**动态 pseudo 权重、自动阈值学习（见 §17 Future Work）。

### 0.3 与既有弱监督的关系

项目里已有先例：`noise_gate` 用 `accepted→bat / rejected→noise` 的原型伪标签
训练门控模型。本设计沿用同一哲学（弱标签、可审计、独立通道），但门槛更高：
门控伪标签来自确定性规则，物种伪标签来自"专家结论 + 聚类结构 + OOF 模型置信"
三重交集。

---

## 1. 安全不变式（红线，实现不得违反）

| # | 不变式 | 对应既有约束 |
|---|---|---|
| I1 | 伪标签**绝不写入** `expert_label`；专家真值只增不改 | `species_model.py`："绝不覆盖 expert_label" |
| I2 | 伪标签走**独立文件 + 独立列**（`label_source=pseudo`），与专家样本物理隔离 | `annotations.py`："review hint，不用于自动传播标签" |
| I3 | 训练/校准/评估**按 `source_recording` 分组隔离**；伪标签样本随录音分组走，绝不跨集合 | `build_group_splits` 既有 group split |
| I4 | **评估只在专家标注的留出录音上进行**；伪标签样本永不进入评估/校准集合 | `species_model.md` §1.2 录音级隔离 |
| I5 | 传播只允许发生在**高纯度簇的高置信核心区**；MIXED 簇零传播 | `cluster_stats.mixed`、`determine_cluster_status` |
| I6 | 每条伪标签保留完整**出处链**（见 §7.3），可审计可回滚 | `rejected_pulses.csv` 的 rejection_reason 传统 |
| I7 | 新模型只有在**专家留出集指标不退步**时才采纳；否则保留旧模型 | — |
| I8 | 伪标签是**可选输入**，由用户显式开启；不开启时训练行为与现在完全一致 | — |
| **I9（v2 新增）** | **决定某 pulse 是否成为伪标签的模型概率，必须来自训练时完全没见过该 pulse 所属 `source_recording` 的评分模型**（recording-level OOF / grouped cross-fitting） | `source_recording_series` 分组约定的自然延伸 |
| **I10（v2 新增）** | OOF 评分模型只以**专家样本**为训练真值（任何轮次都不喂伪标签），且**只用于评分**，绝不作为 Production 模型、绝不参与评估/校准真值 | 防止反馈放大 |
| **I11（v2 新增）** | OOF 概率**不可用时零传播**（`oof_probability_unavailable`），**绝不回退**用 Production 模型概率充当 | "宁缺毋滥" |
| **I12（v3 新增）** | **Evaluation Set 冻结**：`expert_split.json` 一经冻结，评估录音/pulse 清单不再随标注增长而变化；评估指标只在冻结集上、真值**只有**专家标签 | 考试卷不许长大 |
| **I13（v3 新增）** | **Evaluation Set 一条都不参与伪标签传播依据**：reviewed 计数、dominant_fraction、core radius、局部专家邻域、`propagated_from`、OOF 训练，全部只用 Expert Train Pool | 防"用考试题当教材" |
| **I14（v3 新增）** | 专家真值按 **`source_recording` 整体拆分**（Train Pool / Evaluation Set），同一录音的 pulse 绝不拆到两侧 | 录音级隔离的自然延伸 |
| **I15（v3 新增）** | 冻结后 **Evaluation 录音上新增的专家标注 parked**：既不进训练也不进冻结评估集（录音隔离优先于"新标注默认进 Train Pool"）；想启用须显式重新生成划分（`split_version`+1） | 两个要求冲突时的保守裁决 |
| **I16（v4 新增）** | **Fixed Gold Evaluation 不可变**：评估集 v1 的录音 / pulse / 真值一旦建成**永不修改、永不自动扩充**；模型训练、pseudo 轮次、新录音、新 pseudo label 都碰不到它。它在整个 self-training 系统**之外**，是外部锚点 | "考试卷不许长大"升级为"考试卷不许被系统碰" |
| **I17（v4 新增）** | **能进入 Evaluation 的只有独立人工专家真值**：模型判断结果、pseudo label、HIGH_CONSISTENCY / 高概率脉冲、人工只"确认模型预测"一律禁止；**伪标签永远是伪标签**（`label_source=pseudo`），即使 P=0.999 / margin=0.98 / 99% 一致——唯一升级路径 = 人工专家**重新真实审核**该 pulse | 专家真值的来源纯洁性 |
| **I18（v4 新增）** | **Cluster-wide OOF consistency**：簇一致性必须覆盖**整个可合法 OOF 评分的 Train-side Cluster**（禁止只用 reviewed representatives 计算）；`cluster_oof_coverage < PSEUDO_MIN_OOF_CLUSTER_COVERAGE` → `INSUFFICIENT_OOF_COVERAGE`、禁止传播，**绝不 fallback Production 模型** | 修复代表样本偏置 |
| **I19（v4 新增）** | **EVAL_CONTAMINATED 三重校验**：`expert_train_recordings ∩ evaluation_recordings = ∅`、`pseudo_training_recordings ∩ evaluation_recordings = ∅`、`oof_training_recordings ∩ evaluation_recordings = ∅`——任一非空**立即终止**并报 `EVAL_CONTAMINATED` + 具体 `source_recording`，**不自动修** | I3/I4/I12 的统一入口 |

---

## 2. 三种模型角色（实现阶段严禁混用）

| 角色 | 用途 | 训练数据 | 隔离要求 |
|---|---|---|---|
| **A. Production / Current Species Model** | 正常物种预测、⑥ 模型对照（cluster status）、⑦ 片段识别 | 专家样本（+ 已采纳伪标签，带权） | 既有 group split 不变 |
| **B. OOF Pseudo-label Scoring Model（fold 模型组）** | 只为 pseudo candidate 计算 `model_probability / top1 / top2 / margin` | **仅专家样本**，且每个 fold 模型的训练集**不含**被评分 pulse 的 `source_recording` | I9/I10；一次性评分仪器，评分后可丢弃 |
| **C. Final Evaluation** | 采纳判定 | 真值**只有** `expert_label` | recording-level holdout；伪标签永不进 evaluation / calibration truth |

明确声明：为 OOF scoring 增加的任何机制**不得**改变 C 的语义。
B 与 A 是不同的对象：B 的 fold 模型可以"看起来准确率更低"（训练数据更少），
这不影响 A 的展示指标；反之 A 的指标再好也不能替代 B 的评分资格（I9）。

---

## 2.5 专家数据划分（Expert Train Pool / Evaluation Set，v3 新增）

### 2.5.1 拆分算法（`expert_split.py: make_expert_split`）

单位是 **`source_recording` 整条录音**（I14），绝不拆同一录音的 pulse：

1. 统计每物种录音数；录音数不足两侧覆盖（< `MIN_TRAIN_PER_CLASS` +
   `MIN_EVAL_PER_CLASS` = 2）的物种记入 `insufficient_classes`，
   其录音**强制留在 Train**（Train 侧必须能学到该物种）；
2. Phase 1（覆盖）：贪心集合覆盖优先选录音进 Evaluation，保证每个
   物种 Evaluation 侧有真值（平局按 `random_state` 固定置换序）；
3. Phase 2（补量）：按固定置换序补录音到目标 `EXPERT_SPLIT_EVAL_FRACTION`
   （0.20，目标值非硬指标），全程保证每物种 Train 侧 ≥ 1 条录音；
4. 输出：`train_recordings / evaluation_recordings / evaluation_pulse_ids /
   per_class_train_count / per_class_evaluation_count / status /
   insufficient_classes / uncovered_classes`。

确定性（Test 4）：同一数据 + 同一 `random_state` → 划分完全一致。
比例是目标不是承诺——真正约束是录音级整体拆分与逐类覆盖。

### 2.5.2 冻结与版本（I12/I15）

- 首次保存即冻结 `tables/expert_split.json`（`split_version = 1`）；
  之后**不允许静默改写**——重划必须显式 `replace=True` 并 `split_version + 1`
  （UI 强警告："重新划分后，新旧模型指标将不再与之前 evaluation version
  完全可比"）；
- 冻结后新增专家标注的去向（`new_label_destination`）：
  - Train 录音上的新标注 → **Train Pool**（默认进训练池）；
  - **Evaluation 录音上的新标注 → parked**（I15）：录音隔离优先，
    既不训练也不进冻结评估集；想启用须重新生成划分。
- 训练前 `assert_eval_isolation` 强校验：评估录音出现在训练 / 校准 /
  伪标签数据中 → 抛 `EvalContaminationError`（消息含 `EVAL_CONTAMINATED`
  + 违规录音 ID），训练终止。

### 2.5.3 评估表（冻结 Evaluation Set，专家真值）

`classification_table` 逐类输出 **Support / Precision / Recall / F1** +
**Macro-F1**（只对 support ≥ 1 的物种等权平均）：

- **Evaluation 真值 = 0 的物种**：指标显示 **N/A** + `INSUFFICIENT_EVAL_SUPPORT`，
  绝不输出误导性的 F1 = 0（"模型表现为 0"与"根本没有真值"必须区分）；
- 真值**只有** `expert_label`（`eval_labels = "expert_only"`），伪标签 /
  `model_inferred_label` 永不混入；
- 每次训练随模型落盘 `expert_evaluation.csv`（仅作展示缓存）；
  **新旧对比必须走 `compare_models_on_gold_evaluation`：两个模型同时在
  同一个 Gold Evaluation 上重新评估**（§2.5.5，禁止拿旧缓存指标对比
  新模型现算指标）；Macro-F1 delta、逐类 recall 下降 >
  `PSEUDO_MAX_RECALL_DROP` = 0.02 拒绝采纳。

### 2.5.4 Evaluation 防泄漏链（v3 全链路）

| 环节 | 防泄漏手段 |
|---|---|
| 划分 | `source_recording` 整体拆分（I14）；`evaluation_pulse_ids` 冻结清单 |
| 训练 | `apply_expert_split` 只交出 Train Pool；`assert_eval_isolation` 断言训练/伪标签录音 ∩ 评估录音 = ∅（`EvalContaminationError`） |
| 校准 | calibration folds 只在 Train Pool 内构造，校准侧只有专家样本 |
| 伪标签准入 | reviewed 计数 / 纯度 / core radius / 局部邻域 / `propagated_from` / OOF 训练数据全部只用 Train Pool（I13） |
| 评估 | 只在冻结 Evaluation Set 上计算，真值专家-only（I12） |
| 增长 | 评估集不随标注增长（I12）；新标注按 §2.5.2 去向处理（I15） |

### 2.5.5 Fixed Gold Evaluation 生命周期（v4 新增：外部锚点）

**三类数据严格分离**，绝不允许互相流动：

| # | 数据类 | 内容 | 唯一用途 |
|---|---|---|---|
| 1 | **Expert Train Pool** | 专家真实审核的脉冲（Train 侧录音） | 唯一专家来源：模型训练、校准、OOF 评分训练、Cluster 证据（reviewed/dominant/core radius/局部邻域） |
| 2 | **Fixed Gold Evaluation** | 冻结考试卷（录音 / pulse / 专家真值全部固定） | **只出分**：Precision / Recall / F1 / Macro-F1 / log loss / 混淆矩阵 + 新旧模型对比 |
| 3 | **Pseudo Training Pool** | 伪标签（`label_source=pseudo`，权重 0.4） | 只进训练（Train Pool），永远不进 1、永远不进 2 |

**Evaluation 是整个 self-training 系统之外的外部锚点**，回答一个生死问题：
"模型是真的进步了，还是只是越来越相信自己过去生成的 pseudo label"。
正因为如此，它的生命周期被刻意设计为"冷"的：

1. **建卷（v1）**：从一批**新完整录音**的独立人工专家审核生成
   `tables/evaluations/gold_eval_v1.json`（`gold_record_from_split` /
   `create_gold_evaluation`），同时冻结 recording / pulse / expert truth /
   `per_class_count` / `eval_labels="expert_only"`；
2. **不可变（I16）**：v1 一经写入**永不修改、永不自动扩充**；模型训练、
   pseudo 轮次、新录音、新 pseudo label 都碰不到它；同版本重写直接
   `FileExistsError`；
3. **准入（I17）**：只有独立人工专家真值能进 Evaluation。
   `assert_no_pseudo_labels` 在建档与训练前双向校验——
   `label_source=pseudo`、带 `pseudo_label`、非 `reviewed`、缺
   `expert_label` 的行一律拒绝；模型判断结果、HIGH_CONSISTENCY / 高概率
   脉冲、人工只"确认模型预测"的点击同样不是专家真值；
4. **伪标签升级路径唯一（I17）**：pseudo → expert 的**唯一**途径是
   人工专家**重新真实审核该 pulse**（并按 I15 的去向规则落位）；
   P=0.999 / margin=0.98 / 99% agreement 都不改变 `label_source=pseudo`；
5. **未来更新 = 另建 v2**：用**新的完整录音** + 独立人工审核生成
   `gold_eval_v2.json`；v1 与 v2 **独立存储、绝不覆盖**，各自版本指标
   分开解读；v2 创建**默认建议盲审**（先盲审提交 `expert_label`，
   再展示模型结果）——该 UI 可后置，但语义先固定在此；
6. **隔离（I19）**：Evaluation 录音与 Expert Train / Species Model 训练 /
   `CalibratedClassifierCV` fitting / OOF 评分训练 / Cluster purity /
   Cluster reviewed / dominant_fraction / core radius / 局部专家邻域 /
   `propagated_from` / pseudo 候选生成 / Pseudo Training Pool **全部互斥**；
   训练 / OOF / pseudo 反馈开始前统一跑 `assert_eval_isolation`
   （`expert_train ∩ evaluation = pseudo_training ∩ evaluation =
   oof_training ∩ evaluation = ∅`），任一非空立即抛
   `EvalContaminationError`（消息含 `EVAL_CONTAMINATED` + 具体
   `source_recording`），**不自动修**；
7. **同卷对比（§2.5.3）**：`compare_models_on_gold_evaluation` 把
   old / new **两个模型都在同一个 Gold Evaluation 上重新跑一遍**再对比
   （`same_evaluation=True`，真值取冻结快照，活标注被篡改也不影响指标），
   绝不拿旧模型以前缓存的结果与新模型当前结果直接比较。

---

## 3. pseudo-label 决策全流程（v2 总图）

```text
专家审核（reviewed + expert_label）——只取 Expert Train Pool（I13：Evaluation Set 一条不用）
    ↓
reviewed >= 10（PSEUDO_MIN_REVIEWED）
    ↓
dominant_fraction >= 0.90（≥ 9/10 同一物种，Train Pool 口径）
    ↓
OOF Cluster Consistency == HIGH_CONSISTENCY（v3：OOF 预测 vs 专家主导标签；
    复用 determine_cluster_status 同一套阈值，Production 模型不参与）
    ↓
建立 Cluster Core Radius（Train Pool 一致证据距离 P95，宁窄勿宽）
    ↓
所有未审 accepted pulse（排除专家样本 / boundary / outlier / noise / uncertain）
    ↓
distance_to_center <= core_radius
    ↓
局部 Expert Train 邻域检查（K=5 最近同簇专家样本，radius 内；
    邻居 < 3 → insufficient（记录后放行）；一致率 < 0.80 或局部主导 ≠ 簇主导 → 拒）
    ↓
recording-level OOF model prediction（I9：评分模型没见过该录音）
    ↓
top1 == cluster dominant species
    ↓
calibrated P(top1) >= threshold（未校准 0.95）
    ↓
top1 - top2 >= PSEUDO_MIN_PROBABILITY_MARGIN
    ↓
Safe Candidate Pool（此时还不是训练样本）
    ↓
diversity sampling：时间桶去冗余 + 每录音每类上限 + 距离分层轮转
    ↓
pseudo_training_dataset.csv（label_source=pseudo，完整出处链）
    ↓
expert weight 1.0 / pseudo weight 0.4
    ↓
训练 v(n+1)（校准只用专家样本；Evaluation Set 不参与训练/校准/伪标签）
    ↓
冻结 Evaluation Set 上的专家真值评估（Support/Precision/Recall/F1/Macro-F1）
    ↓
和 v(n) 对比（同一 split_version；+ 稳定性报告，诊断项）
    ↓
通过 adoption gate → 采纳 v(n+1) → 下一轮重筛（轮次 ≤ 2）
```

低一致性路径（主动学习闭环，**不是**垃圾桶）：

```text
MODERATE_CONSISTENCY / LOW_CONSISTENCY / MIXED / INSUFFICIENT_SAMPLES
        ↓
人工补审优先队列（pseudo_cluster_gates.csv: recommended_action）
        ↓
专家增加真标签（--add-samples / 继续审核）
        ↓
重新计算簇门控 → 回到主流程
```

关键点：**百分比不参与准入判定**（§11 设计决策 D1），"最近 10%"不做主判据；
数量只受 §6.3 配额约束。

---

## 4. 簇级门控与样本级安全闸门

### 4.1 簇级门控（cluster gate）——**第一层：Cluster-level OOF consistency**

**两层判断严格分离（v4），绝不混为一个 probability**：

| 层 | 回答的问题 | 证据域 |
|---|---|---|
| **第一层：Cluster-level OOF consistency（本节）** | "这个 Cluster **整体**是否与专家 dominant species 一致？" | 整个可合法 OOF 评分的 Train-side Cluster |
| **第二层：Candidate-level OOF scoring（§4.3 #6–9）** | "这一条**具体 pulse** 是否值得生成 pseudo label？" | 该 pulse 自己的合法 OOF 预测 |

对每个 Cluster 计算门控结论，**全部满足**才允许传播。
**全部证据只取 Expert Train Pool**（I13）——Evaluation Set 的专家样本
不计入 reviewed 数 / 纯度，不参与一致性统计、半径与局部邻域。

| 条件 | 默认阈值 | 说明 |
|---|---|---|
| 已审样本数（reviewed 且有标签，Train Pool） | ≥ `PSEUDO_MIN_REVIEWED` = **10** | v2：5 → 10，见 D8 |
| 主导标签占比（Train Pool） | ≥ `PSEUDO_MIN_DOMINANT_FRACTION` = 0.90 | 即 10 审 ≥ 9 同标签；Evaluation 标签**不计入** |
| **OOF 覆盖率（v4 新增）** | ≥ `PSEUDO_MIN_OOF_CLUSTER_COVERAGE` = **0.80** | `cluster_oof_coverage = 合法 OOF 预测脉冲数 / Train-side Cluster 脉冲数`；不足 → `INSUFFICIENT_OOF_COVERAGE`，禁止传播，**绝不 fallback Production** |
| **Cluster-wide OOF consistency（v4 重定义）** | == `HIGH_CONSISTENCY` | = 整个可合法 OOF 评分的 Train-side Cluster 上 `OOF top1 == dominant_species` 的比例与平均 P(dominant)（阈值复用 `MODEL_COMPARE_*`，经 `determine_cluster_oof_consistency`）；Production 模型不参与 |
| 主导标签 | ≠ `SPECIES_UNCERTAIN_LABEL`、≠ 空 | 不确定标签不可传播 |

**Cluster-wide OOF 算法（修复代表样本偏置）**：

- **分母（eligible）**：Train-side Cluster 中**有合法 OOF 预测**的脉冲
  （`oof_available=True` 且 `oof_top1_label` 非空；缺特征行无合法预测，
  不进分母）；**分子（agree）**：其中 `OOF top1 == dominant_species` 的条数；
  `cluster_oof_agreement = agree / eligible`；
- **合法 OOF prediction（I9）**：评分模型训练时**完全没见过**该 pulse 所属
  `source_recording`（source_recording-level grouped cross-fitting，
  **禁止随机 pulse KFold**）；多录音 Cluster 中每个录音由**自己的**
  unseen fold 模型评分后汇总，**绝不**用 Production / 某个统一模型
  预测整个 Cluster；
- **coverage 分母** = Train-side Cluster **全部**脉冲（含 reviewed +
  未审；不含 Evaluation 录音脉冲），分子 = eligible 数；
- **v3 代表样本偏置的修复**：10/10 representatives 为 A 的 500 脉冲簇，
  若全簇 OOF 是 300A/200B（agreement 0.60），**绝不**判 HIGH——
  representatives 只提供 **Expert Truth**（dominant species / purity），
  一致性统计必须走全簇。

**每簇输出字段**（`pseudo_cluster_gates.csv`，审计必留）：
`cluster_id` / `cluster_total_train_side_pulses` / `cluster_oof_eligible_pulses` /
`cluster_oof_coverage` / `cluster_oof_agree_count` / `cluster_oof_disagree_count` /
`cluster_oof_agreement` / `cluster_oof_status`（另有 reviewed / purity /
core_radius / radius_fallback / passed / gate_reason / recommended_action）。

**Representatives 的新职责（v4）**：只提供**专家真值**（谁是 dominant、
纯度证据、core radius 证据、局部邻域证据），**不再**充当簇一致性的
统计样本。

**纯度语义（防误读）**：`dominant_fraction ≥ 0.90` 只证明**专家触及的局部
证据高度一致**，绝不等于"整个 Cluster 已被确认"。因此传播仍然被
core radius（§4.2）与局部专家邻域（§4.3.5）双重圈住，且每条伪标签
保留 `propagated_from` 专家 pulse 清单供审计。
**即使 Cluster-wide OOF 一致性 HIGH，它也只是"模型一致性"证据，
绝不是专家真值**（见 §17 残留风险）。

状态 → 动作映射（写入 `pseudo_cluster_gates.csv: recommended_action`）：

| `cluster_oof_status` | 自动反哺 | recommended_action |
|---|---|---|
| `HIGH_CONSISTENCY` | ✅ 允许进入样本级筛选 | `auto_pseudo` |
| `MODERATE_CONSISTENCY` | ❌ | `suggest_review`（人工补审队列） |
| `LOW_CONSISTENCY` | ❌ | `manual_review`（人工复核） |
| `MIXED` | ❌ | `manual_review` |
| `INSUFFICIENT_SAMPLES` | ❌ | `suggest_review` |
| `INSUFFICIENT_OOF_COVERAGE`（v4） | ❌ | `manual_review`（OOF 覆盖不足，禁止传播） |
| `NO_MODEL` | ❌ | `no_oof_scoring`（先修复 OOF 评分；绝不 Production 兜底） |

额外规则：

- 主导占比在 0.90 ~ 1.00 之间时，簇内**已被审为其他标签的样本**
  本身永远保持专家标签（不做局部排除的复杂逻辑——不达标就整簇跳过）；
- `noise` 状态样本不参与传播，也不被传播覆盖；noise 类扩充仍走
  `noise_gate` 既有路径；
- **只有代表样本可被专家审核**（`annotations.py` 从 `representatives.csv`
  初始化），因此 `PSEUDO_MIN_REVIEWED=10` 意味着默认 20 条代表样本中
  至少一半要审完；不够时用既有 `--add-samples`（`CURATION_ADD_SAMPLES=10`）
  补充代表样本再审。这是 v2 有意提高的人工成本（D8）。
- `CURATION_MIXED_MIN_REVIEWED=5` 的"可能为混合簇"提示保持不变
  （它是早期提醒，与传播门控用途不同，两者互不影响）。

### 4.2 核心区半径（core radius）

`r_cluster` = 簇内"已审 + 标签=主导标签 + `selection_reason` 不含
`boundary|outlier`"样本的 `distance_to_center` 的
P`PSEUDO_CORE_DISTANCE_QUANTILE`（默认 P95）；该类样本不足
`PSEUDO_CORE_MIN_RADIUS_SAMPLES`（3 条）时退化为"全簇距离 P25"
（宁窄勿宽）并在 gate 表打 `radius_fallback` 标记。

### 4.3 样本级安全闸门（Safe Candidate Pool 准入）

每条未审 accepted 脉冲**按序**判定（首个不满足项记为 rejection_reason）：

| # | 闸门 | 拒绝原因 |
|---|---|---|
| 1 | 簇门控通过（§4.1，**第一层** Cluster-level） | `insufficient_reviewed` / `dominant_fraction_too_low` / `insufficient_oof_training_data` / `insufficient_oof_coverage` / `not_high_consistency` |
| 2 | 不在 `expert_annotations.csv` 任何状态（reviewed/pending → `expert_sample`；uncertain → `uncertain_sample`；噪声 → `noise_sample`） | `expert_sample` / `uncertain_sample` / `noise_sample` |
| 3 | `selection_reason` 不含 `boundary|outlier`（代表样本） | `boundary_or_outlier_sample` |
| 4 | `distance_to_center ≤ r_cluster` | `outside_core_radius` |
| 5 | **局部专家邻域无冲突（v3 新增，§4.3.5）** | `local_expert_conflict` |
| 6 | OOF 概率可用（fold 有效且特征齐全，见 §5） | `oof_probability_unavailable` / `insufficient_oof_training_data` |
| 7 | OOF top1 == 簇主导标签 | `model_top1_mismatch` |
| 8 | `P(top1) ≥ threshold`（校准 0.90 / 未校准 0.95；per-class 覆盖见 §4.5） | `low_model_probability` |
| 9 | `top1 − top2 ≥ PSEUDO_MIN_PROBABILITY_MARGIN`（默认 0.20） | `low_probability_margin` |

> **#6–9 是第二层 Candidate-level OOF scoring**（§4.1 两层分离表）：
> 只回答"这一条 pulse 自己的合法 OOF 预测是否值得采信"，与第一层
> "簇整体一致性"分开记录、分开判定，绝不混为一个 probability。
> Evaluation 录音的脉冲**不进入候选宇宙**（I13/I16，连逐条结论表都不进）。

全部通过 → 进入 **safe_candidate_pool**（此时仍不是训练样本，
还要过 §6 多样性抽样与配额）。

### 4.3.5 局部专家邻域检查（v3 新增：防大簇内部异质小区域）

**风险场景**：一个大簇整体 `dominant_fraction ≥ 0.90`、OOF 一致性 HIGH，
但簇内存在一个专家证据显示为 B 的局部小区域（如 30×A + 2×B 的簇里
一块"B 区"）。簇级门控看不见它——纯度 20/22 ≈ 0.909 依然放行，
B 区候选若混进伪标签就把 A 标签传播到 B 区（用户风险 #1）。

**规则**：每条候选取**同簇 Expert Train Pool 已审样本**中特征空间
（StandardScaler 后）最近的 K=5 个邻居，只看
`PSEUDO_LOCAL_EXPERT_RADIUS_FACTOR × core_radius`（默认 1.0×）以内的：

| 情形 | `local_expert_check` | 判定 |
|---|---|---|
| 邻居数 < `PSEUDO_LOCAL_EXPERT_MIN_NEIGHBORS`（3） | `insufficient` | **不硬判**：记录后继续走其余闸门（保守默认；升级 hard gate 见 §17） |
| 局部主导标签 == 簇主导标签 且一致率 ≥ `PSEUDO_MIN_LOCAL_EXPERT_AGREEMENT`（0.80） | `pass` | 继续 |
| 否则（局部异类可见） | `conflict` | 拒绝：`local_expert_conflict` |

用户示例对齐：最近 5 个专家样本 `A A A A B` → 一致率 0.80 → **通过**
（边界恰在阈值上，采用 `≥`）；`A A A B B` → 0.60 → 拒绝。
邻居标签 / 一致率写入候选表（`local_expert_neighbor_count` /
`local_expert_agreement`），可审计。Evaluation Set 样本**不是**合法邻居（I13）。

### 4.4 margin 闸门的数学说明（v2 调研结论）

已确认 `predict_pulse_frame` 输出**完整类别概率分布**
（`all_class_probabilities`，JSON，校准模型为校准分布），因此
top1 / top2 / margin 均可如实计算，无需伪造。

但必须如实记录一个推导：对归一概率分布，

```text
P(top1) ≥ t  ⟹  top2 ≤ 1 − t  ⟹  margin = top1 − top2 ≥ 2t − 1
```

即 t = 0.90 时 **margin ≥ 0.80 恒成立**，`margin ≥ 0.20` 在当前全局阈值下
**不构成有效约束**（防御性断言，拦截概率列损坏/分布不归一的异常数据）。
同理 `top1 == dominant` 也被 `P(dominant) ≥ 0.90` 隐含（t ≥ 0.5 时）。

含义：

- 用户示例中 `0.96/0.02`（margin 0.94）与 `0.91/0.08`（margin 0.83）之别，
  在 t=0.90 下**无法**用 0.20 的 margin 区分；若要区分，等效手段是
  提高 top1 阈值（如 0.95）或将 margin 设为 ~0.85；
- margin 闸门真正开始"咬合"的时刻，是 §4.5 的 **per-class 阈值**
  被调低到 0.60 以下（此时 `2t−1 < 0.20`，margin 成为兜底护栏）。

因此 v2 决定：**实现 margin 闸门，默认值 0.20（下限护栏）**，同时在
config 注释与本节记录上述推导与两个更强的调节杠杆（top1 阈值 / margin≥0.85），
供后续按数据实测选择。`top2_probability` 一并落表（§7.3），便于回看。

### 4.5 per-class 阈值预留（第一版不实现自动学习）

第一版保持全局 `PSEUDO_MIN_MODEL_PROBABILITY = 0.90`，但**数据结构与接口
不写死全局唯一**：

- config 预留 `PSEUDO_CLASS_THRESHOLDS: dict = {}`（如
  `{"species_A": 0.88, "species_B": 0.96}`）；
- 解析规则：`threshold(label) = PSEUDO_CLASS_THRESHOLDS.get(label, PSEUDO_MIN_MODEL_PROBABILITY)`；
- `run_metadata.json` 记录实际生效的逐类阈值快照；
- 候选表保存 `applied_threshold` 列，保证可回放。

**明确声明：global 0.90 是初始默认策略，不是理论假设。** 未来按
expert-only validation 上每个物种的 precision / calibration 表现学习阈值，
目标是"找到使该物种 pseudo label precision 达到目标水平的最低 probability
threshold"。第一版不实现自动 threshold learning。

---

## 5. OOF 评分设计（recording-level cross-fitting，I9 的实现）

### 5.1 问题

若评分模型训练时见过 `recording_A` 的 20 条专家 pulse，再给 `recording_A`
剩余 300 条未审 pulse 打 `P(Species X)`，模型可能记住的是该录音的背景噪声、
设备差异、个体/环境/session 特征，而不是物种特征——概率虚高，形成
**同录音泄漏式自我确认**。

### 5.2 构造（复用既有机制）

```text
1. 专家训练帧 E（仅 reviewed + expert_label 非空；绝不含伪标签，I10）
2. groups = source_recording_series(E)          # 复用既有分组键推导
3. if nunique(groups) < PSEUDO_OOF_MIN_RECORDINGS(=4):
       → OOF 不可用（全候选 oof_probability_unavailable，零传播）
4. k = clamp(2, min(PSEUDO_OOF_FOLDS(=5), nunique(groups)))
   StratifiedGroupKFold(n_splits=k, shuffle=True,
                        random_state=PSEUDO_OOF_RANDOM_STATE)
5. 对每个 fold f（fold 模型 = 角色 B）：
   - 训练集 = E 中【非 f 折】录音的样本；
     评分对象 = f 折录音名下全部 safe-candidate 候选
   - fold 训练集类别 < 2 或任一类 < SPECIES_MIN_CLASS_SAMPLES
     → fold 无效，其名下候选全部 oof_probability_unavailable
   - fold 内校准：在 fold 训练集上再做一层
     StratifiedGroupKFold(groups=source_recording) 生成显式 (train, calib)
     索引对 → 复用 species_model._fit_calibrated；
     校准条件不满足（沿用 build_group_splits 的校准 fold 校验规则）时
     fold 模型不校准 → probability_kind=model_support，阈值提高到 0.95
6. 每条候选恰好被一个 fold 模型评分一次：
   其 source_recording 只出现在该 fold 的【评分侧】，绝不进其训练集
```

**不做普通随机 KFold**：group 恒为 `source_recording`，与
`build_group_splits` / `StratifiedGroupKFold` 的既有约定一致。

### 5.2.1 Cluster-wide 打分宇宙（v4 新增）

一次 `oof.predict` 对**整个 Train-side 簇脉冲宇宙**打分（含未审候选与
Train Pool 专家行；Evaluation 录音脉冲被排除，I13/I16），同一份预测
同时服务两层判断（§4.1）：

- 第一层用它算 `cluster_oof_agreement / cluster_oof_coverage`
  （分母 = 有合法 OOF 预测的 Train-side 簇脉冲；coverage 分母 = 全部
  Train-side 簇脉冲）；
- 第二层用其中**该候选自己的**那一行做样本级判定。

训练侧准入：`build_oof` 前统一 `assert_eval_isolation`
（OOF 训练录音 ∩ Evaluation 录音 = ∅，否则 `EVAL_CONTAMINATED` 立即终止）。

### 5.3 不可用时的行为（I11）

以下情形一律 `oof_probability_unavailable`，**拒绝传播**：

- 独立录音数 < `PSEUDO_OOF_MIN_RECORDINGS`；
- fold 训练集类别不足；
- 候选特征缺失（`predict_pulse_frame` 对该行无有效概率）。

**绝不回退**到 Production 模型概率（I11）；UI 需明确显示
"OOF 评分不可用，本轮无伪标签"，而不是静默换模型。

### 5.4 OOF 出处链（每条候选必填）

- `scoring_model`：fold 模型标识（如 `oof_fold_2_of_3`）；
- `scoring_model_version`：SHA-256（fold 训练 pulse_id 集 + config 阈值快照
  + random_state），保证事后可重建当时的评分器；
- `oof_fold_id`、`oof_train_recordings`（或其 hash）：证明 I9 成立；
- `model_probability` / `probability_kind` / `top1_label` /
  `top1_probability` / `top2_probability` / `probability_margin` /
  `applied_threshold`。

---

## 6. Safe Pool → Diversity Sampling（两阶段）

### 6.1 Stage A：Safe Candidate Pool

即 §4.3 全部闸门通过的集合。**safe first**：先保证标签可信，
再考虑训练信息增量。

### 6.2 Stage B：多样性抽样（simple / deterministic / reproducible）

不用 k-center greedy、farthest point、DPP、submodular 等复杂算法。
选定方案：**时间桶去冗余 + 录音限额 + 距离分层轮转**，全部确定性、无 RNG。

```text
输入：safe_candidate_pool、簇配额 Q（§6.3）
1. normalized_core_distance = distance_to_center / core_radius ∈ (0, 1]
   分层：S1=(0,1/3]   S2=(1/3,2/3]   S3=(2/3,1]
   （样本全部已在核心区内部，分层只为避免挤在 center 附近）
2. 层配额：非空层均分 Q（最大余数法；余数按 S1→S2→S3 分配——宁内勿外）
3. 层内选择顺序（确定性）：
   录音按 source_recording 字典序【轮转】（round-robin），
   录音内按 (distance_to_center, pulse_id) 升序
4. 选择时跳过并记录：
   与已选样本同 (source_recording, floor(start_s / PSEUDO_TIME_BUCKET_S))
       → temporal_diversity_filtered
   （同一录音 1 秒内最多 PSEUDO_MAX_PER_TIME_BUCKET=1 条）
   该录音该类已选数 ≥ PSEUDO_MAX_PER_RECORDING_PER_CLASS
       → recording_quota_full
5. 层配额用满 → 该层剩余记 distance_stratum_quota_full；
   有层未用满时，剩余配额按 S1→S2→S3 回流补满；
   Q 全部用尽 → 余下候选记 cluster_quota_full
```

要点：

- **时间桶去冗余**直接解决"同一录音时间极近、特征近乎相同的
  连续几十个 pulse 全进训练集"的问题（桶边界跨越的相邻样本最多造成
  每桶边 1 条冗余，可接受；`PSEUDO_TIME_BUCKET_S` 可调）；
- **录音轮转 + 限额**保证伪标签不被单条录音垄断；
- **距离分层**保证样本覆盖核心区全深度而非全部贴着质心；
  比例由"非空层均分"自动决定，不写死 40/35/25；
- 全过程无随机数，`random_state` 只影响 OOF fold 划分（记录在案）。

### 6.3 配额

- 簇配额 `Q = min(PSEUDO_MAX_PER_CLUSTER, 已审样本数 × PSEUDO_MAX_PER_CLUSTER_FACTOR)`；
- 每录音每类 ≤ `PSEUDO_MAX_PER_RECORDING_PER_CLASS`；
- 每录音每时间桶 ≤ `PSEUDO_MAX_PER_TIME_BUCKET`；
- 类别级护栏（§8.3）在训练前再校验一次。

---

## 7. 数据结构与前置改动

### 7.1 前置改动（实现顺序第一项）

⑥ 聚类阶段目前**不持久化** `StandardScaler` 与 `KMeans`，且
`clustered_pulses.csv` 只有 `raw_cluster / cluster_id / pca_1 / pca_2`，
只有代表样本在 `representatives.csv` 里带 `distance_to_center`（标准化空间）。
伪标签筛选需要**全量脉冲**的标准化距离：

- **方案 A（推荐）**：`stage_cluster` 落盘 `cluster_model.joblib`
  （scaler + kmeans + 特征列顺序 + `random_state`），并在
  `clustered_pulses.csv` 增加 `distance_to_center` 列（统一标准化空间）；
- 方案 B：只加距离列，不落盘模型（`--add-samples` 场景无法复算）。

**距离口径必须唯一**：所有 `distance_to_center` 都是 StandardScaler 空间
欧氏距离，与 `representatives.py::_distance_matrix` 一致。

### 7.2 输出表

```text
results/curation/<run_id>/tables/
    pseudo_cluster_gates.csv        # 每簇一行：纯度/半径/状态/recommended_action
    pseudo_label_candidates.csv     # 全部候选逐条判定（含落选原因，见 §12）
    pseudo_training_dataset.csv     # 通过全部闸门 + 多样性抽样的伪标签训练集
```

### 7.3 `pseudo_training_dataset.csv` / candidates 表出处链列（I6）

```text
pulse_id, cluster_id, pseudo_label, label_source='pseudo',
expert_label='', review_status='',            # 永远为空，防止冒充
audio_hash, source_file, source_path, source_recording, start_s, end_s,
duration_ms, snr_db,
distance_to_center, core_radius, normalized_core_distance,
cluster_purity,                                # 该簇 dominant_fraction
propagated_from,                               # 定标签/半径的专家 pulse_id 列表（; 分隔）
scoring_model, scoring_model_version,          # OOF fold 模型标识与内容哈希
oof_fold_id, oof_train_recordings_hash,        # cross-fit 信息（证明 I9）
model_probability, probability_kind,
top1_label, top1_probability, top2_probability, probability_margin,
applied_threshold,
selection_round,                               # 多样性抽样轮转步号
pseudo_round, selected_at,
*CLUSTER_FEATURE_COLUMNS
```

- `pseudo_label` 与 `expert_label` 是两个不同的列；
- `source_recording` 沿用 `species_model.py::source_recording_series` 推导规则；
- 这套出处链必须能回答："**为什么这个 pulse 会被自动标成这个物种？**"

`run_metadata.json` 追加 `pseudo_feedback` 段：所用阈值快照（含 per-class
覆盖）、OOF fold 明细（每折训练录音）、轮次、通过/落选计数、随机种子。

---

## 8. 训练协议

### 8.1 带权训练（v2 保持不变）

- 专家样本 `sample_weight = 1.0`；伪标签 `PSEUDO_SAMPLE_WEIGHT = 0.4`；
- RandomForest 经 `sample_weight` 传入；`class_weight="balanced_subsample"`
  保留（权重与类别平衡正交）；
- **第一阶段优先验证 pseudo feedback 是否真有收益**，不做
  distance/probability/purity 相关的动态权重（见 §17 Future Work）。

### 8.2 校准与评估隔离（比 v1 更严）

- 概率校准只用**专家样本**所在的校准 fold（I4/I10）；
- 评估只统计 `label_source=expert` 的行；伪标签永不进 evaluation /
  calibration truth；
- `metadata` 记录 `eval_labels: expert_only`；导入模型校验发现评估混入
  伪标签 → 标记 `EVAL_CONTAMINATED` 并拒绝对外展示指标。

### 8.3 类别配比护栏

- 某类伪标签数 > 该类专家样本数 × `PSEUDO_MAX_PER_CLUSTER_FACTOR` → 截断；
- 训练前记录专家/伪标签逐类数量表（沿用 `_warn_class_imbalance` 风格）。

---

## 9. 迭代闭环与采纳

### 9.1 循环

```text
v(1) = 仅专家样本训练
for round = 1 .. PSEUDO_MAX_ROUNDS（默认 2）:
    用 OOF 评分体系重新筛选伪标签（概率随专家数据变化，每轮重算）
    训练 v(round+1)（专家 + 本轮伪标签，带权）
    固定专家留出集上对比 v(round) 与 v(round+1)
    采纳条件全过 → 采纳 v(round+1)，进入下一轮
    任一不过   → 保留 v(round)，终止循环并输出对比报告
```

### 9.2 采纳硬门槛（不变）

| 指标（专家留出集） | 条件 |
|---|---|
| Macro F1 | 不下降（Δ ≥ 0） |
| 任一类 Recall | 下降不超过 `PSEUDO_MAX_RECALL_DROP` = 0.02 |
| log loss | 不上升超过 2% |
| 训练侧 | 无 `EVAL_CONTAMINATED`、`SPECIES_MIN_CLASS_SAMPLES` 仍满足 |

### 9.3 稳定性报告（v2 新增，第一版为诊断项而非硬门槛）

专家样本足够时（≥ `PSEUDO_STABILITY_MIN_EXPERT_SAMPLES`=60 且
录音 ≥ `PSEUDO_STABILITY_MIN_RECORDINGS`=8），对 v(n) 与 v(n+1) 各跑
`PSEUDO_STABILITY_REPEATS`=5 次 GroupShuffleSplit 重复评估
（每次 holdout 录音不同；**每次重复训练时，holdout 录音名下的伪标签
必须从该次训练中剔除**，否则稳定性数字本身泄漏），报告：

```text
Old: Macro F1 = 0.872 ± 0.018
New: Macro F1 = 0.884 ± 0.015
```

- 数据太少时**跳过**并注明，绝不为做 repeated CV 把结构搞重；
- 第一版不作为 adoption hard gate：**固定 expert holdout 仍是主闸门**；
- 稳定性差（±过大）时 UI 提示"结论不稳定，建议补充专家标注"。

### 9.4 防 confirmation bias（v2 强化）

- 轮次硬上限 2；每轮重筛，旧概率失效即落选；
- OOF 评分（I9）消除同录音自我确认；OOF 模型只吃专家样本（I10）
  防止伪标签逐轮放大；
- MODERATE/LOW/MIXED/INSUFFICIENT 簇进入人工补审队列（主动学习入口）；
- 伪标签可整批作废重来（删 `pseudo_*.csv` 即回纯专家训练）。

---

## 10. 配置参数（`config.py` 新增 §10）

```python
# =========================================================
# 10. 伪标签反哺（Cluster 核心区 → 物种模型自训练）
# =========================================================
PSEUDO_MIN_REVIEWED = 10                   # 簇内最少已审样本（v2: 5→10）
PSEUDO_MIN_DOMINANT_FRACTION = 0.90        # 簇纯度门槛（10 审 ≥ 9 同标签）
PSEUDO_CORE_DISTANCE_QUANTILE = 0.95       # 核心区半径 = 已审一致样本距离 P95
PSEUDO_CORE_MIN_RADIUS_SAMPLES = 3         # 定半径样本不足时退化为全簇 P25

PSEUDO_OOF_MIN_RECORDINGS = 4              # 录音数不足 → OOF 不可用 → 零传播
PSEUDO_OOF_FOLDS = 5                       # cross-fitting 折数上限
PSEUDO_OOF_RANDOM_STATE = RANDOM_STATE

PSEUDO_MIN_MODEL_PROBABILITY = 0.90        # 全局默认策略（非理论假设）
PSEUDO_MIN_MODEL_PROBABILITY_UNCALIBRATED = 0.95
# margin 在 t=0.90 时数学上恒 ≥ 2t-1=0.80，本值是下限护栏/异常数据拦截器；
# 若需区分"0.96/0.02 vs 0.91/0.08"，应提高 top1 阈值或将 margin 设 ~0.85。
PSEUDO_MIN_PROBABILITY_MARGIN = 0.20
PSEUDO_CLASS_THRESHOLDS = {}               # 预留 per-species 阈值覆盖（v1 不学习）

PSEUDO_MAX_PER_CLUSTER_FACTOR = 5          # 每簇伪标签 ≤ 已审数 × 5
PSEUDO_MAX_PER_CLUSTER = 500
PSEUDO_MAX_PER_RECORDING_PER_CLASS = 30
PSEUDO_TIME_BUCKET_S = 1.0                 # 同录音时间桶去冗余
PSEUDO_MAX_PER_TIME_BUCKET = 1
PSEUDO_DISTANCE_STRATA = 3                 # 核心区分层数

PSEUDO_SAMPLE_WEIGHT = 0.4                 # 伪标签训练权重（v2 保持）
PSEUDO_MAX_ROUNDS = 2
PSEUDO_MAX_RECALL_DROP = 0.02

PSEUDO_STABILITY_MIN_EXPERT_SAMPLES = 60   # 稳定性报告触发门槛
PSEUDO_STABILITY_MIN_RECORDINGS = 8
PSEUDO_STABILITY_REPEATS = 5
```

所有阈值（含 per-class 覆盖的实际生效快照）进 `run_metadata.json`，
保证任何一次导出可复现。

---

## 11. 设计决策记录

| # | 决策 | 理由 / 被否方案 |
|---|---|---|
| D1 | 不用"簇内最近 10%"等固定比例做准入 | 固定比例随簇大小漂移；改为距离半径 + 纯度 + OOF 概率闸门 |
| D2 | 伪标签走独立文件/列 | 保住 I1/I2 可审计性 |
| D3 | 半径取"已审一致样本距离 P95" | 半径必须由有标签证据定义；全簇分位会把杂簇半径撑大 |
| D4 | boundary/outlier 不参与定半径、不入池 | 簇边缘/异常尾部与"核心区传播"自相矛盾 |
| D5 | 校准（含 OOF fold 内校准）只用专家样本 | 校准概率语义必须是对专家真值的置信 |
| D6 | 采纳门槛用 Macro F1 + 单类 recall | 类别不平衡时 Accuracy 会掩盖少数类被挤掉 |
| D7 | 轮次上限 2 | 边际收益陡减而 confirmation risk 累积 |
| **D8** | **簇最少已审 5 → 10** | 5 个专家样本不足以可靠推断簇纯度；第一版宁可少产 pseudo。配套：`--add-samples` 补审 |
| **D9** | **自动反哺 = HIGH_CONSISTENCY only** | MODERATE 降为人工补审入口；不为扩大伪标签数量降低门槛 |
| **D10** | **评分概率必须 recording-level OOF** | 同录音自我确认是最大的概率虚高来源（I9）；被否：直接用 Production 模型概率 |
| **D11** | **Safe Pool + 两阶段多样性抽样** | nearest-first 会产出大量高度重复样本、训练信息增量低 |
| **D12** | **margin 闸门保留、默认 0.20** | 调研结论：t=0.90 下 margin ≥ 0.80 恒成立，0.20 不咬合（防御性断言）；真正杠杆是 top1 阈值/per-class 阈值，见 §4.4 |
| **D13** | **全局 0.90 只是初始策略** | 接口/落表预留 per-class 阈值；未来按 expert-only validation 学习 |
| **D14** | **pseudo 权重固定 0.4** | 第一阶段先验证收益；动态权重仅进 Future Work |
| **D15** | **稳定性报告仅诊断** | 专家数据普遍偏少，repeated CV 不做硬门槛；固定 holdout 仍是主闸门 |

---

## 12. rejection_reason 完整清单

按判定顺序（每条落选候选记录**首个**命中原因；簇级原因带 gate 表明细）：

| 顺序 | rejection_reason | 触发 |
|---|---|---|
| 1 | `insufficient_reviewed` | 簇已审 < 10 |
| 2 | `dominant_fraction_too_low` | 簇纯度 < 0.90 或无主导标签（Evaluation 标签不计入） |
| 3 | `insufficient_oof_training_data` | OOF 评分体系不可用（录音 < 4 / fold 全无效）→ 整簇零传播（I11，绝不 Production 兜底） |
| 3 | `insufficient_oof_coverage`（v4） | `cluster_oof_coverage < 0.80`（合法 OOF 预测覆盖不足）→ 禁止传播（I18） |
| 3 | `not_high_consistency` | `cluster_oof_status` ≠ HIGH_CONSISTENCY（含 MODERATE/LOW/MIXED/INSUFFICIENT_SAMPLES；具体状态另列 `cluster_oof_status`） |
| 4 | `expert_sample` | 已在专家标注表（reviewed / pending） |
| 4 | `uncertain_sample` | 专家标为 uncertain |
| 4 | `noise_sample` | 专家标为噪声 |
| 5 | `boundary_or_outlier_sample` | selection_reason 含 boundary/outlier |
| 6 | `outside_core_radius` | 距离 > r_cluster |
| 7 | `local_expert_conflict`（v3） | 局部专家邻域异类可见（`local_expert_check=conflict`，§4.3.5） |
| 8 | `oof_probability_unavailable` | OOF 评分行缺失（特征缺失 / 折不匹配）（绝不回退 Production 概率） |
| 8 | `insufficient_oof_training_data` | OOF 训练数据不足（录音 < 4 / fold 无效）→ 整簇零传播（I11） |
| 9 | `model_top1_mismatch` | OOF top1 ≠ 簇主导标签 |
| 10 | `low_model_probability` | P(top1) < applied_threshold |
| 11 | `low_probability_margin` | top1 − top2 < margin 下限 |
| 12 | `temporal_diversity_filtered` | 同 (录音, 时间桶) 已有更优样本 |
| 13 | `recording_quota_full` | 该录音该类配额用尽 |
| 14 | `distance_stratum_quota_full` | 该距离层配额用尽 |
| 15 | `cluster_quota_full` | 簇配额用尽 |

要求：**所有候选为什么没有成为伪标签训练样本，都必须可追踪**
（漏斗计数落 `run_metadata.json`，逐条原因落 candidates 表）。

---

## 13. UI 与 CLI

### 13.1 ⑥ 页面

**子面板「专家数据划分」（v3 新增，位于「已确认物种模型」之前）**：

- 总量 / Train Pool / Evaluation Set（录音数 + pulse 数）+ 逐类表
  （Evaluation = 0 的物种带 tooltip"无评估真值（Support = 0）：
  无法计算，不是模型表现为 0"）；
- Evaluation 目标比例微调框（默认 0.20，注明"真正拆分单位是 source_recording"）；
- **生成专家数据划分**：先弹**预览对话框**（两侧/逐类/录音数/覆盖警告）
  → 确认才落盘冻结；取消则不写任何文件；
- **重新生成划分**：强警告"重新划分后，新旧模型指标将不再与之前
  evaluation version 完全可比"→ `split_version + 1`；
- `INSUFFICIENT_EVAL_COVERAGE` 提示（"绝不拆同一录音来凑 F1"）；
- **查看评估表**：Species / Support / Precision / Recall / F1 / 状态，
  表头列均有 tooltip（Support/Precision/Recall/F1/Macro-F1 释义）；
  Macro-F1 标注参与平均的物种与被排除的零真值物种；
- **新旧模型对比**：`compare_models_on_gold_evaluation`——**两个模型
  同时在同一个 Fixed Gold Evaluation 上重新评估**后对比 Macro-F1 Δ +
  逐类 recall 下降（真值取冻结快照）；**禁止**拿旧模型缓存的
  `expert_evaluation.csv` 与新模型现算结果直接比较（§2.5.5）。

**子面板「伪标签反哺」**：

- 门控预览（读 `pseudo_cluster_gates` 口径实时计算）：
  "可自动反哺簇 N 个 / 建议人工补审 M 个（MODERATE）/ 需人工复核 K 个
  （LOW/MIXED）/ 样本不足 J 个（INSUFFICIENT，需 ≥10 审/簇）"；
- 按钮 **生成伪标签训练集**：需 OOF 评分可用（录音数 ≥ 4）；显示筛选漏斗
  （候选 → 簇门控 → 样本闸门 → 多样性 → 通过）；OOF 不可用时明确提示
  "OOF 评分不可用，本轮无伪标签"，**不提供'改用当前模型概率'的勾选项**（I11）；
- 按钮 **带权训练 v(n+1) 并对比**：展示专家留出集对比表 + 稳定性报告
  （若触发）+ 采纳/未采纳原因；
- 采纳后模型信息标注
  `training_labels: expert + pseudo(round=k, n=..., weight=0.4)`；
- **人工补审队列**视图（M3）：按 recommended_action 与簇大小排序，
  一键跳转审核工作台（复用 `curation_review.py`）。

### 13.2 命令行（与 ⑥ 同一口径）

```powershell
python -m bat_classifier.curation --run-dir results\curation\<run_id> `
    --pseudo-export                                  # M1：gates + candidates
python -m bat_classifier.curation --run-dir results\curation\<run_id> `
    --pseudo-round 2 --model models\species\<model_name>   # M3：+训练+对比
```

---

## 14. 失败模式与对策

| 失败模式 | 对策 |
|---|---|
| 同录音泄漏式自我确认 | I9：OOF 评分模型绝不含被评记录音（§5）；OOF 不可用零传播（I11） |
| 混合簇/杂簇被传播 | 10 审 ≥ 9 同标签 + HIGH_CONSISTENCY only + MIXED 零传播 |
| 小样本簇误传播 | `PSEUDO_MIN_REVIEWED=10`（D8）；不足则 `insufficient_reviewed` 并进补审队列 |
| 大量高度重复 pseudo | 时间桶去冗余 + 录音轮转/限额 + 距离分层（§6.2） |
| 模率虚高（同上/未校准） | OOF + 未校准阈值 0.95 + `probability_kind` 落表 + margin 断言 |
| 单条录音垄断伪标签 | 每录音每类上限 + 轮转选择 |
| 少数类被挤掉 | 采纳门槛含单类 recall + 类别配比护栏 |
| 伪标签泄漏进评估 | 评估只统计 expert 行 + `EVAL_CONTAMINATED` 校验；稳定性重复评估逐次剔除 holdout 录音的伪标签 |
| 专家后来改判某簇 | `propagated_from` 反查受影响伪标签；重跑即全量重筛 |
| 阈值拍脑袋 | 全部快照落 metadata；候选表含 applied_threshold 与落选原因，可离线回放调参 |

---

## 15. 测试计划

`tests/test_pseudo_labels.py`、`tests/test_oof_scoring.py`、
`tests/test_expert_split.py`、`tests/test_evaluation_lifecycle.py`（v4 新增）
+ `tests/test_species_model.py`（扩展）。

**v4 验收 Test 1–10（当前编号，已实现，全绿）**：

| # | 验收项 | 实现位置 |
|---|---|---|
| 1 | **Evaluation immutable**：模型训练 / pseudo 轮次 / 新标注均不改 Gold Evaluation v1（`gold_eval_v1.json` 逐字节不变；v2 另存不覆盖） | `test_evaluation_lifecycle.TestGoldEvaluationImmutable` |
| 2 | **Pseudo 不能变 Evaluation**：`label_source=pseudo`（含"人工只确认模型预测"）永远进不了 Evaluation | `test_evaluation_lifecycle.TestGoldEvaluationImmutable` |
| 3 | **Evaluation 排除于一切训练侧**：训练/校准/OOF 训练/purity/core radius/局部邻域/pseudo dataset 全部无 Evaluation（含反向泄漏探针） | `test_evaluation_lifecycle.TestEvaluationExcludedFromTraining` |
| 4 | **EVAL_CONTAMINATED**：评估录音混入训练侧 → 立即终止并列出 `source_recording` | `test_evaluation_lifecycle.TestEvalContaminated` |
| 5 | **同一 Evaluation 对比**：旧+新模型都在 Gold Evaluation 上重新评估（缓存指标无效；冻结真值覆盖活标注） | `test_evaluation_lifecycle.TestSameGoldEvaluationComparison` |
| 6 | **代表样本偏置**：500 脉冲簇 reps 10/10=A、全簇 OOF 300A/200B → **不得** HIGH | `test_pseudo_labels.TestClusterWideOOFConsistency.test_representative_bias_no_high_consistency` |
| 7 | **全簇一致**：reps 10/10 A + 全簇 OOF 一致 + 覆盖充足 → 可 HIGH 并传播 | `...test_high_whole_cluster_consistency` |
| 8 | **OOF 覆盖不足**：500 簇仅 ~60 条合法 OOF（即使全对）→ `INSUFFICIENT_OOF_COVERAGE`，零传播 | `...test_insufficient_oof_coverage_blocks_high` |
| 9 | **禁止 Production fallback**：覆盖不足 + Production P=0.999 → 依旧零传播 | `...test_no_production_fallback_when_coverage_low`（+ 既有 `TestNoProductionFallback`） |
| 10 | **多录音簇 grouped OOF**：每个 recording 的预测来自没见过它的 fold 模型（出处链可断言） | `...test_multi_recording_cluster_grouped_oof` |

**v3 验收 Test 1–10（旧编号，保持全绿）**：

| # | 验收项 | 实现位置 |
|---|---|---|
| 1 | 按 recording 划分，同一录音绝不两侧 | `test_expert_split.TestRecordingLevelSplit` |
| 2 | Evaluation 不进入训练（训练指纹零评估录音/pulse） | `test_expert_split.TestEvalContamination` |
| 3 | Evaluation 不进入 pseudo gate（纯度只用 Train Pool，eval 反例会翻盘） | `test_pseudo_labels.TestEvalNotInGates` |
| 4 | 固定 split 可复现 | `test_expert_split.TestFrozenEvaluation.test_split_reproducible` |
| 5 | 冻结 Evaluation（新标注 train/parked 显式去向） | `test_expert_split.TestFrozenEvaluation` |
| 6 | Macro-F1 手算一致 | `test_expert_split.TestEvaluationTable.test_precision_recall_f1_hand_computed` |
| 7 | 缺 class evaluation truth → N/A + `INSUFFICIENT_EVAL_SUPPORT` | `test_expert_split.TestEvaluationTable.test_zero_support_is_na_not_zero` |
| 8 | Cluster OOF consistency（recording_A 不在评分模型训练集） | `test_oof_scoring.TestOOFScoringModels` |
| 9 | 禁止 Production fallback（OOF 不可用 → pseudo=0，Production 0.999 无效） | `test_pseudo_labels.TestNoProductionFallback` |
| 10 | 局部异质区域（B 区候选 `local_expert_conflict` 全拒） | `test_pseudo_labels.TestLocalHeterogeneity` |

以下为完整测试计划（括号内为验收里程碑）：

**红线（M2）**

1. 任何路径下 `expert_label` 不被伪标签写入；导出表 `review_status` 恒为空；
2. 校准/评估集合中无 `label_source=pseudo` 行；混入伪标签触发 `EVAL_CONTAMINATED`。

**OOF 防自我确认（M1）**

3. 构造 `recording_A`：断言用于给 recording_A 候选评分的 fold 模型
   训练数据中**不存在** recording_A 的任何 pulse；
4. OOF 不可用（录音 < 4 / fold 无效 / 特征缺失）→ 零传播，
   `oof_probability_unavailable`，且**不会**回退 Production 概率；
5. 每条候选恰好被一个 fold 模型评分，`oof_fold_id` 与 fold 训练录音一致。

**簇门控（M1）**

6. HIGH → 可传播；MODERATE → 0 pseudo（`suggest_review`）；
   LOW → 0 pseudo；MIXED → 0 pseudo；INSUFFICIENT/NO_MODEL → 0 pseudo；
7. reviewed 阈值：9 条专家 → 不传播（`insufficient_reviewed`）；
   10 条且 9/10 主导 → 允许继续；纯度 0.89 → 不传播；
8. 半径：已审一致样本 P95 正确；不足 3 条退化全簇 P25 且打 `radius_fallback`。

**样本级闸门（M1）**

9. top1 mismatch：簇主导 Species A、模型 top1 为 B → 拒绝
   （即使 P(A) > threshold 也不行——记录 `model_top1_mismatch`）；
10. probability margin：top1 足够但 margin 不足 → 拒绝
    （构造 margin < 0.20 的非归一/异常分布样本）；
11. 专家样本 / boundary / outlier / noise / uncertain 各落选原因逐一对上。

**Diversity sampling（M2）**

12. 构造同一录音、时间连续、center 极近的大量候选：
    断言时间桶去冗余生效（每秒 ≤ 1 条），且最终**不是**全选最近样本
    （三个距离层都有产出或配额回流可解释）；
13. 单录音大簇：每录音每类截断生效（`recording_quota_full`）；
14. 层配额与回流：`distance_stratum_quota_full` / `cluster_quota_full` 可复现。

**训练与采纳（M2/M3）**

15. `sample_weight` 生效（伪标签 0.4）；校准 fold 只含专家样本；
16. 同一 `source_recording` 的伪标签与专家样本同 fold（group split）；
17. 采纳闸门：Macro F1 下降 → 拒绝；单类 recall −0.03 → 拒绝；
18. 轮次：`PSEUDO_MAX_ROUNDS` 生效，第 3 轮不启动；
19. 稳定性报告：数据不足时跳过；触发时 holdout 录音的伪标签不进该次训练。

**确定性（M1 起）**

20. 相同数据 + `random_state` + 模型 + config 重复运行，
    `pseudo_training_dataset.csv` 逐行一致（`selected_at` 除外）。

---

## 16. 实施分期（v2 重排）

### M1 — 数据与 scoring 基础设施（不训练新模型）

| 项 | 内容 |
|---|---|
| 修改内容 | ① `clustered_pulses.csv` 全量 `distance_to_center` + `cluster_model.joblib` 落盘（§7.1 方案 A）；② `source_recording` 列贯通三张输出表；③ OOF cross-fitting 评分（§5，复用 `StratifiedGroupKFold` + `_fit_calibrated`）；④ top1/top2/margin/applied_threshold 计算；⑤ 簇门控（10 审 / 0.90 / HIGH-only）+ 样本级安全闸门；⑥ `pseudo_cluster_gates.csv` + `pseudo_label_candidates.csv`（含完整 rejection_reason） |
| 验收 | 能正确、可解释、可复现地回答"哪些 pulse 会成为 pseudo candidate"；OOF 不可用时零传播 |
| 对应测试 | 3–11、20 |

### M2 — Diversity + Dataset + Weighted Training

| 项 | 内容 |
|---|---|
| 修改内容 | ① 两阶段多样性抽样（时间桶 / 录音轮转限额 / 距离分层，§6.2）；② `pseudo_training_dataset.csv`（完整出处链 §7.3）；③ `species_model` 带权训练（expert 1.0 / pseudo 0.4）；④ 校准隔离 + 评估 expert-only 过滤 + `EVAL_CONTAMINATED` 校验；⑤ 类别配比护栏 |
| 验收 | 训练链路吃伪标签不污染任何真值通道；重复样本被有效去冗余 |
| 对应测试 | 1、2、12–16 |

### M3 — Iteration + Adoption + UI

| 项 | 内容 |
|---|---|
| 修改内容 | ① pseudo round 循环（≤ 2 轮、每轮 OOF 重筛）；② adoption gate（专家留出集对比）；③ 稳定性报告（诊断项）；④ Active Learning 补审队列（recommended_action + 跳转审核）；⑤ ⑥ 页面「伪标签反哺」子面板 + CLI `--pseudo-export/--pseudo-round` |
| 验收 | 端到端演示任务跑通：审核 → 伪标签 → v(n+1) → 对比 → 采纳/拒绝 → 补审回流 |
| 对应测试 | 17–19 + 端到端演示 |

---

## 17. 风险与残留局限 + Future Work

残留风险（v3 已缓解但未消除）：

- **簇内标签传播本身是假设**：10 条审样 ≥ 90% 纯度仍无法排除簇内
  存在未被代表样本触及的异质区域；缓解（v3）：核心区半径 + **局部专家
  邻域检查**（§4.3.5）+ 低权 + 轮次上限；但邻域只在专家样本**存在**的
  地方有效——专家完全没触及的异质区域仍靠 core radius 兜底；
- **cluster status 已改 OOF 判定**（v3）：Production 模型退出全部伪标签
  准入决策，"status 偏乐观"风险消除；但 OOF 一致性本身受 Train Pool
  专家样本分布影响（专家只审了代表样本）；
- **parked 标注（I15 的保守裁决）**：Evaluation 录音上冻结后新增的专家
  标注不进训练也不进评估，可能造成"专家白干"的观感；必须显式重新生成
  划分（`split_version + 1`，旧指标不再完全可比）才能启用；
- **聚类结构本身由全部录音拟合**（含评估录音）：无标签信息，
  一般可接受，但严格的科研口径应披露；
- **OOF fold 校准在小样本下不稳**：fold 内校准失败 → model support +
  0.95 阈值，伪标签更少（可接受方向）；
- **`local_expert_check=insufficient` 放行**：邻居 < 3 时不硬判
  （保守默认按用户"宁缺毋滥"方向其实是折衷）；若要求更严可改为
  直接拒绝，见 Future Work；
- **距离分层偏向外层**可能引入略多边界噪声：全部样本仍在 core radius
  内（P95 证据半径），风险有界；
- **时间桶边界效应**：跨桶相邻样本可各留 1 条，属已知小冗余。

Future Work（本轮明确不做）：

- per-species 阈值自动学习（expert-only validation 上定 precision 目标）；
- 动态 pseudo 权重（purity / OOF 概率 / core distance 加权）；
- ~~OOF-based cluster status~~（v3 已完成）；
- `local_expert_check=insufficient` 升级为 hard gate（当前保守放行 + 记录）；
- 超过 2 轮的自训练与更激进的 pseudo 扩充；
- 伪标签 round 循环 UI（`--pseudo-round`）与采纳链路的界面化。
