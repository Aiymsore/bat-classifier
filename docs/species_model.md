# 已确认物种模型与 Cluster 模型对照（⑥ 批量整理与专家标注）

本文档描述 ⑥ 页面新增的「已确认物种模型」与「模型对照结果」功能的算法与数据流。
核心思想：

```text
聚类负责压缩大量数据
+ 监督模型负责已有知识判断
+ Center 看典型 / Boundary 看交界 / Random 做无偏抽查 / Outlier 看异常尾部
+ Calibrated Probability 衡量经过校准后的模型概率
+ Agreement 衡量这些代表样本是否指向同一物种
+ 低一致性才交给专家
```

## 1. 已确认物种模型（监督物种分类器）

与 A-D 临时声型模型（KMeans 伪标签）、noise/bat 门控模型完全分开，
保存在独立目录：

```text
models/
    species/
        <model_name>/
            model.joblib      # 分类器（可能包着 CalibratedClassifierCV）
            metadata.json     # 训练元数据（人类可读）
            calibration_reliability.png  # reliability plot（有独立评估集时）
```

### 1.1 训练数据

只允许真实专家标签：

- 必需列：`pulse_id`、`expert_label`、`source_recording`/`source_file`/`source_path`
  以及固定声学特征列（`SPECIES_FEATURE_COLUMNS`）；
- 只允许 `review_status == reviewed` 且 `expert_label` 非空的样本；
- **绝不**使用 `cluster_id`、A-D 聚类标签、`cluster_inferred_label` 作为训练标签；
- 来源可以是多个 CSV 或多个 ⑥ 任务目录（`expert_training_dataset.csv`）。

### 1.2 训练流程

```text
专家确认 pulse
  → 录音级 group split（GroupShuffleSplit + StratifiedGroupKFold）
  → RandomForestClassifier（class_weight="balanced_subsample"）
  → Probability Calibration（CalibratedClassifierCV, 默认 sigmoid）
  → 评估（留出录音）
  → 保存 model.joblib + metadata.json
```

录音级隔离要求（科研要求）：

- 训练 / 校准 / 评估三者按 `source_recording` 分组，同一录音的 pulse
  绝不跨集合；
- 评估集是从未参与训练与校准的整条录音（GroupShuffleSplit 留出）；
- 校准 folds 由 `StratifiedGroupKFold(groups=source_recording)` 产生，
  每个 fold 的训练录音与校准录音严格不相交。

### 1.3 概率校准与“不伪造校准”

- 默认 `CalibratedClassifierCV(method="sigmoid")`；
- `isotonic` 仅在样本量 ≥ `SPECIES_ISOTONIC_MIN_SAMPLES` 时允许；
- 当独立录音数 < `SPECIES_MIN_RECORDINGS_FOR_CALIBRATION` 时**不做校准**，
  明确提示「当前数据不足以进行可靠概率校准」，此时概率只能称
  **模型支持度（model support）**，UI 显示「未校准模型」。

### 1.4 评估指标

`Accuracy`、`Macro Precision / Recall / F1`、`Confusion Matrix`、
`log loss`、多分类 `Brier score`，以及 top-label reliability 分箱 + ECE
（附 reliability plot）。评估重点看 Macro F1（类别不平衡时 Accuracy 会误导）。

### 1.5 模型输出

```text
Pipistrellus abramus     0.91   ← calibrated probability（校准概率）
Myotis sp.               0.06
Rhinolophus sp.          0.03
```

- `max calibrated probability < uncertain 阈值（默认 0.60，可配置）`
  → `predicted_label = uncertain`，不硬分；
- 输出列名为 `model_inferred_label` / `predicted_label`，
  **绝不写入 `expert_label`**；
- 若代表样本与模型训练数据重合，打标 `TRAINING_DATA_OVERLAP`
  （结果不代表泛化能力）。

### 1.6 冻结 Evaluation Set 训练路径（⑥ 专家数据划分，详见 pseudo_label_feedback.md §2.5）

当训练源对应的 ⑥ 任务存在 `tables/expert_split.json` 时，
`train_species_model(..., expert_split=..., pseudo_source=...)`
自动走**冻结评估路径**（UI「训练新模型」自动携带该文件）：

- **训练 / 校准**：只用 Expert Train Pool（+ 带权伪标签：expert 1.0 /
  pseudo `PSEUDO_SAMPLE_WEIGHT`=0.4；校准 fold 的校准侧永远只有专家样本，
  伪标签只进 fold 训练侧）；
- **评估**：只在 Fixed Gold Evaluation（真值**只有** `expert_label`，
  `eval_labels="expert_only"`）上计算
  `classification_table`：逐类 **Support / Precision / Recall / F1** +
  **Macro-F1**（support ≥ 1 的物种等权平均）；**Evaluation 真值 = 0 的物种
  显示 N/A + `INSUFFICIENT_EVAL_SUPPORT`**，绝不输出误导性的 F1 = 0；
- **EVAL_CONTAMINATED 断言**：评估录音出现在训练 / 校准 / 伪标签数据中
  → 抛 `EvalContaminationError`（含违规录音 ID），训练终止；
  训练前另跑 `assert_no_pseudo_labels(eval_data, ...)`——伪标签行混进
  评估即拒绝（I17）；
- 每次训练随模型落盘 `expert_evaluation.csv`（**仅展示缓存**）+
  metadata（`expert_split` 版本 / `training_labels` 计数 / split 录音清单）；
  未走冻结路径时保持 §1.2 的 group-split holdout 行为不变。

### 1.7 Fixed Gold Evaluation（v4：外部锚点，详见 pseudo_label_feedback.md §2.5.5）

三类数据严格分离：**1. Expert Train Pool**（唯一专家来源）/
**2. Fixed Gold Evaluation**（只出分：Precision/Recall/F1/Macro-F1/
log loss/混淆矩阵 + 新旧对比）/ **3. Pseudo Training Pool**
（`label_source=pseudo`，权重 0.4）。Evaluation 录音 / pulse / 真值
一旦建成 v1 **永不修改、永不自动扩充**；能进 Evaluation 的只有独立
人工专家真值；伪标签永远是伪标签，唯一升级路径 = 人工专家重新真实审核。
新版本 = 新完整录音 + 独立人工审核另建 `gold_eval_v2.json`（建议盲审），
v1/v2 独立存储绝不覆盖。Evaluation 回答的问题："模型是真的进步了，
还是只是越来越相信自己过去生成的 pseudo label"。

相关 API（`species_model.py` / `expert_split.py`）：

- `evaluate_payload_on_gold_evaluation(payload, frame, gold)`：
  **真值一律取 `gold["truth"]` 冻结快照**（活标注被篡改不影响指标），
  不读任何缓存指标、每次真实 re-predict；缺特征考题上报
  `missing_feature_pulses` 不静默吞掉；
- `compare_models_on_gold_evaluation(old_payload, new_payload, frame, gold)`：
  **两个模型同时在同一个 Gold Evaluation 上重新评估**再对比
  （`same_evaluation=True`）——绝不拿旧模型缓存结果对比新模型现算结果；
  采纳判据同上（Macro-F1 Δ、逐类 recall 下降 >
  `PSEUDO_MAX_RECALL_DROP`=0.02 拒绝）；
- 归档：`gold_record_from_split` / `archive_gold_evaluation` /
  `create_gold_evaluation`（`assert_no_pseudo_labels` + `assert_eval_isolation`
  双校验）/ `load_gold_evaluation` / `list_gold_evaluations`，
  存 `tables/evaluations/gold_eval_v{N}.json`，同版本重写直接
  `FileExistsError`。

## 2. 模型对照（四类 representative 分别统计）

每个 Cluster 的 Center / Boundary / Random / Outlier（均为真实 pulse，
不是数学质心）分别送入已确认物种模型，得到 `predicted_label` 与
`calibrated_probability`，然后**分组独立统计**：

```text
agreement_rate = 该组中预测为主导标签的样本数 / 该组有效模型预测样本数
```

- `agreement` 与 `confidence` 是两个不同指标，绝不混用；
- `average_calibrated_probability`：先定主导标签，再对**组内全部样本**
  取 `P(主导标签)`（包括预测成其他标签的样本）后求平均；
  绝不只平均“预测正确”的样本；
- 同一 pulse 可能被多种策略选中：在各策略统计中分别出现（评估不同
  sampling strategy），但**综合统计必须按 pulse_id 去重**；
- 综合平均概率对 unique pulses 逐条取 `P(overall_dominant_label)` 再平均，
  不是四个组百分比的平均。

### 2.1 Cluster 状态（阈值集中在 config.py）

> **适用范围**：下表 `determine_cluster_status`（center/boundary/overall，
> 基于 representative 采样组）**只服务 ⑥ 模型对照**。伪标签反哺的簇门控
> 用的是 **Cluster-wide OOF consistency**（`determine_cluster_oof_consistency`，
> 覆盖整个可合法 OOF 评分的 Train-side Cluster，见
> pseudo_label_feedback.md §4.1），阈值复用同一组 `MODEL_COMPARE_*` 常量。

| 状态 | 规则 |
|---|---|
| `HIGH_CONSISTENCY` | center ≥ 0.90 且 boundary ≥ 0.80 且 overall ≥ 0.85 且 overall 概率 ≥ 0.80 |
| `MIXED` | Center 与 Boundary 主导物种不同（需要人工审核） |
| `LOW_CONSISTENCY` | overall agreement < 0.70（需要人工审核） |
| `MODERATE_CONSISTENCY` | 以上都不满足但 overall ≥ 0.70 |
| `INSUFFICIENT_SAMPLES` | center/boundary/overall 有效样本数不足 |
| `NO_MODEL` | 未加载模型 |

Cluster ≠ 物种：多个 Cluster 可以被判断为同一物种，这是合法且预期的结果。

### 2.2 输出表

`results/curation/<run_id>/tables/`：

- `cluster_model_comparison.csv`：每个 Cluster 一行（四类各自
  主导物种 / 一致率 / 平均校准概率 / 分布 + 综合判断 + 状态 + 标记）；
- `representative_model_predictions.csv`：逐 representative 一行
  （`cluster_id, pulse_id, selection_reason, source_file, pulse_file,
  model_inferred_label, calibrated_probability, raw_probability,
  all_class_probabilities, is_uncertain` 等）。

## 3. 模型管理（⑥ 页面「已确认物种模型」区域）

- **训练新模型**：选择 CSV / ⑥ 任务目录（可多来源合并）；
- **导入现成模型**：校验 `feature_columns`、类别列表、模型类型、是否校准；
  特征与当前流水线不一致时**拒绝运行并列出具体缺失列**，绝不静默改序/补值；
  A-D 临时声型模型会被拒绝导入；
- **使用标注数据校准此模型**：为导入的 raw RandomForest 补概率校准；
- **切换模型 / 查看模型信息 / 卸载模型**；
- 当前模型路径与 uncertain 阈值保存在 `app_settings.json`，下次启动自动恢复；
  路径失效时显示「模型不可用」，不崩溃。

## 4. 测试

`tests/test_species_model.py` 与 `tests/test_model_comparison.py` 覆盖：

- 训练后 `is_calibrated == True`，概率合法且各类概率和 ≈ 1；
- 同一 `source_recording` 绝不跨训练/校准/评估集合；
- agreement：`A A A A B` → dominant = A，agreement = 0.8；
- 平均校准概率：`0.96/0.94/0.91/0.88/0.43` → 0.824（含预测错样本）；
- 四类 representative 独立统计；综合统计按 pulse 去重；
- 导入模型特征一致可加载、缺失列明确拒绝；
- 未校准模型不输出“校准概率”（只输出模型支持度）；
- 单物种数据拒绝训练；
- 模型结果绝不覆盖 `expert_label`。
