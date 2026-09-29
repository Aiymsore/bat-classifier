# 蝙蝠超声信号检测与声型分类程序 · 代码结构说明

程序从蝙蝠超声录音（WAV，250 kHz）中自动检测脉冲、提取声学特征，并通过两级模型对新录音分类：第一层判断 noise/bat，第二层把 bat 脉冲归入 A–D 四类声型。A–D 是无监督聚类得到的临时声型标签，不是物种标签。

> **项目定位**：当前版本是声学信号处理与机器学习原型，不是经过专家验证的蝙蝠物种鉴定系统。A–D 代表无监督聚类得到的声型，不能直接写成真实物种。
>
> *蝙蝠超声信号检测与声型聚类系统（自用）
基于Python构建蝙蝠超声录音处理流程，完成高通滤波、候选脉冲检测、频谱追踪及声学特征提取；
提取持续时间、峰值频率、频带宽度、频率斜率、SNR等特征，使用PCA和KMeans探索声型结构；
构建noise/bat筛选与A–D临时声型分类的两级机器学习流程；
在10条共466.88秒录音中检测578个候选片段，筛选得到441个有效脉冲；
开发可直接运行的命令行Demo，并添加端到端冒烟测试及CSV、JSON、结果图输出；
明确当前A–D为无监督临时声型，后续计划接入专家物种标签并采用按录音分组的交叉验证。
>
> ## 数据来源说明

本项目使用的数据来源于本人大学（东北师范大学）导师科研项目中的蝙蝠超声录音数据。

数据主要用于验证蝙蝠声学信号自动检测、特征提取以及机器学习分析流程。
由于当前数据仍处于科研整理阶段，暂未包含完整的专家物种标注信息，因此本项目当前版本重点关注：

- 超声信号自动检测；
- 声学特征提取；
- 无监督声型探索；
- 分类流程原型验证。

当前 A-D 标签表示基于声学特征聚类得到的临时声型，而非经过专家确认的物种名称。

后续获得完整物种标注数据后，将进一步开展：
- 物种级分类模型训练；
- 按录音来源划分的数据验证；
- Precision、Recall、F1 等指标评估。

## 核心功能

- 对 250 kHz WAV 录音进行高通滤波和分块分析。
- 自动检测并精修蝙蝠候选脉冲。
- 实验性分离同一时间段内的近频声波分量。
- 提取持续时间、峰值频率、带宽、频率斜率、SNR 等特征。
- 通过两级流程完成 `noise/bat` 判断与 A–D 声型分类。
- 输出 PCA 图、聚类中心、录音级汇总、模型文件和新录音预测结果。

## 技术栈

Python、NumPy、Pandas、SciPy、Librosa、SoundFile、Matplotlib、scikit-learn、Joblib。

## 数据流程

```text
WAV录音
  → audio_io        读取、重采样、高通滤波
  → detection       候选脉冲检测与边界精修
  → features        特征提取、质量筛选
  → separation      同时近频声波分离（实验性）
  → pulse_table     逐脉冲明细建表
  → noise_gate      第一层 noise/bat 门控训练
  → clustering      第二层 A-D 声型聚类训练
  → prediction      新录音两级预测
  → results/        CSV、JSON、图像、分离音频输出
```

## 可直接运行的 Demo

仓库内置一个 4 秒、约 2 MB 的示例 WAV。它用于展示“候选脉冲检测 → noise/bat 判断 → A-D 临时声型分类 → CSV/JSON 输出”的完整推理流程，不需要下载 167 MB 的原始数据包。

安装依赖后运行：

```powershell
python demo.py --self-test
```

Windows 也可以双击根目录的 `run_demo.bat`。

首次运行会使用仓库已有的两个结果表自动构建演示模型：

- `results/tables/acoustic_pulses.csv`：训练第一层 `noise/bat` 模型；
- `results/tables/acoustic_clusters_ABCD.csv`：训练第二层 A-D 临时声型模型。

默认输入为：

```text
data/sample/demo_xianren_0_4s.wav
```

也可以预测自己的 WAV：

```powershell
python demo.py --input "D:\audio\new_recording.wav"
```

需要强制重新构建演示模型时：

```powershell
python demo.py --rebuild-model --self-test
```

成功后主要输出到：

```text
results/predictions/<文件名>_noise_ABCD.csv
results/predictions/demo_summary.json
results/figures/spectrograms/
results/figures/pulse_zooms/
```

本仓库附带样例在当前测试环境中检测到 61 个有效候选，其中 51 个判为 bat、10 个判为 noise；主要临时声型为 C。该结果只用于验证工程链路可运行，不能解释为真实物种识别结论。测试记录见 [`docs/demo_test.md`](docs/demo_test.md)。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

把原始数据按以下任一种方式放置：

- 将 `声波文件.zip` 放到 `data/`；
- 或直接将 WAV 文件放到 `data/raw/`。

然后运行：

```powershell
python run.py
```

训练完成后，可按照终端提示输入一个 WAV 文件或文件夹路径进行预测。

## 各模块功能

### config.py — 路径与参数配置

集中存放全部常量：目录路径与结果表文件名、检测参数（能量阈值 `-28 dB`、脉冲时长 0.8–40 ms、合并间隔、填充余量）、边界精修参数（`CORE_*`）、质量门槛（SNR ≥ 4 dB、轨迹覆盖率 ≥ 0.55 等）、聚类配置（11 个特征列、4 类、单录音样本上限）、分块检测窗口、分离参数（`SEPARATION_*`）、门控参数。调参只需改此文件。

### audio_io.py — 音频读取与频谱基础

- `load_audio`：读取 WAV，重采样到 250 kHz 单声道，去直流、高通滤波、峰值归一化。
- `highpass_filter`：4 阶 butter 高通，滤除 15 kHz 以下背景噪声。
- `calculate_audio_content_hash`：对解码后波形计算 SHA-256，识别内容重复的录音（与文件头无关）。
- `build_spectrogram`：计算功率谱（n_fft=1024，hop=256）。
- `find_wav_files`：递归查找 `.wav` / `.WAV`。
- `ultrasonic_band_mask`、`adaptive_stft_params`、`interval_overlap_ratio`：按片段长度自适应 STFT 参数、超声频段（15–120 kHz）掩码、时间区间重叠比例，供下游多个模块复用。

### detection.py — 候选脉冲检测与边界精修

- `detect_pulses`：对超声频段能量同时施加固定相对阈值和背景自适应阈值，找出候选脉冲区间。
- `merge_active_frames`：把间隔很短的活动帧合并成连续区。
- `select_group_containing_peak` + `refine_pulse_boundary`：在候选区内部重新定位高信噪比脉冲核心，去掉前后背景拖尾。
- `refine_all_pulses`：批量精修，并按重叠比例 ≥ 0.60 去重。
- `remove_duplicate_boundary_pulses`：删除分块重叠区域重复检测到的同一脉冲（重叠 ≥ 0.50 或中心距 ≤ 1.5 ms）。
- `detect_and_refine_pulses_chunked`：长录音抽窗分块检测再合并去重；默认（演示/预测）只抽 2 个窗口约 2×4 秒以缩短运行时间，`max_windows` 参数可扩大分析跨度（⑥ 批量整理默认延长到 60 秒，见 config 的 `CURATION_ANALYSIS_SPAN_S`）。

### features.py — 单脉冲特征提取与质量筛选

- `extract_pulse_features`：只在可靠高能量帧上计算峰值频率、起止频率、频率下降幅度、f05/f95、90% 带宽、频率斜率、主导频率轨迹覆盖率、最大跳频、RMS。
- `evaluate_pulse_quality`：与物种无关的通用质量门槛，返回接受/拒绝及原因（low_snr、low_track_coverage、unstable_frequency_track、peak_frequency_out_of_range）。
- `weighted_frequency_quantile`：功率加权频率分位数。
- `accepted_mask`：兼容布尔列和 CSV 读回的字符串布尔列。
- `numeric_feature_frame`：特征列统一转数值并把 ±inf 替换为 NaN。

### separation.py — 同时近频声波分离（实验性）

- `_link_spectral_peak_tracks`：逐帧寻找频谱局部峰，用带速度预测的贪心匹配连成频率轨迹。
- `_compare_simultaneous_tracks` / `_tracks_are_harmonic_duplicates`：判断两条轨迹是否时间重叠足够、中位频率差可分辨、排除二/三次谐波误判。
- `separate_simultaneous_pulse_components`：确认存在 ≥ 2 条独立轨迹时，按高斯软掩码把混合波形拆成多条分量音频；否则原样返回，避免对普通单脉冲造成失真。
- `save_separated_pulse` / `clear_separated_pulses_for_source`：分离分量 WAV 的写盘与重建前清理。

### pulse_table.py — 数据准备与逐脉冲建表

- `prepare_demo_dataset`：创建全部输出目录；`data/raw` 无 WAV 时从 `data/` 下的压缩包解压。
- `create_demo_pulse_rows`：对每个精修候选尝试分离，逐分量提取特征并做质量筛选；若分离出的分量通不过质量门槛则回退为原混合脉冲。
- `build_demo_pulse_table`：遍历全部录音（内容哈希去重），生成逐脉冲明细 `acoustic_pulses.csv` 和录音级检测汇总。

### visualization.py — 图像输出

- `save_fast_overview_spectrogram`：整段录音概览声谱图，时间帧数限制在约 4000 以内。
- `save_top_pulse_zoom_grid`：能量最强的若干脉冲放大图，叠加每帧主导频率轮廓线。
- `save_pca_plot`：A-D 聚类的 PCA 二维散点图。
- 导入本模块即完成 matplotlib Agg 后端与中文字体设置。

### clustering.py — A-D 声型聚类

- `prepare_cluster_data`：只取合格脉冲，数值清洗，限制单个录音最多贡献 150 个脉冲。
- `standardize_cluster_centers` + `create_acoustic_type_mapping`：在四个聚类中心之间标准化，按 FM 型 / CF 型 / 低频窄带评分把无意义的 KMeans 编号稳定映射为 A-D。
- `cluster_acoustic_types`：标准化 + KMeans（k=声型数量）、PCA、轮廓系数，保存聚类模型 `.joblib` 与元数据 JSON（含 `trained_at` 训练时间与中文摘要 `summary_zh`）。
- `train_temporary_classifier`：训练复现 A-D 标签的临时随机森林（600 棵），输出与聚类标签的一致率。
- `save_cluster_representatives`：每类取距中心最近的 10 个代表脉冲，复制到 `cluster_examples/声型A-D/`。
- `create_recording_cluster_summary`：统计每段录音中 A-D 的数量与占比。

### noise_gate.py — 第一层 noise/bat 门控

- `prepare_noise_gate_data`：用质量筛选结果生成原型伪标签（accepted→bat，rejected→noise），按录音×类别限量保留，两类都保留靠前的难样本。
- `make_noise_gate_model` / `train_noise_gate`：随机森林（600 棵），≥ 4 个独立录音时按录音分组留出 25% 验证，保存模型与验证明细 CSV。
- `apply_noise_gate`：对新候选批量输出 bat/noise 概率，按阈值 0.55 判定。

### prediction.py — 新录音两级预测

- `predict_one_demo_wav_two_stage`：单个 WAV 完整走 检测 → 分离 → 门控 → A-D 分类，写出逐片段预测 CSV、概览声谱图和放大图，返回汇总 dict。
- `predict_demo_path_two_stage`：支持单个文件或整个文件夹，写出预测汇总 CSV。

### segment_analysis.py — 片段识别与叫声统计（⑦）

对**WAV 片段或整条超长录音**（一个 WAV 内部可能含 N 次独立叫声）做 Inference + Counting，绝不重新训练。复用 `detection`/`pulse_table`/`noise_gate`/`features`/`clustering` 现有链路：

- `analyze_segment(wav_path, combined_payload, config, ...)`：单 WAV → 流式检测内部叫声 + 逐叫声分类 + 片段聚合，返回含逐叫声明细的结果 dict；支持 `chunk_progress`（块级进度）/`should_stop`（chunk 边界优雅停止）/`resume`（断点续跑）/`max_workers`（有限并行）。
- `analyze_segment_folder(input_path, combined_payload, config, ...)`：批量（单文件/文件夹），逐个流式 analyze→save→release，失败隔离，写独立输出目录。
- `count_calls_in_segment` 语义由 `analyze_segment` 的 `detected_call_count` 给出（有效蝙蝠叫声次数）。
- `classify_pulse_rows`：逐 pulse 分类 + 置信度 + 不确定标记，严格按 metadata 特征顺序复用 StandardScaler/classifier。
- `summarize_segment_results`：片段汇总 + 总体类别统计（各类别按实际 call 数累加，绝不归给 major_label）。
- `describe_model` / `load_classification_model`：读取现有确定模型，判定是否为临时声型模型、是否支持 predict_proba。

### segment_stream.py — 超长 WAV 流式分析引擎（⑦ 长录音后端）

内存与文件总长无关的 chunk 管线：`ChunkSpec`/`build_chunk_specs`（chunk+overlap 网格，coverage 连续铺满全长）、`StreamingPreprocessor`（流式预处理 + 两遍全局 mean/peak，与 `load_audio` 同口径）、`chunk_may_contain_ultrasound`（保守静音快筛）、`dedupe_cross_chunk_rows`（跨块去重安全网）、`ChunkCheckpoint`（增量 CSV + run_state + 文件身份校验 + 水位线截断）、`stream_chunks`（顺序/有限并行编排，按 global 时间序归并）、`SpectrogramSummary`（流式低分辨率 overview 摘要，仅可视化）。检测/精修/建表复用 `detection`/`pulse_table`，分类复用 `segment_analysis` 小批量 `_finalize_chunk_rows`——模型/特征/阈值完全不变。

### pipeline.py — 完整流程编排

`run_full_training(labels, type_descriptions, data_source)`：数据准备 → 脉冲建表 → 门控训练 → 聚类训练 → 合并保存两级模型 `noise_bat_ABCD_model.joblib`，返回合并 payload；供 CLI 与桌面 App 共同调用。`main()` 在此基础上保留交互式路径输入。只做流程串联，算法细节都在各功能模块。

### demo.py — 轻量演示

从 `results/tables/` 已提交的两个结果表（`acoustic_pulses.csv`、`acoustic_clusters_ABCD.csv`）重建小规模两级模型（随机森林 300/400 棵，与 pipeline 的 600 棵不同，是有意设计），无需原始录音数据即可复现演示。含 argparse CLI：`--input`、`--model`、`--rebuild-model`、`--self-test`。

## 桌面 App（bat_classifier_app.py）

PySide6 窗口程序，七页签，所有耗时操作在后台线程执行、日志回显到底部面板：

| 文件 | 功能 |
|---|---|
| `src/bat_classifier/app/settings.py` | 工作区/标签/数据源设置，持久化到 `workspace/app_settings.json`；启动时先写入 `BAT_WORKSPACE` 环境变量再导入库（config 路径在导入期解析） |
| `src/bat_classifier/app/workers.py` | QThread 工作类（导入数据/训练/预测/批量整理/片段识别），stdout 重定向为 Qt 信号 |
| `src/bat_classifier/app/image_viewer.py` | 结果图像查看：大图对话框（适应窗口/1:1）与"列表+预览"画廊组件 |
| `src/bat_classifier/app/main_window.py` | 七页签主窗口：①数据来源（选 WAV 文件夹或 zip，导入到 `data/raw`）②模型训练（声型数量 2–6、标签名与描述可编辑、可选诊断图；训练完成自动刷新 ③/④/⑦ 的模型列表）③预测（选模型 + WAV/文件夹，结果表格预览）④模型管理（模型列表/导入模型、中文训练摘要、原始元数据 JSON、模型目录）⑤结果图像（声谱图/脉冲放大图/PCA 图的浏览与大图查看）⑥批量整理与专家标注（见下节）⑦片段识别与叫声统计（见下节） |
| `src/bat_classifier/app/model_picker.py` | 共用模型选择控件：模型下拉框 + **导入模型文件…**（复制到 `models/imported/`）+ 刷新；`model_changed` 信号供 ③/④/⑦ 三处同步“当前使用模型” |
| `src/bat_classifier/app/curation_tab.py` | ⑥ 页面：任务创建/参数/进度/导出，串起扫描→处理→聚类与审核工作台 |
| `src/bat_classifier/app/curation_review.py` | 专家审核工作台：Cluster 列表 / 代表样本列表 / 声谱图+声学参数 / 标注表单 / 纯度提示 |
| `src/bat_classifier/app/segment_analysis_tab.py` | ⑦ 页面：输入选择 + 模型状态与模型切换/导入（临时声型显式提示）+ 参数 + 进度 + 片段/总体统计表 + 选中片段详情/声谱图 |
| `src/bat_classifier/app/ui_utils.py` | 布局小工具 `make_scrollable`：把内容包进 QScrollArea，内容超出可视高度时按需出现滚动条（⑥ 整页滚动：表单+审核工作台+模型对照结果同一滚动区；⑦ 上半表单滚动 + 下半结果区固定） |

- 运行：`python bat_classifier_app.py`；`--selftest` 离屏自检（供打包验证）。
- 小屏幕适配：⑥ **整页滚动**（表单 + 审核工作台 + 模型对照结果在同一个滚动区，屏幕再矮也滚轮滚到底）；⑦ 上半部分表单区可滚动、与下半部分结果区用纵向分隔条分栏；底部运行日志可"收起/展开"；窗口大小与位置、日志折叠状态记入 `app_settings.json`，下次启动恢复（无效位置时按屏幕可用区域取 90%）。
- 标签规则：默认 A-D 四类沿用语义归类；其他数量或名称按聚类中心峰值频率从低到高依次分配。
- 工作区：源码态默认为仓库根目录；打包态为 exe 旁的 `workspace/`，含 `data/raw`、`results`、`models`。
- 打包：`build_exe.bat`（内部执行 `pyinstaller bat_app.spec --noconfirm` 并对产物跑 `--selftest`），产物 `dist/BatClassifier/BatClassifier.exe`。

## ⑥ 批量整理与专家标注（大规模未标注 WAV 分诊）

针对"手上有一大堆还没有物种标签的录音，先分诊、再请专家看图打标签"的场景。独立于 ①②③ 工作流，**完全不触碰 `data/raw`**，所有产出写入独立任务目录 `results/curation/<run_id>/`。

### 流程

```text
选择外部 WAV 文件夹（递归，.wav/.WAV）
  → 扫描入册 manifest（pending/processing/completed/failed/duplicate）
  → 内容哈希去重：解码后 PCM 的 SHA-256 相同即重复（文件名/RIFF 元数据不同也能识别）
      重复文件记 duplicate_of，绝不重复分析，绝不写入 data/raw
  → 逐条录音：检测 + 边界精修 + 质量筛选（accepted 与 rejected 全部保留，rejected 带 rejection_reason）
      分析跨度 60 秒：≤60 秒的录音整条全覆盖，更长的录音在整个时长上均匀抽 15 个 4 秒窗口
      （①②③ 演示/预测仍是约 2×4 秒不变；跨度用 config 的 CURATION_ANALYSIS_SPAN_S 调节）
      → 切割每个合格脉冲为独立 WAV，稳定 ID = <audio_hash前16位>_p000001…
        （已分离的分量直接复制分离文件，绝不重新切割混合信号）
  → 聚类：11 维声学特征 + StandardScaler + KMeans
      K 手动 2–50，或自动 K=2..20 逐个算轮廓系数+惯性，取轮廓系数最大者
      ★ 最佳 silhouette 的 K 是"最易分开的份数"，不是物种数量
  → 每 Cluster 四类代表样本：center（最近质心）/ boundary（对最近邻簇 margin 最小）
    / random（固定种子）/ outlier（最远离质心），默认各 5，同一脉冲只留一行
    （多策略命中合并为 selection_reason 如 "center|outlier"）
  → 导出统一尺度声谱图 + 真实 WAV → 专家审核工作台逐条标注
  → 标注原子落盘 expert_annotations.csv，可随时关闭/继续
  → 仅把 reviewed 且有标签的样本导出 expert_training_dataset.csv（可选含噪声）
```

### 使用（⑥ 页面）

1. **数据集**：选 WAV 文件夹或单个 WAV；显示找到的文件数；也可从"已有任务"下拉选择并"打开任务审核"或"继续处理"（断点续跑：completed 跳过，failed 仅在勾选重试时重跑）。
2. **聚类参数**：K 模式（自动/手动 2–50）、四类代表样本数（默认 5）、是否切割合格脉冲 WAV / 保存被拒绝脉冲 WAV / 重试失败文件 → **开始整理**（后台线程，进度条 + 实时统计，单条损坏 WAV 只记 failed 不中断批次）。
3. **审核**：左侧 Cluster 列表（`Cluster 001`…，按质心峰值频率从低到高编号，与既有 A–D 模型无关）→ 中间代表样本列表（带 center/boundary/random/outlier 徽标）→ 右侧声谱图（统一色标，不含预测物种文字）+ 声谱图参数 + 录音信息 → 表单填 标签/置信度/状态（已确认/不确定/噪声/暂不判断）/备注 → "保存并下一条/下一个待审核"。
4. **纯度提示**：展示已审核标签分布，当 ≥5 条已审核且主导标签 <60% 时提示"可能为混合簇"——只是复核提醒，**永不自动把标签复制给同簇其他样本**。
5. **导出**：勾选是否包含噪声 → `tables/expert_training_dataset.csv`（保留 `source_file`/`source_path`/`audio_hash`/`start_s`/`end_s`，供按录音分组切分训练/验证集）。
6. **已确认物种模型**（页面内独立区域）：用**专家确认标签**（`review_status == reviewed` 且 `expert_label` 非空）训练监督物种模型（RandomForest + `CalibratedClassifierCV` 概率校准，训练/校准/评估按 `source_recording` 分组隔离），或导入现成模型（严格校验 `feature_columns`/类别/是否校准，缺失列直接拒绝）；未校准模型只显示"模型支持度"，绝不冒充校准概率。支持 训练/导入/切换/查看信息/卸载，选择保存在 AppSettings。详见 [docs/species_model.md](docs/species_model.md)。
7. **模型对照结果**：每个 Cluster 的 Center / Boundary / Random / Outlier 四类代表样本**分别**送入已确认物种模型，分组统计 主导物种 / 一致率 / 平均校准概率（含预测错样本的 `P(主导)`）/ 类别分布，再给出去重后的综合判断与状态（HIGH/MODERATE/LOW/MIXED/INSUFFICIENT/NO_MODEL）；低一致性交人工审核。Cluster ≠ 物种：多个 Cluster 允许判断为同一物种；模型结果写 `model_inferred_label`，**永不覆盖 `expert_label`**。

### 输出目录结构

```text
results/curation/<run_id>/
├── run_metadata.json          # 输入、时间、git commit、参数、去重/质量/聚类统计
├── manifest.json              # 每条源 WAV 的状态与计数（增量刷盘，支持续跑）
├── tables/
│   ├── recordings.csv         # 录音级清单
│   ├── all_pulses.csv         # 全部脉冲（accepted+rejected+来源+质量字段）
│   ├── accepted_pulses.csv    # 仅合格脉冲
│   ├── rejected_pulses.csv    # 仅被拒绝脉冲（含 rejection_reason）
│   ├── clustered_pulses.csv   # 聚类分配 + pca_1/pca_2
│   ├── k_selection.csv        # 每个候选 K 的 silhouette / inertia / 错误
│   ├── cluster_centers.csv
│   ├── representatives.csv    # 代表样本总表（含距离、margin、selection_reason）
│   ├── expert_annotations.csv # 专家标注（原子写，断点续审）
│   ├── expert_training_dataset.csv  # 导出的监督训练数据
│   ├── cluster_summary.csv
│   ├── cluster_model_comparison.csv        # 模型对照：每 Cluster 一行（四类+综合+状态）
│   ├── representative_model_predictions.csv # 模型对照：逐代表样本一行
│   └── cluster_stats.csv / quality_summary.csv / k_selection_plot.png …
├── figures/
│   ├── cluster_pca.png        # Cluster 分布（Cluster ≠ 物种）
│   └── cluster_quality.png    # 各簇时长/SNR/峰值频率分布
├── clusters/
│   └── cluster_001/
│       ├── audio/<pulse_id>.wav
│       └── spectrograms/<pulse_id>.png
└── audio/
    ├── accepted/<pulse_id>.wav
    └── rejected/<pulse_id>.wav      # 可选
```

### 命令行（可选，低成本）

```powershell
$env:PYTHONPATH='src'
python -m bat_classifier.curation --input D:\audio\2026 --k auto --center 5 --boundary 5 --random 5 --outlier 5
python -m bat_classifier.curation --run-dir results\curation\<run_id>            # 断点续跑
python -m bat_classifier.curation --run-dir results\curation\<run_id> --add-samples 008
python -m bat_classifier.curation --run-dir results\curation\<run_id> --export    # 或 --export <run_dir>
python -m bat_classifier.curation --list-runs
```

### 算法层（无 UI 依赖）

| 模块 | 职责 |
|---|---|
| `src/bat_classifier/curation/pipeline.py` | 任务目录、manifest、扫描/处理/导出编排、原子写、断点续跑 |
| `src/bat_classifier/curation/clustering.py` | K 自动/手动选择（逐 K try/except，单个 K 失败不拖垮整批）、聚类落盘 |
| `src/bat_classifier/curation/representatives.py` | 四类代表样本抽样、去重合并、声谱图/WAV 导出、追加抽样 |
| `src/bat_classifier/curation/annotations.py` | 标注的加载/校验/原子保存/纯度统计/训练集导出 |
| `src/bat_classifier/curation/model_comparison.py` | Cluster 四类代表样本 × 物种模型对照统计（一致率/校准概率/状态） |
| `src/bat_classifier/species_model.py` | 已确认物种模型：训练（RF+概率校准）/导入校验/录音级 group split/推理 |
| `src/bat_classifier/curation/__main__.py` | 命令行入口 |

复用既有 `audio_io`/`detection`/`features`/`separation`/`clustering`/`visualization` 模块，不复制算法代码。

## ⑦ 片段识别与叫声统计（Segment → 物种 → 叫声次数）

> 统计单位是**一次声学 call / pulse**：既支持已切割好的 WAV 片段，
> 也支持**超长整段录音**（3 小时以上、未来整夜连续 WAV）——长录音走
> 流式 chunk 管线（见下节"超长录音流式分析"），内存与时长无关。
> 一个 WAV 内部可能包含 1、3、8、15 直至数千次独立叫声（call / pulse）。
> 本页只做 **Inference + Counting**，绝不重新训练（无 `fit`、无重新 KMeans、无伪标签）。

### 两个独立任务（不混淆）

- **Task A 分类**：这一段是哪种蝙蝠？用**已训练好的确定模型**逐叫声预测（不重新训练、不用 Cluster 代替物种）。
- **Task B 计数**：这一段内部有几次叫声？复用 `detection.py` 的检测 + 边界精修 + 去重 + 质量筛选，保留的独立 pulse 数即叫声数。

### 模型粒度：pulse-level（脉冲级）

当前"确定模型"（`noise_bat_ABCD_model.joblib`）的 `temporary_classifier` 是**逐 pulse 训练**的（一行 = 一个 pulse = 一个声型标签），因此本页采用：

```text
segment_001.wav
   ↓ 检测 7 个 pulse
   ↓ pulse_001…pulse_007 逐个分类
   ↓ 按该 WAV 聚合
Species A × 7  →  segment_001.wav = Species A，call count = 7
```

若一段内出现多个类别，**保留两层结果**，绝不静默覆盖，也绝不把整段 total 归给 major：

```text
segment_002.wav：A=7, B=2, uncertain=1
  segment_major_label = A
  segment_label_purity = 7 / 9（= 已分类 call 中 major 的占比）
  总体统计：A += 7，B += 2，uncertain += 1   （绝不是 A += 10）
```

### 四层计数（严格区分）

| 计数 | 含义 |
|---|---|
| `candidate_call_count` | 检测到的候选叫声 |
| `valid_call_count` | 通过质量筛选（accepted）的 |
| `noise_count` | 被 noise/bat 门控判为噪声（不计入叫声） |
| `bat_call_count` / **`detected_call_count`** | 通过 noise/bat 判为蝙蝠的——**有效蝙蝠叫声次数（主结果）** = 各类别 + uncertain |
| `classified_call_count` | 成功进入具体类别的 |

主界面默认显示 **有效蝙蝠叫声次数**，不把噪声算进叫声。若无可用模型则只做计数，`label = unavailable`。

### 流程

```text
选择单个 WAV / WAV 文件夹（已切割片段）
  → 模型状态（② 训练模型 / 已导入模型可切换；若为 A-D 临时声型模型则显式警告"不是专家确认物种模型"）
  → 逐个片段：load → 检测+精修+去重 → 逐 pulse 建表（含分离+质量筛选）
      → noise/bat 门控 → 逐 pulse 物种分类（特征顺序按 metadata + StandardScaler + classifier）
      → 低于最低置信度 → 标记 uncertain（阈值可配；模型无 predict_proba 则隐藏，不伪造置信度）
      → 按片段聚合 label_counts / major_label / purity
  → 片段汇总 + 总体类别统计（各类别按实际 call 数累加）
```

### 超长录音（3 小时以上整段 WAV）流式分析

⑦ 直接接受整条长录音（含未来整夜连续 WAV），**内存与文件总时长无关**：

```text
流式 chunk 管线（segment_stream.py）
  ├─ chunk + overlap 分块：coverage 连续铺满 0→总时长（FULL RECORDING，绝不抽样）
  │    读块 = coverage ± overlap（默认 10 s / 0.5 s；重叠由最大脉冲+padding+精修
  │    上下文+滤波瞬态推导，太小直接报错）
  ├─ 预处理与 load_audio 同口径（重采样→去均值→高通→峰值归一），
  │    全局 mean/peak 两遍流式扫描得到；短文件走同一路径，与旧结果逐位一致
  ├─ 跨块不漏不重：脉冲按“中心时间”归属 chunk，块间重叠区内重复
  │    用 remove_duplicate_boundary_pulses 同规则去重（重叠≥0.50 或中心距≤1.5ms）
  ├─ 时间统一转 global 绝对时间（chunk_start + local）
  ├─ 静音快筛（可选）：1ms 包络 + 保守三重条件才跳过 STFT/精修/特征/推理；
  │    比检测器自身门限更保守，绝不误删真实叫声（快筛不改变统计口径）
  ├─ 有限并行（默认 2 worker，0=auto→min(4,cpu)）：结果按 chunk 序归并，
  │    计数与单线程完全一致
  └─ 增量落盘 + checkpoint：每 chunk 写 pulse_results.partial.csv + run_state.json，
       stop 在 chunk 边界优雅退出（status=cancelled），下次可断点续跑
       （校验文件身份 path/size/mtime/内容哈希；水位线截断 + 跨块去重，绝不双算）
```

- **速率分母 = 有效监测总时长**：`calls_per_second = 有效蝙蝠叫声 ÷ valid_monitoring_duration_s`；
  静音快筛跳过的 chunk 只是计算优化，**仍算在监测时间内**；只有损坏/无法解码区间才从分母排除（`failed_duration_s`）。
- **计数与导出彻底分离**：`SEGMENT_MAX_PULSE_WAV_EXPORTS`（默认 5000，0=不限）只限制
  磁盘上的 pulse WAV 数量，**绝不影响 detection/计数/叫声率/CSV**；超限在详情页显式提示。
- **块级进度**：已处理 47/182 分钟、速度约 38× real-time、已检测 call 数、分块 n/N、
  快筛跳过块数；停止按钮在 chunk 边界优雅生效，已处理部分即时可见（增量 CSV）。
- **overview 声谱图**用流式累计的低分辨率摘要（限制 time bins）绘制，
  **绝不为绘图生成整条长录音的完整 STFT**；图像仅可视化，不参与 call 统计。

### 输出目录（独立，不污染 predictions/curation/tables）

```text
results/segment_analysis/<run_id>/
├── run_metadata.json
├── segment_summary.csv            # 第二层：片段汇总（major_label / total_calls / A / B / … / uncertain / purity）
├── pulse_predictions.csv          # 第一层：逐叫声明细（pulse_id / start_s / end_s / label / confidence / 峰值频率 / snr / pulse_file）
├── overall_species_summary.csv    # 第三层：总体统计（species / segment_count / call_count）
├── segments/
│   └── segment_001/
│       ├── segment_001__p001.wav  # 每段独立目录，文件名带段前缀，拿出目录也知道来自哪段
│       ├── segment_001__p002.wav
│       ├── pulse_results.csv      # 逐叫声明细（长录音增量写入后原子替换）
│       └── .checkpoint/           # 断点续跑状态（run_state.json + partial CSV，完成后清理）
└── figures/
    └── segment_001.png            # 整段声谱图 + 每次叫声标注（Call n + 预测标签）
```

`overall_species_summary.csv` 中两个概念分开：`segment_count` = 有多少段 WAV 出现该类别；`call_count` = 这些 WAV 中该类别总共检测到多少次叫声。

### 科研语义（务必保持）

本功能统计的是**声学检测到的 call 数量**，**不是蝙蝠个体数量**。"Species A calls = 20" 表示模型识别出 20 次 Species A 类型的声学 call，**不代表这里有 20 只 Species A 蝙蝠**。当前部署的是 **A-D 临时声型模型**（KMeans 声学相似组）时，UI 明确显示"临时声型模型，不是专家确认物种模型"；标签中的 `uncertain` 表示低于置信阈值、无法确定，不强迫每个 call 进入某个类别。模型带校准器时置信度为 **CalibratedClassifierCV 校准概率**（`confidence_kind = calibrated_probability`）；未校准时只是 RF 原始 `predict_proba`（**模型支持度，未做概率校准**），逐行以 `confidence_kind` 标注，仅作参考。

### 调用链

```text
⑦ SegmentAnalysisTab（app/segment_analysis_tab.py）
        ↓ SegmentAnalysisWorker（app/workers.py，QThread + 进度/停止/日志）
analyze_segment_folder（segment_analysis.py）
        ↓ 每个 WAV → analyze_segment
        ↓ detect_and_refine_pulses_chunked → create_demo_pulse_rows（检测/精修/去重/分离/质量）
        ↓ apply_noise_gate（noise/bat）
        ↓ classify_pulse_rows（逐 pulse 物种分类 + 不确定标记）
        ↓ 片段聚合（label_counts / major_label / purity）
        ↓ summarize_segment_results（片段汇总 + 总体类别统计）
```

## 模型选择与导入（③ / ④ / ⑦ 共用）

「用哪个模型」由一个共用控件 `app/model_picker.py` 决定（下拉选择 + **导入模型文件…** + 刷新），③ 预测、④ 模型管理、⑦ 片段识别三处各放一个，任一处切换即同步到另外两处，并持久化到工作区 `app_settings.json` 的 `active_model_path`。

| 模型来源 | 位置 | 说明 |
|---|---|---|
| **② 模型训练**产物 | `models/noise_bat_ABCD_model.joblib` | 固定文件名、永远排在列表第一、**默认使用**；训练完成后三处自动刷新 |
| 用户导入 | `models/imported/<文件名>.joblib` | 由「导入模型文件…」按钮复制进来，同名冲突自动加 `_1`/`_2`，**绝不覆盖 ② 的训练产物** |
| 中间件 | `acoustic_type_ABCD_model.joblib`、`noise_bat_gate_model.joblib` | 不是完整两级模型，不进可用列表 |

规则（`src/bat_classifier/model_store.py`，纯文件与结构校验、不训练）：

- 导入前先做结构校验：必须是 dict payload，且 `acoustic_types` 内含 `feature_columns` 与 `temporary_classifier`/`calibrated_classifier` 之一；不满足直接拒绝并提示原因，不写任何文件。
- ⑥ 的物种模型（`models/species/<name>/model.joblib`）结构不同，**不接受**作为 ③/⑦ 的两级模型导入。
- ② 训练的模型因此“可以被使用，但不是唯一可用模型”：列表里始终同时存在 ② 训练模型与所有已导入模型，可随时互相切换；② 重新训练只会更新它自己的固定文件名。
- ② 训练产物自带 `CalibratedClassifierCV` 校准器（⑦ 置信度=校准概率）；导入的旧模型没有校准器时置信度自动退回“未校准的模型支持度”，或用「补校准当前模型…」补上。

### 置信度怎么算（⑦ 逐叫声分类）

```text
pulse 特征（按模型 metadata 的 feature_columns 顺序）
  → classifier.predict()                    得到预测类别
  → classifier.predict_proba() 取该类别那列  得到 confidence ∈ [0,1]
  → confidence < 最低置信度（默认 0.60，⑦ 页面可调）
        → 该 call 标记为 uncertain（可选“保留原标签”）
```

`confidence` 的**性质**由模型决定，逐行写进 `confidence_kind` 列，绝不含糊：

| confidence_kind | 含义 |
|---|---|
| `calibrated_probability` | 模型带 `CalibratedClassifierCV` 校准器时的**校准概率**（默认路径） |
| `model_support` | 模型未校准：分类器原始 `predict_proba`（投票占比），只是**模型支持度** |
| `noise_gate_prob_bat` | 仅 noise 行：第一层门控的 `prob_bat`，不是物种置信度 |

- 有校准器时 **predict / predict_proba 都走校准分类器**，与 ⑥ 的口径一致；校准方式（`sigmoid` / `isotonic`）写在 `run_metadata.json` 的 `model.calibration_method`。
- 模型若没有 `predict_proba`（`supports_probability=False`）→ confidence 为 NaN，⑦ 页面隐藏置信度、**不伪造**，也不会触发 uncertain。
- 第一层 noise/bat 门控是另一回事：用 `prob_bat ≥ 0.55`（`NOISE_BAT_PROBABILITY_THRESHOLD`）决定 bat/noise，与这里的物种置信度无关。

### 概率校准（CalibratedClassifierCV，`src/bat_classifier/calibration.py`）

⑦ 是纯推理页、**绝不 fit**；校准只发生在训练/补校准阶段：

| 阶段 | 入口 | 产出 |
|---|---|---|
| ② 训练 | `clustering.cluster_acoustic_types` 自动校准 | 校准器随模型写进 `noise_bat_ABCD_model.joblib`（`acoustic_types.calibrated_classifier` + `calibration` 元数据） |
| 补校准 | ④ 模型管理「补校准当前模型…」（`CalibrateModelWorker` → `attach_calibration`） | 只 fit 校准器、**不重训、不覆盖原模型**，另存 `models/imported/<名称>_calibrated.joblib` |

科学约束（与 ⑥ 一致，全部阈值在 `config.py` 第 12 节）：

- 校准切分按 `source_recording` **分组隔离**（`StratifiedGroupKFold`）：同一录音的 pulse 绝不同时出现在训练侧与校准侧（`SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION = 6` 条独立录音起）；
- 每个校准折必须覆盖全部类别且每类 ≥ `SEGMENT_MIN_CLASS_SAMPLES`，否则不校准；
- `isotonic` 只在样本量 ≥ `SEGMENT_ISOTONIC_MIN_SAMPLES` 时用，小样本自动降级 `sigmoid` 并留痕；
- **数据不足以安全校准 → 绝不伪造**：不挂校准器，置信度退回 `model_support`，⑦ 页面显式提示“置信度未校准”；
- 补校准的标注 CSV 必须含真值标签列（`acoustic_type`/`label`/`expert_label`/`true_label`/`species` 之一）；**`predicted_label` 不能当真值**（模型不能校准自己）。

## 入口与测试


| 文件 | 作用 |
|---|---|
| `run.py` | 完整训练流程入口：`python run.py` |
| `demo.py`（根目录） | 演示入口：`python demo.py --self-test` |
| `run_demo.bat` | Windows 双击运行演示（内部调用 `demo.py --self-test`） |
| `src/bat_classifier/__main__.py` | 支持 `python -m bat_classifier` |
| `src/bat_classifier/curation/__main__.py` | 批量整理 CLI：`PYTHONPATH=src python -m bat_classifier.curation` |
| `tests/test_demo_smoke.py` | 端到端冒烟测试：临时目录建模、跑内置样例 WAV、检查输出文件 |
| `tests/test_curation.py` | ⑥ 批量整理测试：扫描/去重/质量保留/K 自动选择/代表样本抽样与去重/标注持久化/断点续跑/导出过滤/端到端 |
| `tests/test_segment_analysis.py` | ⑦ 片段识别与叫声统计测试：N-pulse 计数稳定/多 pulse 明细=汇总/多类别聚合不归并 major/文件夹分组不串目录/损坏 WAV 失败隔离/no-call=0/模型缺失提示/置信度与不确定标记/真实模型链守恒 |
| `tests/test_segment_stream.py` | ⑦ 超长 WAV 流式测试：chunk+overlap 网格铺满/边界 pulse 不丢不重/global 时间/30s 12-call 全覆盖与速率分母/快筛保守（弱脉冲不丢、开 关一致）/单线程=并行结果一致/断点续跑=不中断结果一致（含换文件防错误续跑）/导出上限不改计数/短 WAV golden 逐位一致 |
| `tests/test_model_store.py` | 模型仓库测试：payload 结构校验（拒绝非两级模型）/列表跳过中间件并标出不可用项/导入复制到 `models/imported/`/同名加后缀且不覆盖 ② 训练产物/错误类型 |
| `tests/test_calibration.py` | ⑦ 概率校准测试：isotonic 小样本降级/校准切分按 source_recording 分组隔离（train/calib 录音零重合）/数据不足绝不伪造/补校准不重训不覆盖/⑦ 推理链 confidence_kind=calibrated_probability vs model_support/标签列探测 |

## 输出目录

| 目录 | 内容 |
|---|---|
| `results/tables/` | 脉冲明细、检测汇总、聚类结果、聚类中心、代表脉冲、录音声型汇总、门控验证明细 |
| `results/figures/` | 概览声谱图、脉冲放大图、PCA 图、`cluster_examples/`（每类代表脉冲） |
| `results/audio/separated_pulses/` | 分离出的分量音频 |
| `results/predictions/` | 新录音逐片段预测 CSV 与 `demo_summary.json` |
| `results/curation/<run_id>/` | ⑥ 批量整理任务：manifest、脉冲表、代表样本、专家标注、声谱图与切割音频（结构见上节） |
| `models/` | 门控模型、A-D 聚类模型及元数据、合并两级模型（② 训练产物）；`models/imported/` 存放用户通过「导入模型文件…」导入的模型 |
| `data/raw/` | 原始演示 WAV（本地，不提交）；`data/sample/` 内置 4 秒演示 WAV |

## 当前原型运行结果

本仓库附带一次本地样例运行的汇总结果：

| 指标 | 结果 |
|---|---:|
| 原始录音 | 10 条，共 466.88 秒 |
| 检测到的候选片段 | 578 个 |
| 通过质量筛选的脉冲 | 441 个 |
| 被拒绝的噪声/低质量片段 | 137 个 |
| A/B/C/D 声型数量 | 131 / 70 / 210 / 30 |
| KMeans 轮廓系数 | 0.295 |
| noise/bat 原型伪标签一致率 | 99.20% |
| 临时随机森林复现聚类标签一致率 | 96.40% |

轮廓系数 0.295 表明四类声型存在一定结构，但组间仍有明显重叠。因此，这个结果适合证明流程可行，不足以证明四个自然物种被清晰分开。两个“一致率”衡量的是模型对当前规则标签或聚类标签的复现能力，也不是真实物种识别准确率。

### PCA 声型分布

![PCA acoustic clusters](results/figures/cluster_pca_ABCD.png)

### noise/bat 原型伪标签验证

![Noise bat validation](results/figures/noise_gate_confusion_matrix.png)

详细汇总见 [`results/tables/experiment_summary.csv`](results/tables/experiment_summary.csv)、[`recording_cluster_summary.csv`](results/tables/recording_cluster_summary.csv) 和 [`cluster_centers_ABCD.csv`](results/tables/cluster_centers_ABCD.csv)。

## 当前评估说明

当前第一层 `noise/bat` 标签主要来自自动质量规则，A–D 来自 KMeans 聚类。因此，内部验证分数只能说明模型能否复现这套原型标签，不能当成真实物种识别准确率。正式物种分类需要老师或专家确认的标签，并按“原始录音”分组划分训练集与测试集。

正式版本建议至少报告：
- 录音数量、有效脉冲数量与各类别样本量；
- 宏平均 Precision、Recall、F1；
- 混淆矩阵；
- 按录音分组的交叉验证结果；
- 未见过地点或设备上的外部测试结果。

## 测试

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

包含两组：

- `tests/test_demo_smoke.py` — Demo 端到端冒烟：模型自动构建、示例音频读取、超声脉冲检测、noise/bat 两级预测、结果文件生成。
- `tests/test_curation.py` — ⑥ 批量整理与专家标注：扫描（单文件/递归/大小写扩展名/损坏 WAV 不中断）、PCM 内容去重（不同 RIFF 元数据）、accepted/rejected 保留与 rejection_reason 一致性、K 自动选择边界（不超样本数与配置上限、单个 K 失败不拖垮）、四类代表样本抽样正确性与去重、固定种子可复现、标注持久化与重载、断点续跑（只处理剩余文件）、导出过滤（仅 reviewed+标签、噪声可选）、长录音 60 秒分析跨度（首/中/尾窗口全覆盖且不波及 ①②③）、端到端 run 目录结构。

后续计划增加：
- 特征提取单元测试；
- 异常输入测试；
- 不同录音条件稳定性测试。


## 项目亮点
针对野外蝙蝠监测中人工鉴定成本高的问题，设计基于声学信号特征的自动分类方法。
为减少野外录音人工筛查工作量提供自动化原型。

## 已知限制与下一步

1. 获取可靠物种标签，替换 A–D 临时声型。
2. 增加不同地点、设备、距离和背景噪声条件下的数据。
3. 增加命令行参数，取消交互式路径输入。
4. 提供 1–3 个可公开示例 WAV 和可复现的基准结果。

## 工程优化计划

当前版本已完成：

- 数据读取；
- 信号预处理；
- 脉冲检测；
- 特征提取；
- 聚类分析；
- 模型预测；

完整流程集中管理，便于科研实验迭代。

后续工程化版本计划进一步拆分：

- audio_processing.py
- feature_extraction.py
- clustering.py
- classifier.py
- visualization.py

提高代码复用性和维护性。

更完整的方法说明见 [`docs/methodology.md`](docs/methodology.md)。
>
