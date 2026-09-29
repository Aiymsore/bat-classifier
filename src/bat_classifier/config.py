"""项目路径与算法参数集中配置。

分区顺序与数据流一致：
路径 → 检测 → 边界精修 → 质量门槛 → 特征 → 聚类 → 分块检测 → 同时声波分离 → noise/bat 门控。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    """项目根目录解析顺序：

    1. 环境变量 BAT_WORKSPACE（桌面 App 指定的自定义工作区）；
    2. 打包态（PyInstaller）时 exe 所在目录；
    3. 源码态回退到仓库根目录。
    """
    workspace = os.environ.get("BAT_WORKSPACE")
    if workspace:
        return Path(workspace).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # 原始演示录音（WAV）
RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"       # CSV 结果表
FIGURE_DIR = RESULTS_DIR / "figures"     # 声谱图等图像
AUDIO_OUTPUT_DIR = RESULTS_DIR / "audio" # 切分/分离音频
MODEL_DIR = PROJECT_ROOT / "models"

DEMO_ZIP_CANDIDATES = [
    DATA_DIR / "声波文件.zip",
    DATA_DIR / "声波文件(1).zip",
]

# ---- 结果表 CSV ----
DEMO_PULSE_CSV = TABLE_DIR / "acoustic_pulses.csv"
DEMO_RECORDING_CSV = TABLE_DIR / "recording_detection_summary.csv"
CLUSTERED_PULSE_CSV = TABLE_DIR / "acoustic_clusters_ABCD.csv"
RECORDING_CLUSTER_CSV = TABLE_DIR / "recording_cluster_summary.csv"
CLUSTER_CENTER_CSV = TABLE_DIR / "cluster_centers_ABCD.csv"
CLUSTER_REPRESENTATIVE_CSV = TABLE_DIR / "cluster_representatives.csv"
NOISE_GATE_PREDICTIONS_CSV = TABLE_DIR / "noise_gate_validation.csv"

# ---- 图像与音频输出目录 ----
CLUSTER_PCA_PNG = FIGURE_DIR / "cluster_pca_ABCD.png"
CLUSTER_EXAMPLE_DIR = FIGURE_DIR / "cluster_examples"
DEMO_PREDICTION_DIR = RESULTS_DIR / "predictions"
SPECTROGRAM_DIR = FIGURE_DIR / "spectrograms"
PULSE_ZOOM_DIR = FIGURE_DIR / "pulse_zooms"
SEPARATED_PULSE_DIR = AUDIO_OUTPUT_DIR / "separated_pulses"

# ---- 模型文件 ----
CLUSTER_MODEL_PATH = MODEL_DIR / "acoustic_type_ABCD_model.joblib"
CLUSTER_METADATA_JSON = MODEL_DIR / "acoustic_type_ABCD_metadata.json"
NOISE_GATE_MODEL_PATH = MODEL_DIR / "noise_bat_gate_model.joblib"
COMBINED_DEMO_MODEL_PATH = MODEL_DIR / "noise_bat_ABCD_model.joblib"

# =========================================================
# 2. 信号与候选脉冲检测参数
# =========================================================
TARGET_SAMPLE_RATE = 250_000   # 统一重采样率

LOW_FREQ_HZ = 15_000           # 超声分析频段下限（高通滤波同用）
HIGH_FREQ_HZ = 120_000         # 超声分析频段上限

ENERGY_THRESHOLD_DB = -28.0    # 相对最强能量的检测阈值（还有背景自适应项）
MIN_PULSE_MS = 0.8             # 候选脉冲时长下限
MAX_PULSE_MS = 40.0            # 候选脉冲时长上限
MERGE_GAP_MS = 1.5             # 相邻活动帧间隔小于此值则合并
PULSE_PADDING_MS = 0.4         # 候选区间前后保留的余量

# =========================================================
# 3. 脉冲边界精修参数（refine_pulse_boundary）
# =========================================================
CORE_THRESHOLD_DB = -24.0
CORE_BACKGROUND_MARGIN_DB = 6.0
CORE_MAX_THRESHOLD_DB = -10.0
CORE_CONTEXT_MS = 2.0
CORE_PADDING_MS = 0.20
CORE_MAX_GAP_MS = 0.35

# =========================================================
# 4. 脉冲质量门槛（evaluate_pulse_quality）
# =========================================================
MIN_PULSE_SNR_DB = 4.0
MIN_TRACK_COVERAGE = 0.55
MAX_TRACK_JUMP_KHZ = 35.0

# ---- 可视化相关 ----
TOP_PULSES_PER_FILE = 10
ZOOM_CONTEXT_MS = 3.0
CONTOUR_MIN_DB = -35.0         # 可靠频率轮廓的最低能量

RANDOM_STATE = 42

# =========================================================
# 5. 声型标签与 A-D 声型聚类
# =========================================================
# 声型标签列表（可按需增减，默认 A-D 四类）；
# N_ACOUSTIC_TYPES 保留为兼容别名。
ACOUSTIC_TYPE_LABELS = ["A", "B", "C", "D"]

N_ACOUSTIC_TYPES = len(ACOUSTIC_TYPE_LABELS)


def validate_labels(labels: list[str]) -> list[str]:
    """校验声型标签列表：非空、互不相同、可安全用作输出目录名。"""
    cleaned = [str(label).strip() for label in labels]
    if not cleaned or any(not label for label in cleaned):
        raise ValueError("声型标签不能为空。")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"声型标签不能重复：{cleaned}")
    for label in cleaned:
        if set(label) & set('\\/:*?"<>|'):
            raise ValueError(f"声型标签含非法字符：{label}")
    return cleaned

CLUSTER_FEATURE_COLUMNS = [
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
    "track_coverage",
    "max_track_jump_khz",
]

MAX_CLUSTER_PULSES_PER_RECORDING = 150   # 单录音进入聚类的脉冲上限
REPRESENTATIVE_PULSES_PER_TYPE = 10
MIN_PULSES_FOR_CLUSTERING = 40

ACOUSTIC_TYPE_DESCRIPTIONS = {
    "A": "短促宽带下降FM型",
    "B": "高频窄带或CF型",
    "C": "低频窄带或缓变型",
    "D": "复杂、中间或混合声型",
}

# =========================================================
# 6. 分块检测（长录音抽窗分析）
# =========================================================
DETECTION_CHUNK_SECONDS = 4.0
DETECTION_CHUNK_OVERLAP_MS = 50.0
MAX_ANALYSIS_WINDOWS_PER_RECORDING = 2
SAVE_PER_RECORDING_DIAGNOSTICS = False
DEMO_DETECTION_HOP_LENGTH = 256
OVERVIEW_MAX_TIME_BINS = 4000
MAX_REFINED_PULSES_PER_RECORDING = 350

# =========================================================
# 7. 同时近频声波分离（实验性）
# =========================================================
ENABLE_SIMULTANEOUS_SEPARATION = True

SEPARATION_N_FFT = 1024
SEPARATION_WIN_LENGTH = 256
SEPARATION_HOP_LENGTH = 32
SEPARATION_MAX_COMPONENTS = 3
SEPARATION_MAX_PEAKS_PER_FRAME = 5
SEPARATION_MIN_FREQ_GAP_KHZ = 1.5
SEPARATION_PEAK_PROMINENCE_DB = 2.5
SEPARATION_FRAME_DYNAMIC_RANGE_DB = 22.0
SEPARATION_GLOBAL_MIN_DB = -38.0
SEPARATION_MAX_TRACK_JUMP_KHZ = 5.0
SEPARATION_MAX_TRACK_GAP_FRAMES = 2
SEPARATION_MIN_TRACK_MS = 0.55
SEPARATION_MIN_TRACK_COVERAGE = 0.50
SEPARATION_MIN_TEMPORAL_OVERLAP = 0.45
SEPARATION_MIN_SECONDARY_RELATIVE_DB = -18.0
SEPARATION_MASK_BANDWIDTH_KHZ = 1.2
SEPARATION_HARMONIC_TOLERANCE = 0.06

SAVE_SEPARATED_PULSE_AUDIO = True

# =========================================================
# 8. 批量整理与专家标注（⑥ curation）
# =========================================================
# 独立于 data/raw 训练数据：输入直接读用户选择的外部 WAV，
# 输出只写 results/curation/<run_id>/，不覆盖现有结果目录。
CURATION_DIR = RESULTS_DIR / "curation"

# 聚类数量范围（Cluster 只是声学相似组，与 A-D 声型/物种无关）
# KMeans 本身只要求 K ≤ 样本数-1；这里的上限是给 UI 与自动选 K 的护栏。
CURATION_K_MIN = 2
CURATION_K_MAX = 50
CURATION_AUTO_K_MAX = 20          # Auto 模式的候选 K 上限

# ⑥ 批量整理时每条录音的分析跨度（秒）：
#   短于该时长 → 整条录音全覆盖；
#   长于该时长 → 在整条录音上均匀抽窗，总分析时长 ≈ 该值。
# 演示/预测（①②③）仍沿用 MAX_ANALYSIS_WINDOWS_PER_RECORDING=2（约 2×4 秒）。
CURATION_ANALYSIS_SPAN_S = 60.0

# 自动选 K 与最终拟合的样本上限（超过则等概率抽样拟合，再 predict 全量）
CURATION_MAX_FIT_SAMPLES = 20_000

# 每个 Cluster 默认抽取的代表样本数（四类抽样策略）
CURATION_DEFAULT_N_CENTER = 5
CURATION_DEFAULT_N_BOUNDARY = 5
CURATION_DEFAULT_N_RANDOM = 5
CURATION_DEFAULT_N_OUTLIER = 5

# 混杂 Cluster 的 review hint 阈值（只提示，不是科学结论）
CURATION_MIXED_MIN_REVIEWED = 5
CURATION_MIXED_FRACTION = 0.6

# “增加代表样本”一次补充的数量
CURATION_ADD_SAMPLES = 10

# =========================================================
# 9. 第一层 noise/bat 门控
# =========================================================
NOISE_GATE_FEATURE_COLUMNS = [
    "duration_ms",
    "peak_freq_khz",
    "start_freq_khz",
    "end_freq_khz",
    "frequency_drop_khz",
    "f05_khz",
    "f95_khz",
    "bandwidth_90_khz",
    "slope_khz_per_ms",
    "snr_db",
    "track_coverage",
    "max_track_jump_khz",
]

MAX_NOISE_GATE_ROWS_PER_SOURCE_CLASS = 120
MIN_NOISE_GATE_ROWS_PER_CLASS = 20

NOISE_BAT_PROBABILITY_THRESHOLD = 0.55

# =========================================================
# 10. 片段识别与叫声统计（⑦ segment analysis）
# =========================================================
# 输入是“已切割好的 WAV 片段”，每段内部可能含多次独立叫声（call/pulse）。
# 本功能只做 Inference + Counting，绝不重新训练模型。
# 输出只写 results/segment_analysis/<run_id>/，不污染其他结果目录。
SEGMENT_ANALYSIS_DIR = RESULTS_DIR / "segment_analysis"

# 每段 WAV 的独立输出子目录名（segments/<segment_id>/）
SEGMENT_PULSE_SUBDIR = "segments"
SEGMENT_FIGURE_SUBDIR = "figures"
SEGMENT_SEPARATED_SUBDIR = "separated"   # 分离分量源 WAV（供复制，非最终结果）

# 逐 pulse 分类的默认置信度阈值：低于该值标记为 uncertain。
# 注：当前 RF 的 predict_proba 是模型置信输出，未做概率校准，
# 仅作参考，不代表真实可信度。
SEGMENT_MIN_CONFIDENCE = 0.60
SEGMENT_UNCERTAIN_LABEL = "uncertain"
SEGMENT_UNAVAILABLE_LABEL = "unavailable"   # 无可用分类模型时的占位标签

# 低于置信度时的处理："uncertain" 标记为不确定；"keep" 保留原标签
SEGMENT_LOW_CONFIDENCE_ACTION = "uncertain"

# 纯度（major_label/已分类）低于该值时，UI 提示“混合 / 低一致性”（仅描述，非新物种算法）
SEGMENT_MIXED_PURITY_THRESHOLD = 0.5

# =========================================================
# 11. 已确认物种模型（⑥ 模型对照专用）与四类 representative 统计
# =========================================================
# 与 A-D 临时声型模型 / noise-gate 门控模型完全分开保存，绝不互相覆盖。
SPECIES_MODEL_DIR = MODEL_DIR / "species"

# 物种模型严格使用的固定特征列（顺序即模型输入顺序，绝不静默改序/补值）。
SPECIES_FEATURE_COLUMNS = list(CLUSTER_FEATURE_COLUMNS)

# 导入模型时允许要求的特征全集（当前流水线能提取的全部列）
SPECIES_AVAILABLE_COLUMNS = [*CLUSTER_FEATURE_COLUMNS, "snr_db"]

# ---- RandomForest 超参（沿用项目成熟配置） ----
SPECIES_RF_N_ESTIMATORS = 600
SPECIES_RF_MIN_SAMPLES_LEAF = 2
SPECIES_RF_CLASS_WEIGHT = "balanced_subsample"

# ---- Probability calibration ----
# 默认 sigmoid；isotonic 只允许在样本量足够时使用（SPECIES_ISOTONIC_MIN_SAMPLES）。
SPECIES_CALIBRATION_METHOD = "sigmoid"     # "sigmoid" | "isotonic" | "auto"
SPECIES_ISOTONIC_MIN_SAMPLES = 1000
SPECIES_CALIBRATION_FOLDS = 3
# 少于该条独立录音数时不做概率校准（防止同录音泄漏校准集），UI 必须明示：
# “当前数据不足以进行可靠概率校准”。
SPECIES_MIN_RECORDINGS_FOR_CALIBRATION = 6
SPECIES_MIN_RECORDINGS_FOR_SPLIT = 4
SPECIES_MIN_CLASS_SAMPLES = 2              # 每类最少样本（再少无法分组划分）
SPECIES_WARN_CLASS_SAMPLES = 5             # 每类少于该值给出不平衡提示
SPECIES_GROUP_TEST_FRACTION = 0.25
SPECIES_SPLIT_MAX_ATTEMPTS = 40

# ---- Uncertain / 阈值（可配置，UI 提供输入） ----
SPECIES_UNCERTAIN_LABEL = "uncertain"
SPECIES_UNCERTAIN_THRESHOLD = 0.60

# ---- 模型对照（Cluster 四类 representative）状态判定 ----
# 全部阈值集中在这里，UI/算法层不得硬编码。
MODEL_COMPARE_MIN_GROUP_SAMPLES = 3        # center / boundary 组最少有效预测数
MODEL_COMPARE_MIN_OVERALL_SAMPLES = 5      # 综合统计最少有效预测数

MODEL_COMPARE_HIGH_CENTER_AGREEMENT = 0.90
MODEL_COMPARE_HIGH_BOUNDARY_AGREEMENT = 0.80
MODEL_COMPARE_HIGH_OVERALL_AGREEMENT = 0.85
MODEL_COMPARE_HIGH_OVERALL_PROBABILITY = 0.80
MODEL_COMPARE_MIXED_OVERALL_AGREEMENT = 0.70

# Cluster 模型对照状态常量（不要输出虚假的“绝对正确”）
MODEL_STATUS_HIGH = "HIGH_CONSISTENCY"
MODEL_STATUS_MODERATE = "MODERATE_CONSISTENCY"
MODEL_STATUS_LOW = "LOW_CONSISTENCY"
MODEL_STATUS_MIXED = "MIXED"
MODEL_STATUS_NO_MODEL = "NO_MODEL"
MODEL_STATUS_INSUFFICIENT = "INSUFFICIENT_SAMPLES"
MODEL_STATUS_INSUFFICIENT_OOF_COVERAGE = "INSUFFICIENT_OOF_COVERAGE"  # 合法 OOF 覆盖不足 → 禁传播

# 判断为“与训练数据重合”时打的标记（避免误认为模型在泛化）
MODEL_OVERLAP_FLAG = "TRAINING_DATA_OVERLAP"

# =========================================================
# 12. 概率校准（⑦ 片段识别的置信度校准，CalibratedClassifierCV）
# =========================================================
# ⑦ 推理页绝不 fit：校准只发生在 ② 训练（clustering）或“补校准”阶段。
# 校准失败/数据不足 → 不伪造：置信度退回未校准的模型支持度并显式标注。
SEGMENT_CALIBRATION_METHOD = "sigmoid"     # "sigmoid" | "isotonic" | "auto"
SEGMENT_ISOTONIC_MIN_SAMPLES = 1000        # isotonic 只允许在样本量足够时用
SEGMENT_CALIBRATION_FOLDS = 3              # 校准折数（同时受最小类别样本数约束）
# 按 source_recording 分组校准的最少独立录音数；低于该值不做校准
# （防止同录音脉冲同时进入训练侧与校准侧）。
SEGMENT_MIN_RECORDINGS_FOR_CALIBRATION = 6
SEGMENT_MIN_CLASS_SAMPLES = 2              # 每类最少样本（再少无法安全校准）

# =========================================================
# 13. ⑦ 超长 WAV 流式分析（segment_stream；只影响 ⑦，其余模块不动）
# =========================================================
# chunk 时长（秒）：处理/检测单元。整条 WAV 按此切块 + overlap 流式分析，
# 内存与文件总时长无关。建议 10~30 秒。
SEGMENT_STREAM_CHUNK_SECONDS = 10.0
# chunk 之间必须重叠：真实 pulse 可能跨边界。该值必须覆盖
# MAX_PULSE_MS + 2*PULSE_PADDING_MS + 2*CORE_CONTEXT_MS + 2*CORE_PADDING_MS
# + MERGE_GAP_MS（约 47 ms）以及滤波/重采样的瞬态余量；
# segment_stream.build_chunk_specs 启动时校验不足会直接报错。
SEGMENT_STREAM_OVERLAP_SECONDS = 0.5
# 有限并行：STFT/NumPy/sklearn 自身已多线程，worker 过多会 CPU oversubscribe。
# 0 = auto（min(4, cpu)）；绝不默认开满 os.cpu_count()。
SEGMENT_STREAM_MAX_WORKERS = 2
# 静音/无效 chunk 快筛（保守，recall 优先：宁可多留空 chunk，绝不误删叫声）。
# 只有“块内 1ms 包络峰值 < 绝对上限”且“包络峰值低于块内背景 + MARGIN_DB”
# 才跳过；MARGIN_DB=6 远低于检测器自身的 +10 dB 门限——检测器可能看见的
# 事件，快筛绝不跳过。REL_FACTOR：相对整条录音典型块峰值的倍数护栏。
SEGMENT_FASTSKIP_ENABLED = True
SEGMENT_FASTSKIP_MARGIN_DB = 6.0
SEGMENT_FASTSKIP_ABS_CAP = 0.02
SEGMENT_FASTSKIP_REL_FACTOR = 2.5
# 导出上限：只影响磁盘输出数量，绝不动 detection/classification/计数/叫声率。
# 0 = 不限。短文件（低于该值）行为与旧实现完全一致。
SEGMENT_MAX_PULSE_WAV_EXPORTS = 5000
# 增量落盘/断点续跑：每处理 N 个 chunk 写一次 pulse_results.partial.csv + run_state.json
SEGMENT_CHECKPOINT_EVERY_CHUNKS = 20
# 结果行内存上限：超过后只在内存保留样本用于 UI，全量仍逐块写进 CSV
SEGMENT_MAX_INMEMORY_ROWS = 20000

# =========================================================
# 14. 专家数据划分与伪标签反哺（⑥ Expert Train Pool / Evaluation Set）
# =========================================================
# 专家真值按 source_recording 整体拆分（硬约束：绝不拆同一录音的 pulse）：
#   Expert Train Pool     → 训练 / 校准 / OOF scoring / 伪标签传播依据
#   Expert Evaluation Set → 冻结，只做最终评估（Precision/Recall/F1/Macro-F1），
#                           永久禁止参与训练/校准/OOF/传播（EVAL_CONTAMINATED 断言）
EXPERT_SPLIT_EVAL_FRACTION = 0.20        # 目标比例（真正拆分单位是录音，不是 pulse）
EXPERT_SPLIT_RANDOM_STATE = RANDOM_STATE
EXPERT_SPLIT_MIN_TRAIN_PER_CLASS = 1     # 每物种 Train 侧至少保留的录音数
EXPERT_SPLIT_MIN_EVAL_PER_CLASS = 1      # 每物种 Evaluation 侧至少需要的录音数
EXPERT_SPLIT_STATUS_OK = "OK"
EXPERT_SPLIT_STATUS_INSUFFICIENT = "INSUFFICIENT_EVAL_COVERAGE"

# 数据泄漏断言状态（评估录音混入训练/传播数据时立即终止并列出录音 ID）
EVAL_CONTAMINATED_FLAG = "EVAL_CONTAMINATED"

# ---- 伪标签反哺（Cluster 核心区 → 物种模型自训练） ----
PSEUDO_MIN_REVIEWED = 10                 # 簇内（Train Pool）最少已审样本
PSEUDO_MIN_DOMINANT_FRACTION = 0.90      # 簇纯度门槛（10 审 ≥ 9 同标签）
PSEUDO_CORE_DISTANCE_QUANTILE = 0.95     # 核心区半径 = 已审一致样本距离 P95
PSEUDO_CORE_MIN_RADIUS_SAMPLES = 3       # 定半径样本不足时退化为全簇 P25
PSEUDO_OOF_MIN_RECORDINGS = 4            # 独立录音不足 → OOF 不可用 → 零传播
PSEUDO_OOF_FOLDS = 5                     # recording-level cross-fitting 折数上限
PSEUDO_MIN_OOF_CLUSTER_COVERAGE = 0.80   # 合法 OOF 预测覆盖 Train-side 簇脉冲的比例门槛
                                         # （0.80 为初始设计值，请结合项目实际数据检查；
                                         #  coverage 不足 → INSUFFICIENT_OOF_COVERAGE，禁传播，
                                         #  绝不 fallback Production 模型）
PSEUDO_MIN_MODEL_PROBABILITY = 0.90      # 全局默认策略（非理论假设，见 per-class 预留）
PSEUDO_MIN_MODEL_PROBABILITY_UNCALIBRATED = 0.95
# margin 在 P(top1) ≥ 0.90 时数学上恒 ≥ 2t-1 = 0.80，本值是下限护栏/异常数据拦截器；
# 若需区分“0.96/0.02 vs 0.91/0.08”，应提高 top1 阈值或将 margin 设 ~0.85。
PSEUDO_MIN_PROBABILITY_MARGIN = 0.20
PSEUDO_CLASS_THRESHOLDS = {}             # 预留 per-species 阈值覆盖（第一版不自动学习）

PSEUDO_MAX_PER_CLUSTER_FACTOR = 5        # 每簇伪标签 ≤ 已审数 × 5
PSEUDO_MAX_PER_CLUSTER = 500
PSEUDO_MAX_PER_RECORDING_PER_CLASS = 30
PSEUDO_TIME_BUCKET_S = 1.0               # 同录音时间桶去冗余（每桶最多 1 条）
PSEUDO_MAX_PER_TIME_BUCKET = 1
PSEUDO_DISTANCE_STRATA = 3               # 核心区分层数（多样性抽样）

# 局部专家邻域保护（防大簇内部未被代表样本覆盖的异质区域误传播）
PSEUDO_LOCAL_EXPERT_K = 5                # 最近专家邻居上限
PSEUDO_LOCAL_EXPERT_MIN_NEIGHBORS = 3    # 少于该数 → local_expert_check=insufficient
PSEUDO_MIN_LOCAL_EXPERT_AGREEMENT = 0.80 # 局部一致率门槛（A A A A B → 0.8 通过）
PSEUDO_LOCAL_EXPERT_RADIUS_FACTOR = 1.0  # 邻域半径 = core_radius × 该系数

PSEUDO_SAMPLE_WEIGHT = 0.4               # 伪标签训练权重（expert = 1.0）
PSEUDO_MAX_ROUNDS = 2                    # 自训练轮次硬上限
PSEUDO_MAX_RECALL_DROP = 0.02            # 采纳容忍的单类 recall 下降
