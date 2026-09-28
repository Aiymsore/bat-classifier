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
# 8. 第一层 noise/bat 门控
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
