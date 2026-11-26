"""片段识别与叫声统计的路径、声学处理、模板特征与流式分析参数。"""

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

RESULTS_DIR = PROJECT_ROOT / "results"

AUDIO_OUTPUT_DIR = RESULTS_DIR / "audio" # 切分/分离音频

SEPARATED_PULSE_DIR = AUDIO_OUTPUT_DIR / "separated_pulses"

TARGET_SAMPLE_RATE = 250_000   # 统一重采样率

LOW_FREQ_HZ = 15_000           # 超声分析频段下限（高通滤波同用）

HIGH_FREQ_HZ = 120_000         # 超声分析频段上限

ENERGY_THRESHOLD_DB = -28.0    # 相对最强能量的检测阈值（还有背景自适应项）

MIN_PULSE_MS = 0.8             # 候选脉冲时长下限

MAX_PULSE_MS = 40.0            # 候选脉冲时长上限

MERGE_GAP_MS = 1.5             # 相邻活动帧间隔小于此值则合并

PULSE_PADDING_MS = 0.4         # 候选区间前后保留的余量

CORE_THRESHOLD_DB = -24.0

CORE_BACKGROUND_MARGIN_DB = 6.0

CORE_MAX_THRESHOLD_DB = -10.0

CORE_CONTEXT_MS = 2.0

CORE_PADDING_MS = 0.20

CORE_MAX_GAP_MS = 0.35

CONTOUR_MIN_DB = -35.0         # 可靠频率轮廓的最低能量

# 边界精修的滞回延伸：严格阈值找到叫声核心后，在叫声自身频带内向两侧延伸，
# 只要能量仍高于噪声底 + 余量就继续（跨过不超过最大间隔的短暂凹陷），
# 找回弱 CF 叫声被切掉的首尾横频段、上扫起点和下扫尾巴。
BOUNDARY_EXTEND_CONTEXT_MS = 10.0      # 兼容旧重试步数；实际一次按完整背景上下文追踪
BOUNDARY_BACKGROUND_CONTEXT_MS = 100.0  # 固定背景估计范围，避免短窗全被叫声覆盖
BOUNDARY_TRACK_STEP_KHZ = 3.0           # 相邻帧轨迹最大搜索步长
BOUNDARY_EXTEND_MARGIN_DB = 6.0        # 每频点固定背景及邻频对比度下限
BOUNDARY_EXTEND_MAX_GAP_MS = 0.5       # 延伸时允许跨过的最长凹陷
BOUNDARY_EXTEND_BAND_BELOW_KHZ = 15.0  # 延伸频带：核心峰值频率下方（含上扫起点与下扫尾巴）
BOUNDARY_EXTEND_BAND_ABOVE_KHZ = 3.0   # 延伸频带：核心峰值频率上方

# 同频相邻碎片合并：精修去重后，起止相隔不超过该间隔、核心峰值频率相差不超过该值的
# 候选视为同一叫声被截成的碎片，合并为一个候选再做分离与特征。实测同频相邻叫声的
# 起点间隔中位数 57–83 ms，5 ms 以内的间隙几乎只出现在同一叫声内部。
FRAGMENT_MERGE_GAP_MS = 5.0
FRAGMENT_MERGE_FREQ_KHZ = 1.5
# 每个候选取核心段平均谱中最显著的几个窄带峰（相对 ±5 kHz 中位数突出不少于该值）；
# 两个候选只要有一个峰频率相同就算同频。只看最强格点会被低频底噪或咔嗒声振铃带偏。
FRAGMENT_PEAK_COUNT = 3
FRAGMENT_PEAK_PROMINENCE_DB = 6.0
# 间隙两侧各取这么长，看主频是否离开共同频率（叫声结尾下扫 / 起点上扫）。
# 任一侧偏离超过阈值说明是两条叫声的交界，不合并。实测两条叫声 7.6/3.6 kHz，同一叫声碎片 0–0.4 kHz。
FRAGMENT_EDGE_MS = 1.5
FRAGMENT_EDGE_MAX_SHIFT_KHZ = 2.0

TEMPLATE_FEATURE_COLUMNS = [
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

# CF 平台特征列：所有模板都记录；CF 族只用它们打分（不看首尾 FM，也不用频率轨迹），
# 因此只保留横频段的 CF 模板也能匹配带首尾 FM 的实测叫声。
PLATEAU_FEATURE_COLUMNS = [
    "cf_plateau_khz",
    "cf_plateau_ms",
    "cf_plateau_spread_khz",
]

# 模板快照保存的全部特征列。
TEMPLATE_MODEL_COLUMNS = TEMPLATE_FEATURE_COLUMNS + PLATEAU_FEATURE_COLUMNS

# 各族参与打分的特征列；未列出的族（CF-FM、FM、其他）用 TEMPLATE_FEATURE_COLUMNS 加频率轨迹。
# CF 族只比平台主频和平台稳定性：CF 类别之间靠频率区分，实测常是截短的片段，
# 平台时长只用于下面的判族，不参与 CF 族打分。
TEMPLATE_FAMILY_FEATURE_COLUMNS = {
    "CF": ["cf_plateau_khz", "cf_plateau_spread_khz"],
}

# 判族：CF 族与 CF-FM 族同时达到最低相似度时，平台时长 ≥ 该值判 CF，否则判 CF-FM。
# 边界滞回延伸后实测叫声的平台变长；17 ms 用 2025 罗汉洞 20250807 人工计数标定
# （CFFM1_67K 与 CF3_66K 的比例与人工一致）。模板本身：CF-FM 平台 5–14 ms，同频段 CF 11.5–72 ms。
TEMPLATE_CF_MIN_PLATEAU_MS = 17.0

# CF 族按频率区间识别：每个 CF 类别取其模板平台主频的 [最低-余量, 最高+余量]，
# 实测平台主频落入区间即判为该类（多个区间都包含时取离最近模板频率最近的类）。
TEMPLATE_CF_FREQUENCY_MARGIN_KHZ = 2.0

# 平台短于该值的声波没有可用横频段，不参与 CF 族区间识别（避免纯 FM 被当成 CF）。
TEMPLATE_CF_MIN_ELIGIBLE_PLATEAU_MS = 2.0

# 兼容旧外部引用。尾段不参与判族，分类器始终忽略旧快照中的该设置。
TEMPLATE_CFFM_MIN_TAIL_RATIO: float | None = None

# CF-FM 尾段特征（features.extract_tail_features）。
TAIL_ACTIVE_MIN_DB = -30.0        # 活动帧：帧峰值不低于全段最强 -30 dB
TAIL_MIN_MS = 1.0                 # 尾段时长下限
TAIL_MAX_MS = 3.0                 # 尾段时长上限（长 CF 不被稀释）
TAIL_MAX_FRACTION = 0.30          # 尾段不超过活动跨度的比例
TAIL_BAND_BELOW_KHZ = 15.0        # 局部带：CF 主频下方范围
TAIL_BAND_ABOVE_KHZ = 3.0         # 局部带：CF 主频上方范围
TAIL_LOW_OFFSET_KHZ = 1.5         # 低于 CF 主频多少算“尾巴能量”
TAIL_BIN_DYNAMIC_RANGE_DB = 25.0  # 丢弃低于该段最强点此值的格点（抑制平坦底噪）

# CF 平台特征（features.extract_plateau_features）。
PLATEAU_ACTIVE_MIN_DB = -30.0     # 平台帧：帧峰值不低于全段最强 -30 dB
PLATEAU_REFERENCE_MIN_DB = -12.0  # 参考频率取这些强帧主频的中位数
PLATEAU_TOLERANCE_KHZ = 1.0       # 主频偏离参考频率不超过该值算在平台上
PLATEAU_MAX_GAP_MS = 0.3          # 平台内允许的最长缺帧

DETECTION_HOP_LENGTH = 256

OVERVIEW_MAX_TIME_BINS = 4000

ENABLE_SIMULTANEOUS_SEPARATION = True

SEPARATION_N_FFT = 1024

SEPARATION_WIN_LENGTH = 256

SEPARATION_MAX_WIN_LENGTH = 1024

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

SEPARATION_HARMONIC_MAX_ORDER = 6

SEPARATION_HARMONIC_RATIO_MAD = 0.015

SEPARATION_HARMONIC_MIN_ENVELOPE_CORR = 0.75

SEPARATION_HARMONIC_MIN_COMMON_FRAMES = 6

SEPARATION_HARMONIC_MIN_OVERLAP = 0.80

SEPARATION_HARMONIC_MIN_RATIO_COVERAGE = 0.90

SEPARATION_HARMONIC_MIN_CONTOUR_CORR = 0.80

SAVE_SEPARATED_PULSE_AUDIO = True

SEGMENT_ANALYSIS_DIR = RESULTS_DIR / "segment_analysis"

SEGMENT_PULSE_SUBDIR = "segments"

SEGMENT_FIGURE_SUBDIR = "figures"

SEGMENT_SEPARATED_SUBDIR = "separated"   # 分离分量源 WAV（供复制，非最终结果）

SEGMENT_MIN_CONFIDENCE = 0.60

SEGMENT_MIXED_PURITY_THRESHOLD = 0.5

SEGMENT_STREAM_CHUNK_SECONDS = 10.0

SEGMENT_STREAM_OVERLAP_SECONDS = 0.75

SEGMENT_STREAM_MAX_WORKERS = 2

SEGMENT_FASTSKIP_ENABLED = True

SEGMENT_FASTSKIP_MARGIN_DB = 6.0

SEGMENT_FASTSKIP_ABS_CAP = 0.02

SEGMENT_FASTSKIP_REL_FACTOR = 2.5

SEGMENT_MAX_PULSE_WAV_EXPORTS = 5000

SEGMENT_CHECKPOINT_EVERY_CHUNKS = 20

SEGMENT_MAX_INMEMORY_ROWS = 20000

# 门控集中度的局部带：近场半径的倍数与最小半宽（kHz）。
# 局部带集中度的平谱基线约 近场带宽/局部带宽，用于判断轨迹处是否存在窄带峰。
GATE_LOCAL_CONCENTRATION_MIN_HALF_WIDTH_KHZ = 6.0

GATE_LOCAL_CONCENTRATION_RADIUS_FACTOR = 4.0

# 检测器自适应门限：背景中位数 + 该余量（dB）。诊断 CSV 按同一数值复现判据。
DETECTION_ADAPTIVE_MARGIN_DB = 10.0

# 检测诊断的子带划分（kHz）：与模板类别所在频段大致对齐。
DETECTION_DIAGNOSTIC_BANDS_KHZ = (
    (15.0, 40.0),
    (40.0, 55.0),
    (55.0, 70.0),
    (70.0, 85.0),
    (85.0, 100.0),
    (100.0, 120.0),
)


# 模板子带检测：每个类别取模板频率轨迹 10–90 分位的范围，两侧加余量（kHz）。
# 在全带检测之外，再在这些子带内独立检测，找回被宽带底噪稀释的弱叫声。
TEMPLATE_BAND_MARGIN_KHZ = 1.5

TEMPLATE_BAND_LOW_PERCENTILE = 10.0

TEMPLATE_BAND_HIGH_PERCENTILE = 90.0
