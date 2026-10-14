"""第①/②模块共享的路径、声学处理、模板特征与流式分析参数。"""

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

SEGMENT_STREAM_OVERLAP_SECONDS = 0.5

SEGMENT_STREAM_MAX_WORKERS = 2

SEGMENT_FASTSKIP_ENABLED = True

SEGMENT_FASTSKIP_MARGIN_DB = 6.0

SEGMENT_FASTSKIP_ABS_CAP = 0.02

SEGMENT_FASTSKIP_REL_FACTOR = 2.5

SEGMENT_MAX_PULSE_WAV_EXPORTS = 5000

SEGMENT_CHECKPOINT_EVERY_CHUNKS = 20

SEGMENT_MAX_INMEMORY_ROWS = 20000
