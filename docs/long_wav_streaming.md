# ⑦ 片段/长录音识别与叫声统计 —— 超长 WAV 流式优化报告

> 本文记录本轮优化的**实测基准**（区分实测与推算）、设计要点与正确性结论。
> 功能说明见 README「⑦ 片段识别与叫声统计」与「超长录音流式分析」小节。

## 1. 优化前的三个问题（60/600 秒实测复现）

| 输入 | wall | peak RSS | 计数 / 真实 | 叫声率 |
|---|---|---|---|---|
| 60 s，30 call | 2.42 s | **749 MB** | 4 / 30 | 0.07/s（真值 0.5/s） |
| 600 s，300 call | 7.05 s | **5069 MB** | 4 / 300 | 0.007/s（真值 0.5/s） |

1. **P1 内存**：整条 `load_audio`（float64）+ 整条 STFT —— RSS 随音频时长线性
   （实测斜率 ≈ 8 MB/音频秒），3 小时推算 **≈ 86 GB**（60/600 秒两点外推；
   3 小时优化前**未实测**，按 5 GB/10 分钟趋势无法运行）。
2. **P2 抽样统计**：`MAX_ANALYSIS_WINDOWS_PER_RECORDING=2`×4 s → 600 s 只分析约 8 s，
   计数恒为 4，叫声率被全长分母稀释（0.007/s 而非 0.5/s）。
3. **P3/P4 截断**：`MAX_REFINED_PULSES_PER_RECORDING=350` 截断计数；逐 pulse WAV 无上限。

## 2. 优化后（同一合成语料、同一检测/分类链路；`scripts/benchmark_segment.py` 实测）

| 输入 | wall | peak RSS | 计数 / 真实 | 叫声率 | 备注 |
|---|---|---|---|---|---|
| 60 s，30 call | 3.22 s | **658 MB** | **30 / 30** | **0.5/s** | 6 chunks |
| 600 s，300 call | 14.75 s | **636 MB** | **300 / 300** | **0.5/s** | 60 chunks |
| **3 h（10800 s）**，5400 call | **259.9 s（4.3 分钟）** | **680 MB** | **5400 / 5400** | **0.5/s** | 1080 chunks，RF64 5.4 GB |

- **内存与文件总长无关**：60 s/600 s/3 h 的 RSS 分别为 658/636/680 MB——
  3 小时 5.4 GB 文件也只用 680 MB（优化前 600 s 已 5069 MB，3 h 推算 ≈86 GB 无法运行）。
- **计数正确性**：全时段 coverage（FULL RECORDING），不再抽样；
  30/30、300/300、5400/5400 无漏检、无重复；叫声率分母 = `valid_monitoring_duration_s`。
- 速度：3 小时录音 **4.3 分钟**跑完（41.6× real-time）；全量分析比旧“抽 8 秒”
  慢是正确性代价（旧 7.05 s 是因为只分析了 8 秒音频）。

## 3. 设计要点（代码位置）

| 能力 | 实现 | 文件 |
|---|---|---|
| chunk + overlap 网格（coverage 铺满全长） | `ChunkSpec` / `build_chunk_specs`；重叠 ≥ 由最大脉冲+padding+精修上下文+滤波瞬态推导（`minimum_overlap_seconds`），不足直接报错 | `segment_stream.py` |
| 流式预处理与 `load_audio` 同口径 | `StreamingPreprocessor`（块读 + 两遍流式 mean/peak；短文件与旧结果逐位一致） | 同上 |
| 跨块不漏不重 | 脉冲按中心时间归属 chunk；跨块重复用 `remove_duplicate_boundary_pulses` 同规则去重（重叠≥0.50 或中心距≤1.5ms，保高 snr） | 同上 |
| global 时间 | `chunk_start + local`，CSV/图上为绝对时间 | 同上 |
| 保守静音快筛 | `chunk_may_contain_ultrasound`：1ms 包络 + 三重条件（绝对幅度 < −34 dBFS **且** 包络平坦 <背景+6 dB **且** < 整条典型块 2.5×）才跳过 | 同上 |
| 有限并行 | `stream_chunks`：ThreadPoolExecutor（默认 2，0=auto→min(4,cpu)），按 chunk 序归并 | 同上 |
| 增量落盘 + checkpoint | `ChunkCheckpoint`：每 chunk 写 `pulse_results.partial.csv` + `run_state.json`（原子替换）；文件身份 = path/size/mtime/内容哈希 | 同上 |
| 断点续跑 | `analyze_segment(resume=True)`：水位线截断 + 时间键去重，绝不双算；换文件自动拒绝续跑 | `segment_analysis.py` |
| 计数/导出分离 | `create_demo_pulse_rows(max_pulses=None)` 计数不截断；`SEGMENT_MAX_PULSE_WAV_EXPORTS`（5000）只限磁盘 WAV | `pulse_table.py` / `segment_analysis.py` |
| 流式 overview 声谱图 | `SpectrogramSummary`（限制 time bins 的逐块 max 累计）→ `save_segment_call_overview_from_summary`；仅可视化 | `segment_stream.py` / `visualization.py` |
| 块级进度 + 优雅停止 | `chunk_progress`（已处理 x/y 分钟、速度 n×、call 数、分块 n/N、快筛数）；stop 在 chunk 边界生效，checkpoint=cancelled，可续跑 | `app/workers.py` / `app/segment_analysis_tab.py` |

## 4. 正确性验证（`tests/test_segment_stream.py`，21 项）

- chunk 覆盖铺满 / 最小重叠校验 / probe 时长；
- 边界 pulse 不丢不重（横跨 2.0 s 边界、完整落在重叠区各 1 次）；
- global 时间绝对正确；跨块去重保高 snr、同 chunk 多轨迹不去重；
- 30 s×12 call 全检出（修复抽样 bug）、叫声率分母=有效监测时长；
- 快筛保守：平坦静音块跳过、弱脉冲（含安静段弱 call）**绝不跳过**、
  开/关结果逐行一致、快筛不改分母；
- 单线程 vs 3 worker：行/计数/标签分布**完全一致**；
- 断点续跑：中断→续跑 = 不中断基线（计数一致、无重复行）；换文件拒绝续跑；
- 导出上限 1 个 vs 不限时：计数相同，仅 WAV 数受限；
- 短 WAV golden：1 s/2 s/3 s 的计数、叫声率、逐行时间/标签/置信度与优化前**逐位一致**。

全量回归：`unittest discover`（含 ①–⑥ 既有测试）+ `bat_classifier_app.py --selftest` 全绿。

## 5. 已知边界（明确声明）

- **快筛比例**：合成/均匀噪声数据上 `fast_skipped_chunk_count=0`（均匀噪声的包络
  max/median 天然 > 判据，不会被跳过）——快筛只在真正接近数字静音的块触发。
  这是 recall-first 的有意取舍：宁可少跳，绝不丢 call。
- **>4 GB WAV**：3 h @250 kHz PCM_16 ≈ 5.4 GB，超过经典 RIFF 4 GB 上限，
  必须用 RF64/W64 容器（基准脚本自动选 RF64）；不支持的文件给出明确报错
  （`probe_audio`），绝不静默半途崩溃。
- **多文件批量断点续跑**：续跑以“每个 WAV 的 checkpoint”为粒度；批量模式下
  已完整跑完的 WAV 会被重新分析（结果覆盖，不重复计数），未完成的 WAV 从
  checkpoint 继续。整夜单文件（主要场景）精确续跑。
- **分类模型/特征/阈值/标签完全不变**：本轮只改“音频怎么送进现有算法”；
  ⑦ 仍绝不 fit；①–⑥ 业务逻辑未动。
