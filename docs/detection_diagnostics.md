# 检测诊断 CSV（逐块子带能量与阈值）

用于回答"为什么这个文件一个候选都没有/为什么漏检"。开关默认关闭：界面勾选「输出检测诊断 CSV（逐块子带能量/阈值）」，或设置 `SegmentAnalysisConfig.save_detection_diagnostics=True`。

**只读**：它按同一份音频、同一套判据重算一遍并记录，检测结果、计数、图片都不受任何影响。

## 产物

`results/segment_analysis/<run_id>/detection_diagnostics.csv`，每个音频块 × 每个频带一行；同时 `segment_summary.csv` 多出两列（见下）。

| 列 | 含义 |
|---|---|
| `segment_file`、`chunk_index` / `chunk_count` | 文件与块序号 |
| `coverage_start_s` / `coverage_end_s` | 该块负责的时间区间（global 秒） |
| `skipped` | 该块是否被静音快筛跳过 |
| `chunk_raw_candidates` | 该块最终产生的候选数（与全带行一致，可用于核对） |
| `band_khz` | `all` = 15–120 kHz 全带；其余为子带（15-40、40-55、55-70、70-85、85-100、100-120） |
| `background_median_db` | 该带能量包络的中位数（dB，相对本块该带最大值） |
| `peak_db` | 5 帧平滑后的峰值（dB） |
| `contrast_db` | `peak_db − background_median_db` |
| `threshold_db` | 检测阈值 = `max(-28 dB, background_median_db + 10 dB)` |
| `peak_over_threshold_db` | `peak_db − threshold_db`；≥0 才可能有活动帧 |
| `active_frames` / `total_frames` | 超过阈值的帧数 / 总帧数 |
| `groups` | 活动帧合并后的分组数 |
| `raw_candidates` | 通过时长筛选（0.8–40 ms）的候选数 |
| `refined_candidates` | 再经边界精修后仍然有效的候选数 |

## 怎么读

检测器只有一条判据：**该频带的 `contrast_db` 必须 ≥ 10 dB**（`DETECTION_ADAPTIVE_MARGIN_DB`，与 `detect_pulses` 完全一致）。

- `contrast_db < 10` ⇒ 该带在这个块里**不可能**产生候选，`active_frames` 必然为 0。
- 全带行与检测器逐位一致：`raw_candidates` 等于该块的 `raw_candidate_count`。
- 子带行是"如果只在该频带检测，会看到什么"。若全带 `contrast_db` 很低、而某个子带 `contrast_db` 很高并且有候选，说明**叫声在，但被全带求和稀释了**。

## 片段级两列

`segment_summary.csv`：

- `max_subband_contrast_db`：该文件所有块、所有子带里最大的 `contrast_db`（越高说明至少有一个子带看得见）。
- `subband_only_candidates`：对"全带 0 候选"的块，取各子带候选数的**最大值**再累加（不是求和，避免同一叫声出现在相邻两个子带被重复计数）。它是"多子带检测可能额外看到多少"的上界估计，不是真实候选数。

## 实例（202508070301 / 202508070302）

`202508070301.wav` 六个块的全带 `contrast_db` 只有 5.97–9.02 dB，全部低于 10 dB，所以 `raw_candidate_count=0`；但同文件的子带：

| 子带 | contrast_db（chunk0） | 候选 |
|---|---|---|
| 55–70 | 20.8 | 23 |
| 70–85 | 16.2 | 12 |
| 85–100 | 16.8 | 5 |
| 15–40 / 40–55 / 100–120 | 7.8 / 2.0 / 1.7 | 0 |

也就是：82–86 kHz 的叫声在窄带里很清楚（峰值比背景高 16–21 dB），但全带 15–120 kHz 求和后，宽带底噪把对比度压到 10 dB 以下。`202508070302.wav` 则只在两个安静块（contrast 11.7 / 12.2）产生候选，正好对应 `raw_candidate_count=5`。

## 注意

- 开启后每块会多一次 STFT 与逐带精修，180 个文件的整批 run 会明显变慢；建议只在排查时打开。
- 诊断只覆盖"开了开关之后处理的块"；中途改开关不会让已有块补写。
- 子带候选数**不是**最终结果，只用于判断"值不值得改成多子带检测"。
