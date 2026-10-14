# 蝙蝠声波模板分析工具

软件保留两个模块：① 片段识别与叫声统计、② 噪声模板初筛。
前六个模块以及①的旧分类模型模式已移除。历史录音、模板、模型文件和分析结果不会被自动删除。

## 启动

Windows 双击 `run_app.bat`。首次配置环境：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe bat_classifier_app.py
```

`run.py` 和 `python -m bat_classifier`（需设置 `PYTHONPATH=src`）也启动桌面界面。
窗口大小、位置和日志显示状态保存在工作区 `app_settings.json`；可读取旧版设置，旧训练设置不再使用。

## ① 片段识别与叫声统计

1. 选择模板 WAV 文件夹，文件名决定可信类别。
2. 选择待分析的 WAV 或 WAV 文件夹。
3. 设置最低相似度、分离窗、谐波处理及图片输出方式，开始分析。

每个检测到的 pulse 匹配全部模板，最终归入一个类别；相似度低于阈值归入 `unknown`。
相似度不是分类正确率或校准概率。所有同类模板都保留，不做聚类或去重。

| 模板文件名 | 类别 |
|---|---|
| `CF1_47K1.wav`、`CF1_47K2.wav` | `CF1_47K` |
| `CFFM1_67K3.wav` | `CFFM1_67K` |
| `npulse000001.wav` | `noise pulse` |
| `pulse000001.wav` | `可疑声波` |

原始音频转为 250 kHz 单声道，去直流、15 kHz 高通和归一化后，进行检测、边界精修、分量分离与特征提取。
不再提供质量筛选或训练模型的 noise/bat 门控。`noise pulse` 是模板类别名称，不表示自动删除匹配到的声波。

分析固定导出独立 WAV、片段线性声谱概览和逐声波声谱图，不提供关闭开关。
声谱图可选“合并显示”或“分开显示”。分开显示使用分离音频；谐波随所属叫声保留在同一张图，不单独计数。
图片上方显示波形，下方显示声谱，默认 pulse 上下文为 0 ms。可放大、拖动、适应窗口查看。
“直接打开 WAV/SON 查看”支持独立查看录音和 Avisoft SON 声谱文件。

长录音使用连续分块、跨块去重、有限并行与增量输出；停止后可从断点继续。
本次处理格式更新后，旧版未完成断点需重新开始；已输出的文件保留。
统计单位是声学 call，不代表蝙蝠个体数量。

结果在 `results/segment_analysis/<run_id>/`：

- `pulse_predictions.csv`：逐 pulse 匹配及音频/图片路径。
- `segment_summary.csv`、`overall_species_summary.csv`：片段及类别统计。
- `template_model.json`、`run_metadata.json`：模板快照和运行参数。
- `segments/`、`separated/`、`figures/`、`pulse_spectrograms/`：音频、概览和逐声波图片。

## ② 噪声模板初筛

复用①的声波处理，使用“蝙蝠声波模板相似度”阈值（默认 0.2）。只导出严格低于阈值的 pulse；等于阈值不导出。

结果在 `results/noise_template_scan/<run_id>/`：

- `templates/npulse000001.wav`：可直接用于①模板匹配的 WAV。
- `pulse_spectrograms/npulse000001.png`：对应图片。
- `pulse_predictions.csv`、`review_state.json`：明细和人工选择记录。

人工选中后点击“移入候选”，WAV 和图片一起移动并改为 `pulse` 编号：

```text
candidates/templates/pulse000001.wav
candidates/pulse_spectrograms/pulse000001.png
candidates/pulse_predictions.csv
```

第②区域仍叫“候选”；候选 WAV 导入①时按“可疑声波”归类。未移入的 `npulse` 在①中按 `noise pulse` 归类。
人工移动保留原扫描分数与匹配结果，重新打开结果目录后可继续查看。

## 源码与打包

- `app/`：双页桌面界面、后台任务与图片查看器。
- `segment_analysis.py`、`segment_stream.py`：分析、统计、流式运行与断点。
- `template_matching.py`、`noise_template_scan.py`：模板匹配与初筛/候选管理。
- `audio_io.py`、`detection.py`、`pulse_table.py`、`features.py`、`separation.py`、`harmonics.py`：共享声波处理。
- `pulse_export.py`、`visualization.py`、`son_reader.py`：导出与查看。

`build_exe.bat` 使用 `bat_app.spec` 打包。修改源码后重启源码版程序生效；已有 EXE 需重新打包。
科学计算依赖仍需保留；librosa 可能间接安装 scikit-learn/joblib，不能因删除训练功能就强行卸载。
