# Demo 冒烟测试记录

## 测试命令

```bash
python demo.py --self-test --rebuild-model
```

## 测试环境

- 日期：2026-08-01
- Python：3.13.5
- NumPy：2.3.5
- Pandas：2.2.3
- SciPy：1.17.0
- Librosa：0.11.0
- scikit-learn：1.8.0
- SoundFile：0.13.1

## 实际结果

- 进程退出码：0
- 候选片段：61
- bat：51
- noise：10
- bat 占比：83.61%
- 主要临时声型：C
- A/B/C/D 数量：5 / 0 / 46 / 0
- 逐片段 CSV：成功生成
- 汇总 JSON：成功生成
- 频谱图与脉冲局部图：成功生成

Self-test 检查了模型构建、样例音频检测、两级预测、CSV 输出和 JSON 输出。这里记录的是一次 Linux 环境的实际冒烟测试；Windows 下仍需按 `requirements.txt` 安装对应依赖。

> A-D 是当前数据上形成的临时声型，不是经过专家确认的蝙蝠物种。
