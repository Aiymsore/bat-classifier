# 结果目录

程序输出按用途分类：

- `figures/`：适合展示的 PCA 图、频谱图和代表性脉冲图。
- `tables/`：检测汇总、聚类中心、验证结果和逐脉冲特征表。
- `predictions/`：新录音的预测结果。
- `audio/`：切分后的脉冲、拒绝片段和分离结果。


## Demo 输出

运行 `python demo.py` 后，会在 `predictions/` 生成逐候选预测 CSV 和 `demo_summary.json`，并在 `figures/spectrograms/`、`figures/pulse_zooms/` 生成可视化。这些属于运行产物，默认由 `.gitignore` 排除。
