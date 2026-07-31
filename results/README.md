# 结果目录

程序输出按用途分类：

- `figures/`：适合展示的 PCA 图、频谱图和代表性脉冲图。
- `tables/`：检测汇总、聚类中心、验证结果和逐脉冲特征表。
- `predictions/`：新录音的预测结果，通常不提交。
- `audio/`：切分后的脉冲、拒绝片段和分离结果，通常不提交。

GitHub 中建议保留 1–3 张最能说明效果的图，以及 2–4 个体积较小的汇总表。大量中间文件不要全部提交。主 README 只展示关键指标与关键图片，并链接到这里的详细文件。

## Demo 输出

运行 `python demo.py` 后，会在 `predictions/` 生成逐候选预测 CSV 和 `demo_summary.json`，并在 `figures/spectrograms/`、`figures/pulse_zooms/` 生成可视化。这些属于运行产物，默认由 `.gitignore` 排除。
