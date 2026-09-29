"""批量整理与专家标注（⑥）：大规模未标注 WAV 的自动筛选、聚类与专家辅助审核。

对外常用入口：

    from bat_classifier.curation import CurationParams, CurationRun

    run = CurationRun.create(input_path)     # 或 CurationRun.open(run_dir)
    run.stage_scan()                          # Stage 1 扫描
    run.stage_process(params)                 # Stage 2 检测/筛选/切割（可续跑）
    run.stage_cluster(params)                 # Stage 3 聚类 + 代表样本
    run.stage_export()                        # 导出专家标注训练数据集

命令行（可选）：python -m bat_classifier.curation --input <wav或文件夹>

语义提醒：Cluster ≠ 物种 ≠ A-D 声型；真实物种标签只来自专家人工鉴定。
"""

from __future__ import annotations

from .annotations import AnnotationStore, export_training_dataset
from .model_comparison import run_model_comparison
from .pipeline import CurationParams, CurationRun, create_run, open_run
from .representatives import (
    add_cluster_representatives,
    select_representatives,
)

__all__ = [
    "AnnotationStore",
    "CurationParams",
    "CurationRun",
    "add_cluster_representatives",
    "create_run",
    "export_training_dataset",
    "open_run",
    "run_model_comparison",
    "select_representatives",
]
