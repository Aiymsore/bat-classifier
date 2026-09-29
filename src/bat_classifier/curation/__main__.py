"""批量整理与专家标注的可选命令行入口。

    python -m bat_classifier.curation --input D:\\BatMonitoring\\2026
    python -m bat_classifier.curation --resume results/curation/20260928_143500
    python -m bat_classifier.curation --list-runs
    python -m bat_classifier.curation --export results/curation/20260928_143500

桌面 App 是主入口；本 CLI 只为未来服务器跑批提供低成本通道。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import CURATION_K_MAX, CURATION_K_MIN
from .pipeline import CurationParams, CurationRun
from .representatives import add_cluster_representatives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bat_classifier.curation",
        description=(
            "大规模未标注 WAV 的批量整理与专家标注："
            "扫描 → 检测/筛选/切割 → 聚类 → 代表样本 → 导出专家标注数据。"
            "Cluster 只是声学相似组，不是物种。"
        ),
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="WAV 文件或文件夹（新建任务）。")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="已有任务目录（继续处理/重新聚类）。")
    parser.add_argument("--k", type=str, default="auto",
                        help=f"聚类数量：auto（默认）或 "
                             f"{CURATION_K_MIN}~{CURATION_K_MAX} 的整数。")
    parser.add_argument("--center", type=int, default=None,
                        help="每 Cluster 中心样本数。")
    parser.add_argument("--boundary", type=int, default=None,
                        help="每 Cluster 边界样本数。")
    parser.add_argument("--random", type=int, default=None,
                        help="每 Cluster 随机样本数。")
    parser.add_argument("--outlier", type=int, default=None,
                        help="每 Cluster 离群样本数。")
    parser.add_argument("--retry-failed", action="store_true",
                        help="续跑时重试 failed 文件。")
    parser.add_argument("--save-rejected-audio", action="store_true",
                        help="同时切割保存被拒绝脉冲的 WAV（耗磁盘）。")
    parser.add_argument("--skip-cluster", action="store_true",
                        help="只做扫描与处理，不聚类。")
    parser.add_argument("--add-samples", type=str, default=None,
                        help="为指定 Cluster 追加代表样本，例如 008。")
    parser.add_argument("--export", type=Path, nargs="?",
                        const=Path("__from_run_dir__"), default=None,
                        help="导出 expert_training_dataset.csv："
                             "可直接跟任务目录，或省略路径改用 --run-dir。")
    parser.add_argument("--include-noise", action="store_true",
                        help="导出时包含 noise 状态的样本。")
    parser.add_argument("--list-runs", action="store_true",
                        help="列出已有任务目录。")
    return parser


def _build_params(args: argparse.Namespace) -> CurationParams:
    params = CurationParams()
    text = str(args.k).strip().lower()
    if text in ("auto", ""):
        params.k_mode = "auto"
    else:
        params.k_mode = "manual"
        params.k = int(text)
    for name, value in (
        ("n_center", args.center),
        ("n_boundary", args.boundary),
        ("n_random", args.random),
        ("n_outlier", args.outlier),
    ):
        if value is not None:
            setattr(params, name, int(value))
    params.retry_failed = bool(args.retry_failed)
    params.save_rejected_audio = bool(args.save_rejected_audio)
    return params.validate()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_runs:
        runs = CurationRun.list_runs()
        if not runs:
            print("还没有批量整理任务。")
        for run_dir in runs:
            print(run_dir)
        return 0

    if args.export is not None:
        if args.export.name == "__from_run_dir__":
            if args.run_dir is None:
                print("--export 需要任务目录：--export <任务目录> "
                      "或 --run-dir <任务目录> --export。")
                return 2
            target = args.run_dir
        else:
            target = args.export
        path = CurationRun.open(target).stage_export(
            include_noise=args.include_noise
        )
        print(f"专家标注训练数据集：{path}")
        return 0

    if args.add_samples is not None:
        if args.run_dir is None:
            print("--add-samples 需要配合 --run-dir 指定已有任务。")
            return 2
        run = CurationRun.open(args.run_dir)
        params = _build_params(args)
        added = add_cluster_representatives(
            run, args.add_samples.strip(), params
        )
        print(f"Cluster {args.add_samples} 追加代表样本 {len(added)} 个。")
        return 0

    if args.input is None and args.run_dir is None:
        build_parser().print_help()
        return 2

    params = _build_params(args)

    if args.run_dir is not None:
        run = CurationRun.open(args.run_dir)
        print(f"继续已有任务：{run.run_dir}")
    else:
        run = CurationRun.create(args.input)
        print(f"新建任务：{run.run_dir}")

    last_percent = {"value": -1}

    def _progress(current: int, total: int) -> None:
        percent = int(current * 100 / total) if total else 100
        if current == total or current % 10 == 0 or percent != last_percent["value"]:
            last_percent["value"] = percent
            print(f"[{current}/{total}] {percent}%")

    stats = run.stage_process(params, progress=_progress)
    print(
        f"处理完成：候选 {stats['candidate_pulses']}，"
        f"合格 {stats['accepted_pulses']}，拒绝 {stats['rejected_pulses']}，"
        f"重复 {stats['duplicate_files']}，错误 {stats['failed_files']}"
    )

    if args.skip_cluster:
        return 0 if stats["failed_files"] == 0 else 1

    result = run.stage_cluster(params)
    if not result.get("ok"):
        print(f"聚类未完成：{result.get('error')}")
        return 1
    print(
        f"聚类完成：K={result['selected_k']}，"
        f"{result['cluster_count']} 个 Cluster，"
        f"代表样本 {result['representative_count']} 个；"
        f"轮廓系数 {result['silhouette']}"
    )
    print("提醒：Cluster 是声学相似组，不等于物种数量。")
    print(f"任务目录：{run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
