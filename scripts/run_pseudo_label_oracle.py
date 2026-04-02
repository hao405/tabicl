#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tabicl import TabICLClassifier
from tabicl.experiments import load_oracle_classification_dataset
from tabicl.experiments import run_oracle_pseudo_label_experiment
from tabicl.experiments import write_oracle_experiment_outputs
from tabicl.experiments.dataset_loading import parse_kv_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 TabICL 的 oracle 伪标签分析实验")
    parser.add_argument("--data-root", type=Path, default=None, help="数据集根目录；与 --dataset-name 搭配使用")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset-name", help="数据集目录名（相对 --data-root）")
    dataset_group.add_argument("--dataset-dir", type=Path, help="数据集目录的绝对或相对路径")
    parser.add_argument("--model-path", default=None, help="TabICL checkpoint 路径；为空时按 checkpoint version 自动解析")
    parser.add_argument(
        "--checkpoint-version",
        default="tabicl-classifier-v2-20260212.ckpt",
        help="使用的 TabICL 分类 checkpoint 版本",
    )
    parser.add_argument("--n-estimators", type=int, default=8, help="TabICL 集成成员数量")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size")
    parser.add_argument("--device", default=None, help="推理设备，例如 cpu、cuda、cuda:0")
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False, help="fit() 的 kv_cache 模式")
    parser.add_argument("--max-rounds", type=int, default=3, help="最多进行多少轮伪标签吸收")
    parser.add_argument("--min-added", type=int, default=1, help="当单轮新增样本数小于该阈值时停止")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--merge-val", action="store_true", default=False, help="如果存在验证集，则并入初始有标签集")
    parser.add_argument(
        "--save-full-predictions",
        action="store_true",
        default=False,
        help="额外输出 full_predictions_round_*.csv",
    )
    parser.add_argument("--verbose", action="store_true", help="打印更详细的日志")
    return parser


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.resolve()
    if args.data_root is None:
        raise ValueError("使用 --dataset-name 时必须同时提供 --data-root")
    return (args.data_root / args.dataset_name).resolve()


def build_estimator_factory(args: argparse.Namespace):
    def factory() -> TabICLClassifier:
        return TabICLClassifier(
            n_estimators=args.n_estimators,
            batch_size=args.batch_size,
            model_path=args.model_path,
            checkpoint_version=args.checkpoint_version,
            device=args.device,
            random_state=args.random_state,
            verbose=args.verbose,
        )

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    dataset_dir = resolve_dataset_dir(args)
    output_dir = args.output_dir or (Path.cwd() / "oracle_pseudo_label_results" / dataset_dir.name)

    bundle = load_oracle_classification_dataset(
        dataset_dir,
        merge_val=args.merge_val,
        random_state=args.random_state,
    )

    config = vars(args).copy()
    config["dataset_dir"] = str(dataset_dir)
    config["output_dir"] = str(output_dir)
    if config.get("data_root") is not None:
        config["data_root"] = str(config["data_root"])
    if config.get("dataset_dir") is not None:
        config["dataset_dir"] = str(dataset_dir)
    if config.get("output_dir") is not None:
        config["output_dir"] = str(output_dir)

    result = run_oracle_pseudo_label_experiment(
        X_labeled=bundle.X_labeled,
        y_labeled=bundle.y_labeled,
        X_target=bundle.X_target,
        y_target=bundle.y_target,
        estimator_factory=build_estimator_factory(args),
        dataset_name=bundle.dataset_name,
        config=config,
        kv_cache=args.kv_cache,
        max_rounds=args.max_rounds,
        min_added=args.min_added,
        save_full_predictions=args.save_full_predictions,
    )
    write_oracle_experiment_outputs(
        result,
        output_dir,
        save_full_predictions=args.save_full_predictions,
    )

    summary = result.summary
    logging.info("数据集: %s", bundle.dataset_name)
    logging.info("初始准确率: %.4f", summary["baseline_accuracy"])
    logging.info("最终准确率: %.4f", summary["final_accuracy"])
    logging.info("初始 macro-F1: %.4f", summary["baseline_macro_f1"])
    logging.info("最终 macro-F1: %.4f", summary["final_macro_f1"])
    logging.info("累计吸收样本数: %d", summary["total_selected"])
    logging.info("完成伪标签轮数: %d", summary["total_rounds"])
    logging.info("停止原因: %s", summary["stop_reason"])
    logging.info("结果已写入: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
