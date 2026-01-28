#!/usr/bin/env python3
"""
批量在 TALENT 数据目录上评测 TabICLClassifier 的并行脚本。

用法示例：
  python scripts/bench_talent_tabicl_parallel.py --model-path /path/to/checkpoint --data-root /path/to/TALENT/data --num-gpus 8

说明：
 - 脚本会尝试在每个子目录下寻找单个数据文件或 TRAIN/TEST 文件，若只找到单文件则按 80/20 做分割。
 - 支持并行处理，将数据集分配到多个 GPU 上运行。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
import sys
from typing import Optional, Tuple, Union, List

import json
import numpy as np
import pandas as pd
import time
import multiprocessing
import math
import torch

# ------------------------------
# 数据预处理工具 (unchanged)
# ------------------------------

def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
    """可选：把特征矩阵强制转换为数值型。"""
    X = np.asarray(X)
    if not enabled:
        return X

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    df = pd.DataFrame(X)
    encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')

        if series.isna().equals(numeric_series.isna()):
            encoded[col] = numeric_series
        else:
            string_series = series.astype("string")
            codes, uniques = pd.factorize(string_series, sort=True)
            codes = codes.astype(np.int32)
            if (codes == -1).any():
                codes[codes == -1] = len(uniques)
            encoded[col] = codes

    return encoded.fillna(0).values.astype(np.float32)


def handle_missing_entries(X: np.ndarray, y: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    """处理缺失值并保证 X/y 对齐。"""
    X = np.asarray(X)
    y = np.asarray(y)
    context = context or "dataset"

    df = pd.DataFrame(X)
    y_series = pd.Series(y, index=df.index)

    drop_mask = pd.Series(False, index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')

        if series.isna().equals(numeric_series.isna()):
            nan_mask = numeric_series.isna()
            if nan_mask.any():
                mean_value =  float(numeric_series.mean(skipna=True))
                if np.isnan(mean_value):
                    mean_value = 0.0
                df.iloc[:, col] = numeric_series.fillna(mean_value)
                logging.debug(
                    "%s: 数值列 %s 使用均值 %.6f 填充 %d 个 NaN",
                    context,
                    col,
                    mean_value,
                    int(nan_mask.sum()),
                )
        else:
            nan_mask = series.isna()
            if nan_mask.any():
                drop_mask |= nan_mask

    if drop_mask.any():
        drop_count = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
        y_series = y_series.loc[df.index]
        logging.debug("%s: 删除 %d 行包含字符串缺失值", context, drop_count)

    return df.values, y_series.values


def count_missing(values: np.ndarray) -> int:
    """统计数组中的 NaN/None 数量。"""
    if values is None:
        return 0

    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())

    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(context: str, values: np.ndarray, *, dataset_id: str | None = None,
                     missing_registry: set[str] | None = None) -> None:
    """如果存在缺失值就记录 warning，并可登记到缺失集合。"""
    missing = count_missing(values)
    if missing:
        # logging.warning(f"{context}: 原始数据包含 {missing} 个 NaN/缺失值")
        if dataset_id and missing_registry is not None:
            missing_registry.add(dataset_id)


# ------------------------------
# 数据集文件查找与加载 (unchanged)
# ------------------------------

def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str):
        for name, p in lower_names.items():
            if name.endswith(key):
                return p
        return None

    n_train = find_by_suffix('n_train.npy')
    c_train = find_by_suffix('c_train.npy')
    y_train = find_by_suffix('y_train.npy')
    n_val = find_by_suffix('n_val.npy')
    c_val = find_by_suffix('c_val.npy')
    y_val = find_by_suffix('y_val.npy')
    n_test = find_by_suffix('n_test.npy')
    c_test = find_by_suffix('c_test.npy')
    y_test = find_by_suffix('y_test.npy')

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {'.npy', '.npz', '.csv', '.tsv', '.parquet'}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None

    return None, None, None


def load_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == '.parquet':
        return pd.read_parquet(file_path).values
    sep = '\t' if suffix == '.tsv' else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def load_table(file_path: Union[Path, Tuple], context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[
    np.ndarray, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(Xp, yp, context=context, coerce_numeric=coerce_numeric, dataset_id=dataset_id,
                             missing_registry=missing_registry)
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(
                Path(num_path) if num_path else None,
                Path(cat_path) if cat_path else None,
                Path(y_path),
                context=context,
                coerce_numeric=coerce_numeric,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        raise ValueError(f"Unsupported tuple format for load_table: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == '.parquet':
        df = pd.read_parquet(file_path)
        data = df.values
    else:
        sep = '\t' if file_path.suffix.lower() == '.tsv' else None
        df = pd.read_csv(file_path, sep=sep, header=None)
        data = df.values

    if data.ndim == 1:
        raise ValueError(f"Unsupported 1D data in {file_path}")

    log_target = context or str(file_path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    heuristic_column = None
    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = col0
        X = data[:, 1:]
        heuristic_column = 'first'
    else:
        y = data[:, -1]
        X = data[:, :-1]
        heuristic_column = 'last'

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=log_target)
    X = convert_features(X, coerce_numeric)

    if heuristic_column:
        logging.info(f"{log_target}: 使用单文件启发式拆分标签 (取 {heuristic_column} 列)")
    return X, y


def load_pair(X_path: Path, y_path: Path, context: str = "", coerce_numeric: bool = False,
              dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    X = load_array(X_path)
    y = load_array(y_path)

    ctx = context or X_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=ctx)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(num_path: Optional[Path], cat_path: Optional[Path], y_path: Path, context: str = "",
               coerce_numeric: bool = False, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> \
Tuple[np.ndarray, np.ndarray]:
    features = []
    ctx_base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    if num_path:
        X_num = load_array(num_path)
        X_num = np.asarray(X_num)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        log_nan_presence(f"{ctx_base}-num_raw", X_num, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_num)
    if cat_path:
        X_cat = load_array(cat_path)
        X_cat = np.asarray(X_cat)
        if X_cat.ndim == 1:
            X_cat = X_cat.reshape(-1, 1)
        log_nan_presence(f"{ctx_base}-cat_raw", X_cat, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(X_cat)

    if not features:
        raise ValueError("No numeric or categorical feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feat in enumerate(features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"Feature array #{idx} has mismatched sample count: {feat.shape[0]} vs {n_samples}")

    X = features[0] if len(features) == 1 else np.concatenate(features, axis=1)
    log_nan_presence(f"{ctx_base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)

    y = load_array(y_path)
    y = np.asarray(y)
    log_nan_presence(f"{ctx_base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=ctx_base)
    X = convert_features(X, coerce_numeric)
    return X, y


# 仅允许的分类任务类型
CLASSIFICATION_TASKS = {'binclass', 'multiclass','unknown'}


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    """读取 info.json（任务类型、元信息）。"""
    info_path = dataset_dir / 'info.json'
    if not info_path.exists():
        return None
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"读取 {info_path} 失败: {exc}")
        return None


def summarize_task_types(dirs: list[Path]) -> dict[str, int]:
    counts = {'regression': 0, 'binclass': 0, 'multiclass': 0, 'unknown': 0}
    for dataset_dir in dirs:
        info = load_dataset_info(dataset_dir)
        task_type = None
        if info:
            task_type = str(info.get('task_type', '')).lower()

        if not task_type:
            counts['unknown'] += 1
        elif task_type in counts:
            counts[task_type] += 1
        else:
            counts['unknown'] += 1
    return counts


# ------------------------------
# 核心评测逻辑 (Worker)
# ------------------------------

def evaluate_datasets_worker(rank: int, device_id: int, model_path: str, dataset_dirs: List[Path],
                            verbose: bool = False, skip_regression: bool = True, bins: int = 0,
                            merge_val: bool = False, coerce_numeric: bool = True):
    """
    Worker function to evaluate a subset of datasets on a specific GPU.
    Returns a list of results (name, acc, duration) and a set of datasets with missing values.
    """
    try:
        from tabicl import TabICLClassifier
        from sklearn.utils.multiclass import type_of_target
        from sklearn.preprocessing import KBinsDiscretizer
    except ImportError as e:
        print(f"[Worker {rank}] Import error: {e}")
        return [], set()

    # Set device
    # Use cuda:X logic
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    
    msg_prefix = f"[GPU {device_id}]"

    print(f"{msg_prefix} Initializing model on {device_str} for {len(dataset_dirs)} datasets...")
    
    try:
        clf = TabICLClassifier(verbose=verbose, model_path=model_path, device=device_str)
    except Exception as e:
        print(f"{msg_prefix} Model initialization failed: {e}")
        return [], set()

    results = []
    datasets_with_missing: set[str] = set()
    
    for d in dataset_dirs:
        try:
            info = load_dataset_info(d)
            task_type = None
            if info:
                task_type = str(info.get('task_type', '')).lower()
                if task_type == 'regression':
                    print(f"{msg_prefix} 跳过数据集 {d.name}: task_type=regression")
                    continue
                if task_type and task_type not in CLASSIFICATION_TASKS:
                    print(f"{msg_prefix} 跳过数据集 {d.name}: 未知 task_type={task_type}")
                    continue

            train_path, val_path, test_path = find_data_files(d)
            if train_path is None and test_path is None:
                print(f"{msg_prefix} 跳过：{d.name} (无数据文件)")
                continue

            if train_path and test_path:
                X_train, y_train = load_table(train_path, context=f"{d.name}-train", coerce_numeric=coerce_numeric,
                                              dataset_id=d.name, missing_registry=datasets_with_missing)
                X_test, y_test = load_table(test_path, context=f"{d.name}-test", coerce_numeric=coerce_numeric,
                                            dataset_id=d.name, missing_registry=datasets_with_missing)
            else:
                print(f"{msg_prefix} 数据集：{d.name} (即使是单文件也暂不支持自动拆分或逻辑未触发)")
                val_path = None
                continue

            X_val = y_val = None
            if val_path:
                X_val, y_val = load_table(val_path, context=f"{d.name}-val", coerce_numeric=coerce_numeric,
                                          dataset_id=d.name, missing_registry=datasets_with_missing)
                if X_val.ndim == 3 and X_val.shape[1] == 1:
                    X_val = X_val.squeeze(1)
                if X_val.ndim == 1:
                    X_val = X_val.reshape(-1, 1)
                y_val = np.asarray(y_val)
                if y_val.ndim > 1 and y_val.shape[-1] == 1:
                    y_val = y_val.reshape(-1)
                if merge_val:
                    X_train = np.concatenate([X_train, X_val], axis=0)
                    y_train = np.concatenate([y_train, y_val], axis=0)

            if X_train.ndim == 3 and X_train.shape[1] == 1:
                X_train = X_train.squeeze(1)
            if X_test.ndim == 3 and X_test.shape[1] == 1:
                X_test = X_test.squeeze(1)

            # Target type check
            tgt_type = None
            try:
                tgt_type = type_of_target(y_train)
            except Exception:
                tgt_type = None

            if task_type is None:
                if tgt_type is not None and tgt_type.startswith('continuous'):
                    if bins and bins > 1:
                        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
                        y_train = est.fit_transform(y_train.reshape(-1, 1)).astype(int).ravel()
                        y_test = est.transform(y_test.reshape(-1, 1)).astype(int).ravel()
                    elif skip_regression:
                        print(f"{msg_prefix} 跳过数据集 {d.name}: 检测到连续标签")
                        continue

            ds_start = time.time()
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = float(np.mean(y_pred == y_test))
            duration = time.time() - ds_start

            print(f"{msg_prefix} {d.name}: accuracy={acc:.4f} time={duration:.2f}s")
            results.append((d.name, acc, duration))

        except Exception as e:
            print(f"{msg_prefix} 评测失败 {d.name}: {e}")
            import traceback
            traceback.print_exc()

    return results, datasets_with_missing


def main(argv=None):
    p = argparse.ArgumentParser(description='Parallel Benchmark TabICLClassifier on TALENT datasets')
    p.add_argument('--model-path', default='tabicl-classifier-v1.1-0506.ckpt', help='Path to TabICL checkpoint')
    p.add_argument('--data-root', default='data181', help='Root path to TALENT data folder')
    p.add_argument('--outdir', default='evalution_talent_test', help='Directory to save results')
    p.add_argument('--max-datasets', type=int, default=None, help='Limit number of datasets')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--merge-val', default=True, action='store_true')
    p.add_argument('--num-gpus', type=int, default=8, help='Number of GPUs to use')
    p.add_argument('--no-coerce-numeric', dest='coerce_numeric', action='store_false')
    p.set_defaults(coerce_numeric=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    script_start_time = time.time()
    
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Get all datasets
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if args.max_datasets:
        dirs = dirs[:args.max_datasets]
    
    total_datasets = len(dirs)
    logging.info(f"Total datasets to process: {total_datasets} using {args.num_gpus} GPUs")

    # Split datasets into chunks
    num_gpus = args.num_gpus
    chunk_size = math.ceil(total_datasets / num_gpus)
    chunks = [dirs[i:i + chunk_size] for i in range(0, total_datasets, chunk_size)]
    
    # Prune empty chunks if datasets < num_gpus
    chunks = [c for c in chunks if len(c) > 0]
    
    logging.info(f"Split into {len(chunks)} chunks. Max chunk size: {chunk_size}")

    # Prepare arguments for starmap
    # (rank, device_id, model_path, dataset_dirs, verbose, skip_regression, bins, merge_val, coerce_numeric)
    tasks = []
    for i, chunk in enumerate(chunks):
        # Determine device ID. If we have more chunks than GPUs (unlikely with this logic logic), mod it.
        # Logic here assumes we launch len(chunks) processes.
        # Map process i to gpu i % valid_gpus (if we want to oversubscribe, but here we assume dedicated)
        # Actually user said "8 cards parallel", so mapping rank i to gpu i is fine.
        device_id = i % num_gpus
        tasks.append((
            i, 
            device_id,
            args.model_path,
            chunk,
            args.verbose,
            True, # skip_regression
            0,    # bins
            args.merge_val,
            args.coerce_numeric
        ))

    # Run in parallel using multiprocessing
    # Note: 'spawn' context is safer for PyTorch/CUDA
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        results_list = pool.starmap(evaluate_datasets_worker, tasks)

    # Aggregate results
    all_results = []
    all_missing_registry = set()

    for res, missing in results_list:
        all_results.extend(res)
        all_missing_registry.update(missing)
    
    # Sort results by dataset name for consistency
    all_results.sort(key=lambda x: x[0])

    if all_results:
        # 汇总统计
        total_time = sum(duration for _, _, duration in all_results)
        
        script_duration = time.time() - script_start_time

        csv_path = outdir / 'talent_detailed.csv'
        write_header = not csv_path.exists()
        with open(csv_path, 'a') as f:
            if write_header:
                f.write('dataset,accuracy,time_s\n')
            for name, acc, duration in all_results:
                f.write(f"{name},{acc:.6f},{duration:.3f}\n")
            f.write(f"Average,{avg_acc:.6f},{avg_time:.3f}\n")

        # Summary text
        missing_results = [(name, acc) for name, acc, _ in all_results if name in all_missing_registry]
        missing_names = sorted(name for name, _ in missing_results)
        avg_missing_acc = sum(acc for _, acc in missing_results) / len(missing_results) if missing_results else None

        summary_path = outdir / 'talent_summary.txt'
        with open(summary_path, 'a') as f:
            f.write(f"\n--- Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Total datasets: {len(all_results)}\n")
            f.write(f"Average accuracy: {avg_acc:.6f}\n")
            f.write(f"Total inference time s: {total_time:.3f}\n")
            f.write(f"Average inference time s: {avg_time:.3f}\n")
            f.write(f"Script total execution time s: {script_duration:.3f}\n")
            if missing_results:
                f.write(f"Datasets with NaN values: {len(missing_names)}\n")
                f.write(f"Average accuracy (NaN datasets): {avg_missing_acc:.6f}\n")
                f.write(f"List (NaN datasets): {', '.join(missing_names)}\n")
            else:
                f.write("Datasets with NaN values: 0\n")

        logging.info(f"Evaluation complete. Results saved to {outdir}")
        logging.info(f"Average Accuracy: {avg_acc:.4f}, Average Time: {avg_time:.2f}s, Script Duration: {script_duration
        logging.info(f"Evaluation complete. Results saved to {outdir}")
        logging.info(f"Average Accuracy: {avg_acc:.4f}, Average Time: {avg_time:.2f}s")
    else:
        logging.info("No successful results obtained.")

if __name__ == '__main__':
    main()
