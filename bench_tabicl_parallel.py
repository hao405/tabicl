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
from pathlib import Path

src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from typing import Optional, Tuple, Union, List

import json
import numpy as np
import pandas as pd
import time
import multiprocessing
import math
import torch

# ------------------------------
# 数据预处理工具
# ------------------------------


def parse_kv_cache(value: str) -> bool | str:
    """将命令行中的 kv_cache 参数解析为 TabICLClassifier.fit() 所需的类型。"""
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache 必须是以下之一: false, true, kv, repr")

def make_feature_frame(
    values: pd.DataFrame | np.ndarray,
    *,
    kind: str = "infer",
    prefix: str = "x",
) -> pd.DataFrame:
    """构造一个 DataFrame，并尽量保留特征语义，交给 TabICL 自身做预处理。"""
    if isinstance(values, pd.DataFrame):
        df = values.copy()
    else:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)

    if kind == "numeric":
        df = df.apply(pd.to_numeric, errors="coerce")
    elif kind == "categorical":
        df = df.astype("string")
    elif kind == "infer":
        df = df.convert_dtypes()
    else:
        raise ValueError(f"不支持的特征类型: {kind}")

    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


def make_target_array(values: np.ndarray) -> np.ndarray:
    """将标签整理为一维 NumPy 数组，同时保留原始缺失值。"""
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def split_single_file_dataset(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, np.ndarray, np.ndarray]:
    """将单表数据集切分为 train/test；如果条件允许则优先分层抽样。"""
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    if len(y) < 2:
        raise ValueError(f"{dataset_name}: 行数不足，无法进行 train/test 切分")

    stratify = None
    y_series = pd.Series(y)
    class_counts = y_series.value_counts(dropna=False)
    if y_series.nunique(dropna=False) > 1 and not class_counts.empty and int(class_counts.min()) >= 2:
        stratify = y

    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        if stratify is None:
            raise
        logging.warning("%s: 分层 80/20 切分失败，回退到随机切分", dataset_name)
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


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
# 数据集文件查找与加载
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


def load_frame(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        return make_feature_frame(load_array(file_path), kind="infer", prefix=file_path.stem)
    if suffix == '.parquet':
        return make_feature_frame(pd.read_parquet(file_path), kind="infer", prefix=file_path.stem)
    sep = '\t' if suffix == '.tsv' else None
    return make_feature_frame(pd.read_csv(file_path, sep=sep, header=None), kind="infer", prefix=file_path.stem)


def load_table(file_path: Union[Path, Tuple], context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[
    pd.DataFrame, np.ndarray]:
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
        raise ValueError(f"load_table 不支持该元组格式: {file_path}")

    data = load_frame(file_path)
    if data.ndim == 1:
        raise ValueError(f"{file_path} 中的数据为 1D，当前不支持")

    log_target = context or str(file_path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data.iloc[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    heuristic_column = None
    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = col0
        X = data.iloc[:, 1:].copy()
        heuristic_column = 'first'
    else:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1].copy()
        heuristic_column = 'last'

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = make_feature_frame(X, kind="infer", prefix=f"{Path(log_target).stem}_x")
    y = make_target_array(y)

    if heuristic_column:
        logging.info(f"{log_target}: 使用单文件启发式拆分标签 (取 {heuristic_column} 列)")
    return X, y


def load_pair(X_path: Path, y_path: Path, context: str = "", coerce_numeric: bool = False,
              dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    X = load_array(X_path)
    y = load_array(y_path)

    ctx = context or X_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = make_feature_frame(X, kind="infer", prefix=f"{ctx}_x")
    y = make_target_array(y)
    return X, y


def load_split(num_path: Optional[Path], cat_path: Optional[Path], y_path: Path, context: str = "",
               coerce_numeric: bool = False, dataset_id: str | None = None, missing_registry: set[str] | None = None) -> \
Tuple[pd.DataFrame, np.ndarray]:
    features: list[pd.DataFrame] = []
    ctx_base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    if num_path:
        X_num = load_array(num_path)
        log_nan_presence(f"{ctx_base}-num_raw", X_num, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(make_feature_frame(X_num, kind="numeric", prefix=f"{ctx_base}_n"))
    if cat_path:
        X_cat = load_array(cat_path)
        log_nan_presence(f"{ctx_base}-cat_raw", X_cat, dataset_id=dataset_id, missing_registry=missing_registry)
        features.append(make_feature_frame(X_cat, kind="categorical", prefix=f"{ctx_base}_c"))

    if not features:
            raise ValueError("split 数据中未找到数值特征文件或类别特征文件")

    n_samples = features[0].shape[0]
    for idx, feat in enumerate(features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"特征数组 #{idx} 的样本数不一致: {feat.shape[0]} vs {n_samples}")

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    log_nan_presence(f"{ctx_base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)

    y = load_array(y_path)
    log_nan_presence(f"{ctx_base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    y = make_target_array(y)
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
# 核心评测逻辑（Worker）
# ------------------------------

def evaluate_datasets_worker(rank: int, device_id: int, model_path: str, checkpoint_version: str, dataset_dirs: List[Path],
                            verbose: bool = False, skip_regression: bool = True, bins: int = 0,
                            merge_val: bool = False, coerce_numeric: bool = True,
                            n_estimators: int = 32, kv_cache: bool | str = False) -> Tuple[List[Tuple[str, float, float]], set[str]]:
    """
    在指定 GPU 上评测一部分数据集的 worker 函数。
    返回值包含结果列表 (name, acc, duration) 和一个记录含缺失值数据集名称的集合。
    """
    import sys
    from pathlib import Path
    src_path = str(Path(__file__).resolve().parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from tabicl import TabICLClassifier
        from sklearn.utils.multiclass import type_of_target
        from sklearn.preprocessing import KBinsDiscretizer
    except ImportError as e:
        print(f"[Worker {rank}] 导入失败: {e}")
        return [], set()

    # 设置运行设备
    # 使用 cuda:X 的设备表示方式
    device_str = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    
    msg_prefix = f"[GPU {device_id}]"

    print(f"{msg_prefix} 在 {device_str} 上初始化模型，待处理数据集数: {len(dataset_dirs)}")
    
    try:
        clf = TabICLClassifier(verbose=verbose, model_path=model_path, device=device_str, checkpoint_version=checkpoint_version,n_estimators=n_estimators)
    except Exception as e:
        print(f"{msg_prefix} 模型初始化失败: {e}")
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
                X_all, y_all = load_table(train_path, context=f"{d.name}-single", coerce_numeric=coerce_numeric,
                                          dataset_id=d.name, missing_registry=datasets_with_missing)
                X_train, X_test, y_train, y_test = split_single_file_dataset(X_all, y_all, dataset_name=d.name)
                val_path = None
                print(f"{msg_prefix} 数据集：{d.name} (单文件数据按 80/20 自动切分)")

            X_val = y_val = None
            if val_path:
                X_val, y_val = load_table(val_path, context=f"{d.name}-val", coerce_numeric=coerce_numeric,
                                          dataset_id=d.name, missing_registry=datasets_with_missing)
                y_val = np.asarray(y_val)
                if y_val.ndim > 1 and y_val.shape[-1] == 1:
                    y_val = y_val.reshape(-1)
                if merge_val:
                    X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
                    y_train = np.concatenate([y_train, y_val], axis=0)

            # 检查标签类型
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
            clf.fit(X_train, y_train, kv_cache=kv_cache)

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
    p = argparse.ArgumentParser(description='在 TALENT 数据集上并行评测 TabICLClassifier')
    p.add_argument('--model-path', default='tabicl-classifier-v1.1-20250506.ckpt', help='TabICL checkpoint 路径')
    p.add_argument('--data-root', default='data181', help='TALENT 数据目录根路径')
    p.add_argument('--outdir', default='tabiclv1_ensmble8_data181', help='结果输出目录')
    p.add_argument('--max-datasets', type=int, default=None, help='限制最多处理的数据集数量')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--n-estimators', type=int, default=8, help='集成中的估计器数量')
    p.add_argument(
        '--kv-cache',
        type=parse_kv_cache,
        default=False,
        help='clf.fit() 使用的 KV cache 模式: false、true、kv 或 repr',
    )
    p.add_argument('--merge-val', default=True, action='store_true')
    p.add_argument('--num-gpus', type=int, default=8, help='使用的 GPU 数量')
    p.add_argument(
        '--no-coerce-numeric',
        dest='coerce_numeric',
        action='store_false',
        help='兼容旧命令行的废弃参数。特征编码和缺失值处理现在由 TabICLClassifier 内部完成。',
    )
    p.add_argument('--checkpoint-version', default='tabicl-classifier-v1.1-20250506.ckpt', help='使用的 checkpoint 版本')
    p.set_defaults(coerce_numeric=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("特征预处理已交给 TabICLClassifier 内部完成（包括类别编码和缺失值处理）。")
    
    script_start_time = time.time()
    
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 收集所有数据集目录
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if args.max_datasets:
        dirs = dirs[:args.max_datasets]
    
    total_datasets = len(dirs)
    logging.info(f"待处理数据集总数: {total_datasets}，使用 GPU 数量: {args.num_gpus}")

    # 按 GPU 数量将数据集切分成多个分块
    num_gpus = args.num_gpus
    chunk_size = math.ceil(total_datasets / num_gpus)
    chunks = [dirs[i:i + chunk_size] for i in range(0, total_datasets, chunk_size)]
    
    # 当数据集数量少于 GPU 数量时，移除空分块
    chunks = [c for c in chunks if len(c) > 0]
    
    logging.info(f"已切分为 {len(chunks)} 个分块，最大分块大小: {chunk_size}")

    # 为 starmap 组织参数
    # 参数顺序: (rank, device_id, model_path, dataset_dirs, verbose, skip_regression, bins, merge_val, coerce_numeric)
    tasks = []
    for i, chunk in enumerate(chunks):
        # 计算该 worker 对应的设备编号。
        # 这里默认每个分块启动一个进程，并将 rank i 映射到 gpu i。
        device_id = i % num_gpus
        tasks.append((
            i, 
            device_id,
            args.model_path,
            args.checkpoint_version,
            chunk,
            args.verbose,
            True, # 是否跳过回归任务
            0,    # 连续标签分桶数
            args.merge_val,
            args.coerce_numeric,
            args.n_estimators,
            args.kv_cache,
        ))

    # 使用 multiprocessing 并行运行
    # 对 PyTorch/CUDA 来说，spawn 上下文通常更安全
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=len(tasks)) as pool:
        results_list = pool.starmap(evaluate_datasets_worker, tasks)

    # 汇总结果
    all_results = []
    all_missing_registry = set()

    for res, missing in results_list:
        all_results.extend(res)
        all_missing_registry.update(missing)
    
    # 按数据集名称排序，保证结果输出稳定
    all_results.sort(key=lambda x: x[0])

    if all_results:
        # 计算汇总统计指标
        total_time = sum(duration for _, _, duration in all_results)
        avg_time = total_time / len(all_results)
        avg_acc = sum(acc for _, acc, _ in all_results) / len(all_results)
        
        script_duration = time.time() - script_start_time

        csv_path = outdir / 'talent_detailed.csv'
        write_header = not csv_path.exists()
        with open(csv_path, 'a') as f:
            if write_header:
                f.write('dataset,accuracy,time_s\n')
            for name, acc, duration in all_results:
                f.write(f"{name},{acc:.6f},{duration:.3f}\n")
            f.write(f"Average,{avg_acc:.6f},{avg_time:.3f}\n")

        # 写入文字版摘要
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

        logging.info(f"评测完成，结果已保存到 {outdir}")
        logging.info(f"平均准确率: {avg_acc:.4f}，平均耗时: {avg_time:.2f}s，脚本总耗时: {script_duration:.2f}s")
    else:
        logging.info("没有获得成功的评测结果。")

if __name__ == '__main__':
    main()
