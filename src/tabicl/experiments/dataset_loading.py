from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target


CLASSIFICATION_TASKS = {"binclass", "multiclass", "unknown"}


@dataclass
class ClassificationDataset:
    dataset_name: str
    dataset_dir: Path
    X_labeled: pd.DataFrame
    y_labeled: np.ndarray
    X_target: pd.DataFrame
    y_target: np.ndarray
    info: Optional[dict]
    used_single_file_split: bool
    has_missing_values: bool


def parse_kv_cache(value: str) -> bool | str:
    """Parse CLI kv_cache values into the types expected by TabICLClassifier.fit()."""
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
    """Construct a DataFrame while preserving enough feature semantics for TabICL preprocessing."""
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
    """Convert a target array to a flat NumPy array while preserving missing values."""
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
    """Split a single-table dataset into train/test, preferring stratification when possible."""
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


def count_missing(values: np.ndarray | pd.DataFrame | pd.Series) -> int:
    if values is None:
        return 0

    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())

    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def log_nan_presence(
    context: str,
    values: np.ndarray | pd.DataFrame | pd.Series,
    *,
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> None:
    missing = count_missing(values)
    if missing and dataset_id and missing_registry is not None:
        missing_registry.add(dataset_id)


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("读取 %s 失败: %s", info_path, exc)
        return None


def find_data_files(dataset_dir: Path):
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower_names = {p.name.lower(): p for p in files}

    def find_by_suffix(key: str):
        for name, path in lower_names.items():
            if name.endswith(key):
                return path
        return None

    n_train = find_by_suffix("n_train.npy")
    c_train = find_by_suffix("c_train.npy")
    y_train = find_by_suffix("y_train.npy")
    n_val = find_by_suffix("n_val.npy")
    c_val = find_by_suffix("c_val.npy")
    y_val = find_by_suffix("y_val.npy")
    n_test = find_by_suffix("n_test.npy")
    c_test = find_by_suffix("c_test.npy")
    y_test = find_by_suffix("y_test.npy")

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    table_candidates = [p for p in files if p.suffix.lower() in {".npy", ".npz", ".csv", ".tsv", ".parquet"}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None

    return None, None, None


def load_array(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == ".parquet":
        return pd.read_parquet(file_path).values
    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def load_frame(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return make_feature_frame(load_array(file_path), kind="infer", prefix=file_path.stem)
    if suffix == ".parquet":
        return make_feature_frame(pd.read_parquet(file_path), kind="infer", prefix=file_path.stem)
    sep = "\t" if suffix == ".tsv" else None
    return make_feature_frame(pd.read_csv(file_path, sep=sep, header=None), kind="infer", prefix=file_path.stem)


def load_table(
    file_path: Union[Path, Tuple],
    *,
    context: str = "",
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            x_path, y_path = Path(file_path[0]), Path(file_path[1])
            return load_pair(x_path, y_path, context=context, dataset_id=dataset_id, missing_registry=missing_registry)
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(
                Path(num_path) if num_path else None,
                Path(cat_path) if cat_path else None,
                Path(y_path),
                context=context,
                dataset_id=dataset_id,
                missing_registry=missing_registry,
            )
        raise ValueError(f"load_table 不支持该元组格式: {file_path}")

    data = load_frame(file_path)
    if data.ndim == 1:
        raise ValueError(f"{file_path} 中的数据为 1D，当前不支持")

    log_target = context or str(file_path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    first_column = data.iloc[:, 0]
    try:
        first_uniques = np.unique(first_column)
    except Exception:
        first_uniques = np.array([])

    heuristic_column = None
    if 0 < first_uniques.size < max(2, data.shape[0] // 2):
        y = first_column
        X = data.iloc[:, 1:].copy()
        heuristic_column = "first"
    else:
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1].copy()
        heuristic_column = "last"

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = make_feature_frame(X, kind="infer", prefix=f"{Path(log_target).stem}_x")
    y = make_target_array(y)

    if heuristic_column:
        logging.info("%s: 使用单文件启发式拆分标签 (取 %s 列)", log_target, heuristic_column)
    return X, y


def load_pair(
    X_path: Path,
    y_path: Path,
    *,
    context: str = "",
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = load_array(X_path)
    y = load_array(y_path)

    ctx = context or X_path.stem
    log_nan_presence(f"{ctx}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{ctx}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = make_feature_frame(X, kind="infer", prefix=f"{ctx}_x")
    y = make_target_array(y)
    return X, y


def load_split(
    num_path: Optional[Path],
    cat_path: Optional[Path],
    y_path: Path,
    *,
    context: str = "",
    dataset_id: str | None = None,
    missing_registry: set[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
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


def load_oracle_classification_dataset(
    dataset_dir: Path,
    *,
    merge_val: bool = True,
    random_state: int = 42,
) -> ClassificationDataset:
    dataset_dir = dataset_dir.resolve()
    info = load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type == "regression":
        raise ValueError(f"{dataset_dir.name}: task_type=regression，当前脚本仅支持分类任务")
    if task_type and task_type not in CLASSIFICATION_TASKS:
        raise ValueError(f"{dataset_dir.name}: 不支持的 task_type={task_type}")

    missing_registry: set[str] = set()
    train_path, val_path, test_path = find_data_files(dataset_dir)
    if train_path is None and test_path is None:
        raise ValueError(f"{dataset_dir.name}: 未找到可识别的数据文件")

    used_single_file_split = False
    if train_path and test_path:
        X_labeled, y_labeled = load_table(
            train_path,
            context=f"{dataset_dir.name}-train",
            dataset_id=dataset_dir.name,
            missing_registry=missing_registry,
        )
        X_target, y_target = load_table(
            test_path,
            context=f"{dataset_dir.name}-test",
            dataset_id=dataset_dir.name,
            missing_registry=missing_registry,
        )
    else:
        X_all, y_all = load_table(
            train_path,
            context=f"{dataset_dir.name}-single",
            dataset_id=dataset_dir.name,
            missing_registry=missing_registry,
        )
        X_labeled, X_target, y_labeled, y_target = split_single_file_dataset(
            X_all,
            y_all,
            dataset_name=dataset_dir.name,
            random_state=random_state,
        )
        val_path = None
        used_single_file_split = True

    if val_path and merge_val:
        X_val, y_val = load_table(
            val_path,
            context=f"{dataset_dir.name}-val",
            dataset_id=dataset_dir.name,
            missing_registry=missing_registry,
        )
        X_labeled = pd.concat([X_labeled, X_val], axis=0, ignore_index=True)
        y_labeled = np.concatenate([y_labeled, y_val], axis=0)

    target_type = type_of_target(y_labeled)
    if target_type not in {"binary", "multiclass"}:
        raise ValueError(
            f"{dataset_dir.name}: 仅支持分类任务，检测到标签类型为 {target_type!r}。"
        )

    return ClassificationDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        X_labeled=X_labeled.reset_index(drop=True),
        y_labeled=np.asarray(y_labeled),
        X_target=X_target.reset_index(drop=True),
        y_target=np.asarray(y_target),
        info=info,
        used_single_file_split=used_single_file_split,
        has_missing_values=dataset_dir.name in missing_registry,
    )
