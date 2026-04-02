from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class ClassifierProtocol(Protocol):
    def fit(self, X: Any, y: Any, kv_cache: bool | str = False): ...
    def predict(self, X: Any) -> np.ndarray: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...


@dataclass
class OracleExperimentResult:
    summary: dict[str, Any]
    rounds: list[dict[str, Any]]
    selected_by_round: dict[int, pd.DataFrame]
    full_predictions_by_round: dict[int, pd.DataFrame]


def _to_row_indices(indexer: np.ndarray | list[int] | list[bool]) -> np.ndarray:
    arr = np.asarray(indexer)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    return arr.astype(int, copy=False)


def _slice_rows(X: pd.DataFrame | np.ndarray, indexer: np.ndarray | list[int] | list[bool]):
    row_indices = _to_row_indices(indexer)
    if isinstance(X, pd.DataFrame):
        return X.iloc[row_indices].reset_index(drop=True)
    return np.asarray(X)[row_indices]


def _concat_rows(left: pd.DataFrame | np.ndarray, right: pd.DataFrame | np.ndarray):
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        return pd.concat([left, right], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=0)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def _fit_estimator(estimator: ClassifierProtocol, X, y, kv_cache: bool | str) -> ClassifierProtocol:
    try:
        estimator.fit(X, y, kv_cache=kv_cache)
    except TypeError as exc:
        if "kv_cache" not in str(exc):
            raise
        estimator.fit(X, y)
    return estimator


def _predict_with_confidence(estimator: ClassifierProtocol, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.ndim != 2:
        raise ValueError("predict_proba() 必须返回二维数组")
    predictions = np.asarray(estimator.predict(X))
    confidence = probabilities.max(axis=1)
    return predictions, probabilities, confidence


def _build_prediction_frame(
    *,
    round_id: int,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round_id": round_id,
            "target_index": indices,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
        }
    )


def _build_selected_frame(
    *,
    round_id: int,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round_id": round_id,
            "target_index": indices,
            "y_true": y_true,
            "y_hat": y_hat,
            "confidence": confidence,
        }
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run_oracle_pseudo_label_experiment(
    *,
    X_labeled: pd.DataFrame | np.ndarray,
    y_labeled: np.ndarray,
    X_target: pd.DataFrame | np.ndarray,
    y_target: np.ndarray,
    estimator_factory: Callable[[], ClassifierProtocol],
    dataset_name: str,
    config: dict[str, Any] | None = None,
    kv_cache: bool | str = False,
    max_rounds: int = 1,
    min_added: int = 1,
    save_full_predictions: bool = False,
) -> OracleExperimentResult:
    if max_rounds < 0:
        raise ValueError("max_rounds 不能小于 0")
    if min_added < 1:
        raise ValueError("min_added 不能小于 1")

    labeled_X = X_labeled.reset_index(drop=True) if isinstance(X_labeled, pd.DataFrame) else np.asarray(X_labeled)
    target_X = X_target.reset_index(drop=True) if isinstance(X_target, pd.DataFrame) else np.asarray(X_target)
    labeled_y = np.asarray(y_labeled)
    target_y = np.asarray(y_target)

    rounds: list[dict[str, Any]] = []
    selected_by_round: dict[int, pd.DataFrame] = {}
    full_predictions_by_round: dict[int, pd.DataFrame] = {}

    current_estimator = _fit_estimator(estimator_factory(), labeled_X, labeled_y, kv_cache)
    remaining_X = target_X
    remaining_y = target_y.copy()
    remaining_indices = np.arange(len(target_y))

    full_pred, _, full_conf = _predict_with_confidence(current_estimator, target_X)
    full_accuracy, full_macro_f1 = _classification_metrics(target_y, full_pred)
    if save_full_predictions:
        full_predictions_by_round[0] = _build_prediction_frame(
            round_id=0,
            indices=np.arange(len(target_y)),
            y_true=target_y,
            y_pred=full_pred,
            confidence=full_conf,
        )

    rounds.append(
        {
            "round_id": 0,
            "n_labeled_before": int(len(labeled_y)),
            "n_selected": 0,
            "n_labeled_after": int(len(labeled_y)),
            "n_remaining_after": int(len(remaining_y)),
            "accuracy_before_round_on_remaining": full_accuracy,
            "accuracy_after_refit_on_remaining": full_accuracy,
            "accuracy_on_full_target": full_accuracy,
            "macro_f1_on_full_target": full_macro_f1,
            "mean_confidence_of_selected": math.nan,
            "mean_confidence_of_remaining": _safe_mean(full_conf),
            "stop_reason": "",
        }
    )

    stop_reason = "max_rounds" if max_rounds == 0 else ""

    for round_id in range(1, max_rounds + 1):
        if len(remaining_y) == 0:
            stop_reason = "empty_pool"
            break

        before_pred, _, before_conf = _predict_with_confidence(current_estimator, remaining_X)
        accuracy_before = float(accuracy_score(remaining_y, before_pred))
        selected_mask = before_pred == remaining_y
        selected_indices = remaining_indices[selected_mask]
        n_selected = int(selected_mask.sum())

        selected_by_round[round_id] = _build_selected_frame(
            round_id=round_id,
            indices=selected_indices,
            y_true=remaining_y[selected_mask],
            y_hat=before_pred[selected_mask],
            confidence=before_conf[selected_mask],
        )

        if n_selected > 0:
            # Oracle only decides which target samples may be absorbed; training still uses y_hat.
            labeled_X = _concat_rows(labeled_X, _slice_rows(remaining_X, selected_mask))
            labeled_y = np.concatenate([labeled_y, before_pred[selected_mask]], axis=0)
            remaining_X = _slice_rows(remaining_X, ~selected_mask)
            remaining_y = remaining_y[~selected_mask]
            remaining_indices = remaining_indices[~selected_mask]

            # Re-fitting here rebuilds the TabICL context; it does not optimize model weights.
            current_estimator = _fit_estimator(estimator_factory(), labeled_X, labeled_y, kv_cache)
        else:
            stop_reason = "no_selection"

        if len(remaining_y) > 0:
            after_pred, _, after_conf = _predict_with_confidence(current_estimator, remaining_X)
            accuracy_after = float(accuracy_score(remaining_y, after_pred))
            mean_conf_remaining = _safe_mean(after_conf)
        else:
            accuracy_after = math.nan
            mean_conf_remaining = math.nan

        full_pred, _, full_conf = _predict_with_confidence(current_estimator, target_X)
        full_accuracy, full_macro_f1 = _classification_metrics(target_y, full_pred)
        if save_full_predictions:
            full_predictions_by_round[round_id] = _build_prediction_frame(
                round_id=round_id,
                indices=np.arange(len(target_y)),
                y_true=target_y,
                y_pred=full_pred,
                confidence=full_conf,
            )

        round_record = {
            "round_id": round_id,
            "n_labeled_before": int(len(labeled_y) - n_selected),
            "n_selected": n_selected,
            "n_labeled_after": int(len(labeled_y)),
            "n_remaining_after": int(len(remaining_y)),
            "accuracy_before_round_on_remaining": accuracy_before,
            "accuracy_after_refit_on_remaining": accuracy_after,
            "accuracy_on_full_target": full_accuracy,
            "macro_f1_on_full_target": full_macro_f1,
            "mean_confidence_of_selected": _safe_mean(before_conf[selected_mask]),
            "mean_confidence_of_remaining": mean_conf_remaining,
            "stop_reason": "",
        }

        if len(remaining_y) == 0:
            stop_reason = "empty_pool"
            round_record["stop_reason"] = stop_reason
            rounds.append(round_record)
            break

        if n_selected == 0:
            stop_reason = "no_selection"
            round_record["stop_reason"] = stop_reason
            rounds.append(round_record)
            break

        if n_selected < min_added:
            stop_reason = "min_added"
            round_record["stop_reason"] = stop_reason
            rounds.append(round_record)
            break

        if round_id == max_rounds:
            stop_reason = "max_rounds"
            round_record["stop_reason"] = stop_reason

        rounds.append(round_record)

    if not stop_reason:
        stop_reason = "max_rounds" if max_rounds > 0 else "max_rounds"

    summary = {
        "dataset_name": dataset_name,
        "baseline_accuracy": rounds[0]["accuracy_on_full_target"],
        "final_accuracy": rounds[-1]["accuracy_on_full_target"],
        "baseline_macro_f1": rounds[0]["macro_f1_on_full_target"],
        "final_macro_f1": rounds[-1]["macro_f1_on_full_target"],
        "total_selected": int(sum(round_info["n_selected"] for round_info in rounds[1:])),
        "total_rounds": max(0, len(rounds) - 1),
        "stop_reason": stop_reason,
        "config": config or {},
    }

    return OracleExperimentResult(
        summary=summary,
        rounds=rounds,
        selected_by_round=selected_by_round,
        full_predictions_by_round=full_predictions_by_round,
    )


def write_oracle_experiment_outputs(
    result: OracleExperimentResult,
    output_dir: str | Path,
    *,
    save_full_predictions: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rounds_df = pd.DataFrame(result.rounds)
    rounds_df.to_csv(output_path / "rounds.csv", index=False)

    for round_id, selected_df in result.selected_by_round.items():
        selected_df.to_csv(output_path / f"selected_indices_round_{round_id}.csv", index=False)

    if save_full_predictions:
        for round_id, prediction_df in result.full_predictions_by_round.items():
            prediction_df.to_csv(output_path / f"full_predictions_round_{round_id}.csv", index=False)

    with (output_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable(result.summary), f, ensure_ascii=False, indent=2)
