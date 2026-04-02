import json

import numpy as np
import pandas as pd

from tabicl.experiments import run_oracle_pseudo_label_experiment
from tabicl.experiments import write_oracle_experiment_outputs


class SequentialEstimatorFactory:
    def __init__(self, stage_probabilities):
        self.stage_probabilities = stage_probabilities
        self.created = 0

    def __call__(self):
        stage_id = min(self.created, len(self.stage_probabilities) - 1)
        self.created += 1
        return FakeClassifier(self.stage_probabilities[stage_id])


class FakeClassifier:
    def __init__(self, probability_map):
        self.probability_map = probability_map
        self.fit_calls = 0

    def fit(self, X, y, kv_cache=False):
        self.fit_calls += 1
        return self

    def _ids(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].to_numpy()
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):
        ids = self._ids(X)
        return np.asarray([self.probability_map[int(sample_id)] for sample_id in ids], dtype=float)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


def test_oracle_selection_and_remaining_pool_shrinks():
    X_labeled = pd.DataFrame({"sample_id": [10, 11], "f1": [0.0, 1.0]})
    y_labeled = np.array([0, 1])
    X_target = pd.DataFrame({"sample_id": [0, 1, 2, 3], "f1": [0.1, 0.2, 0.3, 0.4]})
    y_target = np.array([0, 1, 1, 0])

    factory = SequentialEstimatorFactory(
        [
            {
                0: [0.90, 0.10],
                1: [0.05, 0.95],
                2: [0.80, 0.20],
                3: [0.10, 0.90],
            },
            {
                0: [0.90, 0.10],
                1: [0.05, 0.95],
                2: [0.10, 0.90],
                3: [0.85, 0.15],
            },
        ]
    )

    result = run_oracle_pseudo_label_experiment(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        X_target=X_target,
        y_target=y_target,
        estimator_factory=factory,
        dataset_name="toy",
        max_rounds=3,
        min_added=1,
        save_full_predictions=True,
    )

    assert result.summary["total_selected"] == 4
    assert result.summary["total_rounds"] == 2
    assert result.summary["final_accuracy"] == 1.0
    assert result.summary["stop_reason"] == "empty_pool"

    assert result.rounds[1]["n_selected"] == 2
    assert result.rounds[1]["n_remaining_after"] == 2
    assert result.rounds[2]["n_selected"] == 2
    assert result.rounds[2]["n_remaining_after"] == 0
    assert result.selected_by_round[1]["target_index"].tolist() == [0, 1]
    assert result.selected_by_round[2]["target_index"].tolist() == [2, 3]
    assert sorted(result.full_predictions_by_round) == [0, 1, 2]


def test_min_added_stop_condition_is_recorded():
    X_labeled = pd.DataFrame({"sample_id": [20, 21], "f1": [1.0, 2.0]})
    y_labeled = np.array([0, 1])
    X_target = pd.DataFrame({"sample_id": [0, 1, 2], "f1": [0.1, 0.2, 0.3]})
    y_target = np.array([0, 1, 1])

    factory = SequentialEstimatorFactory(
        [
            {
                0: [0.95, 0.05],
                1: [0.70, 0.30],
                2: [0.80, 0.20],
            },
            {
                0: [0.95, 0.05],
                1: [0.65, 0.35],
                2: [0.75, 0.25],
            },
        ]
    )

    result = run_oracle_pseudo_label_experiment(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        X_target=X_target,
        y_target=y_target,
        estimator_factory=factory,
        dataset_name="toy-min-added",
        max_rounds=5,
        min_added=2,
    )

    assert result.summary["total_rounds"] == 1
    assert result.summary["total_selected"] == 1
    assert result.summary["stop_reason"] == "min_added"
    assert result.rounds[1]["n_selected"] == 1
    assert result.rounds[1]["n_labeled_after"] == 3
    assert result.rounds[1]["stop_reason"] == "min_added"


def test_output_files_contain_required_fields(tmp_path):
    X_labeled = pd.DataFrame({"sample_id": [30, 31], "f1": [1.0, 2.0]})
    y_labeled = np.array([0, 1])
    X_target = pd.DataFrame({"sample_id": [0, 1], "f1": [0.1, 0.2]})
    y_target = np.array([0, 1])

    factory = SequentialEstimatorFactory(
        [
            {
                0: [0.90, 0.10],
                1: [0.05, 0.95],
            }
        ]
    )

    result = run_oracle_pseudo_label_experiment(
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        X_target=X_target,
        y_target=y_target,
        estimator_factory=factory,
        dataset_name="toy-output",
        max_rounds=1,
        min_added=1,
        config={"seed": 42},
        save_full_predictions=True,
    )
    write_oracle_experiment_outputs(result, tmp_path, save_full_predictions=True)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert {
        "dataset_name",
        "baseline_accuracy",
        "final_accuracy",
        "baseline_macro_f1",
        "final_macro_f1",
        "total_selected",
        "total_rounds",
        "config",
    }.issubset(summary.keys())

    rounds = pd.read_csv(tmp_path / "rounds.csv")
    assert {
        "round_id",
        "n_labeled_before",
        "n_selected",
        "n_labeled_after",
        "n_remaining_after",
        "accuracy_before_round_on_remaining",
        "accuracy_after_refit_on_remaining",
        "accuracy_on_full_target",
        "macro_f1_on_full_target",
        "mean_confidence_of_selected",
        "mean_confidence_of_remaining",
    }.issubset(rounds.columns)

    assert (tmp_path / "selected_indices_round_1.csv").exists()
    assert (tmp_path / "full_predictions_round_0.csv").exists()
    assert (tmp_path / "full_predictions_round_1.csv").exists()
