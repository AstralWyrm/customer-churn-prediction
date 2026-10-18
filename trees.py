from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NUMERIC_FEATURES = {
    "tenure_months",
    "monthly_charges",
    "avg_monthly_gb",
    "support_tickets_90d",
    "late_payments_6m",
    "outage_hours_90d",
    "num_services",
}

TARGET = "churned"
ID_COLUMN = "customer_id"


def load_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in NUMERIC_FEATURES:
                    parsed[key] = float(value)
                elif key == TARGET:
                    parsed[key] = int(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def f1_score_positive(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_positive = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    false_positive = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    false_negative = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0

    
def class_counts(rows: Sequence[dict]) -> Counter:
    return Counter(row[TARGET] for row in rows)


def majority_label(rows: Sequence[dict]) -> int:
    counts = class_counts(rows)
    return max(sorted(counts), key=lambda label: counts[label])


def entropy(rows: Sequence[dict]) -> float:
    counts = class_counts(rows)
    total = len(rows)
    result = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0.0:
            result -= probability * math.log2(probability)
    return result


def entropy_from_counts(counts: Counter, total: int) -> float:
    result = 0.0
    for count in counts.values():
        if count > 0:
            probability = count / total
            result -= probability * math.log2(probability)
    return result


@dataclass
class Split:
    feature: str
    score: float
    kind: str
    threshold: Optional[float] = None
    partitions: Optional[dict] = None


@dataclass
class Node:
    prediction: int
    feature: Optional[str] = None
    threshold: Optional[float] = None
    children: Optional[dict] = None
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    def __init__(
        self,
        criterion: str = "information_gain",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        feature_subsample_size: Optional[int] = None,
        random_state: int = 3411,
        use_numeric: bool = True,
        min_gain: float = 0.0,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsample_size = feature_subsample_size
        self.random = random.Random(random_state)
        self.root: Optional[Node] = None
        self.features: List[str] = []
        self.use_numeric = use_numeric
        if min_gain < 0 or min_samples_split < 2:
            raise ValueError("min_gain must be non-negative and min_samples_split at least 2")
        self.min_gain = min_gain

    def fit(self, rows: Sequence[dict]) -> None:
        if not rows:
            raise ValueError("Cannot train a decision tree on an empty dataset")
        self.features = [
            key for key in rows[0]
            if key not in {TARGET, ID_COLUMN}
            and (self.use_numeric or key not in NUMERIC_FEATURES)
        ]
        self.root = self._build_tree(list(rows), depth=0)

    def predict_one(self, row: dict) -> int:
        assert self.root is not None
        node = self.root
        while not node.is_leaf:
            assert node.feature is not None
            value = row[node.feature]
            if node.threshold is not None:
                branch = "<=" if float(value) <= node.threshold else ">"
                node = node.children[branch]
            else:
                if value not in node.children:
                    return node.prediction
                node = node.children[value]
        return node.prediction

    def predict(self, rows: Sequence[dict]) -> List[int]:
        return [self.predict_one(row) for row in rows]

    def tree_depth(self) -> int:
        return self._tree_depth(self.root)

    def count_nodes(self) -> Tuple[int, int]:
        return self._count_nodes(self.root)

    def print_tree(self, max_depth: int = 2) -> None:
        self._print_tree(self.root, max_depth=max_depth)

    def _build_tree(self, rows: List[dict], depth: int) -> Node:
        prediction = majority_label(rows)
        if self._should_stop(rows, depth):
            return Node(prediction=prediction, depth=depth)

        split = self._best_split(rows)
        if split is None or split.score <= 0.0 or split.score < self.min_gain:
            return Node(prediction=prediction, depth=depth)

        if split.kind == "numeric":
            assert split.threshold is not None
            left_rows = [row for row in rows if float(row[split.feature]) <= split.threshold]
            right_rows = [row for row in rows if float(row[split.feature]) > split.threshold]
            children = {
                "<=": self._build_tree(left_rows, depth + 1),
                ">": self._build_tree(right_rows, depth + 1),
            }
            return Node(
                prediction=prediction,
                feature=split.feature,
                threshold=split.threshold,
                children=children,
                depth=depth,
            )

        assert split.partitions is not None
        children = {}
        for value, subset in split.partitions.items():
            children[value] = self._build_tree(subset, depth + 1)
        return Node(
            prediction=prediction,
            feature=split.feature,
            children=children,
            depth=depth,
        )

    def _should_stop(self, rows: Sequence[dict], depth: int) -> bool:
        if len(rows) < self.min_samples_split:
            return True
        if len(class_counts(rows)) == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _candidate_features(self) -> List[str]:
        if self.feature_subsample_size is None or self.feature_subsample_size >= len(self.features):
            return list(self.features)
        return self.random.sample(self.features, self.feature_subsample_size)

    def _best_split(self, rows: Sequence[dict]) -> Optional[Split]:
        best: Optional[Split] = None
        for feature in self._candidate_features():
            if feature in NUMERIC_FEATURES:
                # numerical features for Section 2.2
                if not self.use_numeric:
                    continue
                candidate = self.best_numeric_split(rows, feature)
            else:
                candidate = self.best_categorical_split(rows, feature)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def score_split(self, rows: Sequence[dict], partitions: Sequence[Sequence[dict]]) -> float:
        if self.criterion == "information_gain":
            return self.information_gain(rows, partitions)
        raise ValueError(f"Unknown criterion: {self.criterion}")

    def information_gain(self, rows: Sequence[dict], partitions: Sequence[Sequence[dict]]) -> float:
        parent_entropy = entropy(rows)
        total_samples = len(rows)

        weighted_entropy = 0.0

        # Weight each subset's entropy by its proportion of samples
        for subset in partitions:
            subset_weight = len(subset) / total_samples
            subset_entropy = entropy(subset)

            weighted_entropy = (
                weighted_entropy
                + subset_weight * subset_entropy
            )

        information_gain = parent_entropy - weighted_entropy

        return information_gain



    def best_categorical_split(self, rows: Sequence[dict], feature: str) -> Optional[Split]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[feature]].append(row)
        partitions = list(grouped.values())
        if len(partitions) <= 1:
            return None
        score = self.score_split(rows, partitions)
        return Split(feature=feature, score=score, kind="categorical", partitions=dict(grouped))

    def best_numeric_split(self, rows: Sequence[dict], feature: str) -> Optional[Split]:
        # Evaluate every distinct-value midpoint with cumulative label counts.
        # This avoids rescanning all rows for every candidate threshold.
        if len(rows) < 2:
            return None
        ordered = sorted(rows, key=lambda row: row[feature])
        total = len(ordered)
        right_counts = class_counts(ordered)
        left_counts = Counter()
        parent_entropy = entropy(rows)
        best = None

        for index in range(total - 1):
            row = ordered[index]
            label = row[TARGET]
            left_counts[label] += 1
            right_counts[label] -= 1
            current_value = float(row[feature])
            next_value = float(ordered[index + 1][feature])
            if current_value == next_value:
                continue

            left_size = index + 1
            right_size = total - left_size
            left_entropy = entropy_from_counts(left_counts, left_size)
            right_entropy = entropy_from_counts(right_counts, right_size)
            score = parent_entropy - (
                (left_size / total) * left_entropy
                + (right_size / total) * right_entropy
            )
            threshold = (current_value + next_value) / 2
            if best is None or score > best.score:
                best = Split(feature, score, "numeric", threshold=threshold)
        return best

    def _tree_depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(child) for child in node.children.values())

    def _count_nodes(self, node: Optional[Node]) -> Tuple[int, int]:
        if node is None:
            return (0, 0)
        if node.is_leaf:
            return (0, 1)
        internal = 1
        leaves = 0
        for child in node.children.values():
            child_internal, child_leaves = self._count_nodes(child)
            internal += child_internal
            leaves += child_leaves
        return (internal, leaves)

    def _print_tree(self, node: Optional[Node], prefix: str = "", max_depth: int = 2) -> None:
        if node is None:
            return
        if node.is_leaf:
            print(f"{prefix}Predict {node.prediction}")
            return
        if node.depth >= max_depth:
            test = node.feature
            if node.threshold is not None:
                test += f" <= {node.threshold:.3f}"
            print(f"{prefix}{test} [subtree hidden; majority={node.prediction}]")
            return
        if node.threshold is not None:
            print(f"{prefix}{node.feature} <= {node.threshold:.3f}")
            self._print_tree(node.children["<="], prefix + "  [T] ", max_depth)
            self._print_tree(node.children[">"], prefix + "  [F] ", max_depth)
            return
        print(f"{prefix}{node.feature}")
        for value, child in sorted(node.children.items()):
            self._print_tree(child, prefix + f"  [{value}] ", max_depth)


class BaggingEnsemble:
    def __init__(
        self,
        n_estimators: int = 10,
        criterion: str = "information_gain",
        max_depth: Optional[int] = None,
        feature_subsample_size: Optional[int] = None,
        random_state: int = 3411,
        min_samples_split: int = 2,
        min_gain: float = 0.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        self.trees: List[DecisionTree] = []

    def fit(self, rows: Sequence[dict]) -> None:
        if not rows:
            raise ValueError("Cannot train an ensemble on an empty dataset")
        self.trees = []
        # Reset the random generator for reproducible training
        self.random = random.Random(self.random_state)
        for index in range(self.n_estimators):
            sample = [self.random.choice(rows) for _ in range(len(rows))]
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                feature_subsample_size=self.feature_subsample_size,
                random_state=self.random.randint(0, 1000000),
                use_numeric=True,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
            )
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, rows: Sequence[dict]) -> List[int]:
        if not self.trees:
            raise ValueError("Fit the ensemble before prediction")
        all_predictions = [tree.predict(rows) for tree in self.trees]
        final_predictions = []
        for row_index in range(len(rows)):
            votes = Counter(predictions[row_index] for predictions in all_predictions)
            final_predictions.append(max(sorted(votes), key=lambda label: votes[label]))
        return final_predictions



class ValidationWeightedEnsemble(BaggingEnsemble):
    """Bootstrap entropy trees whose votes are weighted by validation accuracy.

    Validation labels determine weights, so validation performance is a tuning
    result rather than an independent estimate. Test labels never set weights.
    """

    def fit(self, rows: Sequence[dict], validation_rows: Sequence[dict]) -> None:
        if not validation_rows:
            raise ValueError("Validation rows are required to estimate vote weights")
        super().fit(rows)
        labels = [row[TARGET] for row in validation_rows]
        self.weights = [
            accuracy_score(labels, tree.predict(validation_rows))
            for tree in self.trees
        ]
        if not any(self.weights):
            self.weights = [1.0] * len(self.trees)

    def predict(self, rows: Sequence[dict]) -> List[int]:
        if not self.trees or not hasattr(self, "weights"):
            raise ValueError("Fit the weighted ensemble before prediction")
        all_predictions = [tree.predict(rows) for tree in self.trees]
        predictions = []
        for index in range(len(rows)):
            totals = {0: 0.0, 1: 0.0}
            for weight, votes in zip(self.weights, all_predictions):
                totals[votes[index]] += weight
            predictions.append(1 if totals[1] > totals[0] else 0)
        return predictions


def evaluate_model(
    model,
    train_rows: Sequence[dict],
    valid_rows: Sequence[dict],
    test_rows: Sequence[dict],
) -> None:
    for split_name, rows in [
        ("train", train_rows),
        ("valid", valid_rows),
        ("test", test_rows),
    ]:
        y_true = [row[TARGET] for row in rows]
        y_pred = model.predict(rows)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score_positive(y_true, y_pred)

        print(f"{split_name} accuracy: {accuracy:.4f}")
        print(f"{split_name} F1 score: {f1:.4f}")


def run_starter_summary(base_dir: Path) -> None:
    train_rows = load_csv(base_dir / "data" / "churn_train.csv")
    valid_rows = load_csv(base_dir / "data" / "churn_valid.csv")
    test_rows = load_csv(base_dir / "data" / "churn_test.csv")

    print(f"Training rows: {len(train_rows)}")
    print(f"Validation rows: {len(valid_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Features: {len(train_rows[0]) - 2}")

    print(f"Training class counts: {class_counts(train_rows)}")
    print(f"Validation class counts: {class_counts(valid_rows)}")
    print(f"Test class counts: {class_counts(test_rows)}")

    baseline = majority_label(train_rows)
    print(f"Majority class from training data: {baseline}")

    for split_name, rows in [
        ("train", train_rows),
        ("validation", valid_rows),
        ("test", test_rows),
    ]:
        labels = [row[TARGET] for row in rows]
        predictions = [baseline] * len(rows)
        baseline_accuracy = accuracy_score(labels, predictions)
        baseline_f1 = f1_score_positive(labels, predictions)

        print(
            f"Majority baseline {split_name}: "
            f"accuracy={baseline_accuracy:.4f}, "
            f"F1={baseline_f1:.4f}"
        )

    print("\nImplemented: entropy, information gain, categorical and numerical splits,")
    print("bagging, random forests, validation-weighted voting and early stopping.")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_starter_summary(here)
