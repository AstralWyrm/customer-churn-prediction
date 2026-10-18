"""Experiment orchestration and reporting for the entropy-based churn models."""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

from trees import (ID_COLUMN, TARGET, BaggingEnsemble, DecisionTree,
                   ValidationWeightedEnsemble, accuracy_score, class_counts,
                   f1_score_positive, load_csv, majority_label)

SEED = 3411
BASE_DIR = Path(__file__).resolve().parent


def load_datasets(data_dir=None):
    data_dir = Path(data_dir) if data_dir else BASE_DIR / "data"
    return {name: load_csv(data_dir / filename) for name, filename in (
        ("train", "churn_train.csv"), ("validation", "churn_valid.csv"),
        ("test", "churn_test.csv"))}


def baseline_metrics(datasets):
    label = majority_label(datasets["train"])
    metrics = {}
    for name, rows in datasets.items():
        actual = [row[TARGET] for row in rows]
        predictions = [label] * len(rows)
        metrics[name] = {"accuracy": accuracy_score(actual, predictions),
                         "f1": f1_score_positive(actual, predictions)}
    return metrics


def print_summary(datasets):
    print("Random seed:", SEED)
    print("Majority class from training:", majority_label(datasets["train"]))
    for name, rows in datasets.items():
        print(f"{name}: rows={len(rows)}, class counts={dict(class_counts(rows))}")
    for name, values in baseline_metrics(datasets).items():
        print(f"Baseline {name}: accuracy={values['accuracy']:.4f}, F1={values['f1']:.4f}")
    print("Implemented: information gain, midpoint search and all ensemble experiments.")


def evaluate(model, datasets, name, training_seconds=None):
    result = {"model": name}
    if training_seconds is not None:
        result["training_seconds"] = training_seconds
        print(f"Training time: {training_seconds:.6f} seconds")
    baselines = baseline_metrics(datasets)
    for split, rows in datasets.items():
        actual = [row[TARGET] for row in rows]
        predictions = model.predict(rows)
        acc = accuracy_score(actual, predictions)
        f1 = f1_score_positive(actual, predictions)
        result[split] = {"accuracy": acc, "f1": f1}
        base = baselines[split]
        print(f"{split}: accuracy={acc:.4f}, F1={f1:.4f}; "
              f"baseline accuracy={base['accuracy']:.4f}, F1={base['f1']:.4f}; "
              f"accuracy difference={acc-base['accuracy']:+.4f}")
    if isinstance(model, DecisionTree):
        result["actual_depth"] = model.tree_depth()
        result["internal_nodes"], result["leaf_nodes"] = model.count_nodes()
    else:
        result["average_depth"] = sum(t.tree_depth() for t in model.trees) / len(model.trees)
        print(f"Average tree depth: {result['average_depth']:.2f}")
    print()
    return result


def run_depths(datasets):
    results = []
    models = {}
    for depth in (2, 3, 4, 5, None):
        name = f"categorical_depth_{depth}"
        print("Maximum depth allowed:", "unrestricted" if depth is None else depth)
        model = DecisionTree(max_depth=depth, random_state=SEED, use_numeric=False)
        model.fit(datasets["train"])
        print("Actual depth:", model.tree_depth())
        print("Internal nodes:", model.count_nodes()[0])
        result = evaluate(model, datasets, name)
        result["maximum_depth"] = depth
        results.append(result)
        models[depth] = model
    print("First three levels of the unrestricted categorical tree:")
    models[None].print_tree(max_depth=2)
    return results


def run_thresholds(datasets):
    tree = DecisionTree(max_depth=2, random_state=SEED, use_numeric=True)
    results = {}
    for feature in ("tenure_months", "monthly_charges", "avg_monthly_gb", "support_tickets_90d"):
        split = tree.best_numeric_split(datasets["train"], feature)
        results[feature] = {"threshold": split.threshold, "information_gain": split.score}
        print(f"{feature}: threshold={split.threshold:.4f}, information gain={split.score:.6f}")
    tree.fit(datasets["train"])
    print("Tree using categorical and numerical attributes:")
    tree.print_tree(max_depth=2)
    return results


def run_bagging(datasets):
    results = []
    models = {}
    for b in (5, 10, 20):
        print("Bagging: B =", b, flush=True)
        model = BaggingEnsemble(n_estimators=b, random_state=SEED)
        model.fit(datasets["train"])
        results.append(evaluate(model, datasets, f"bagging_B{b}"))
        models[b] = model
    return results, models


def run_forests(datasets, bagging_models):
    d = len([key for key in datasets["train"][0] if key not in (TARGET, ID_COLUMN)])
    results = []
    for m in (math.isqrt(d), d):
        print(f"Random forest: B=10, m={m}, d={d}", flush=True)
        model = BaggingEnsemble(n_estimators=10, feature_subsample_size=m, random_state=SEED)
        model.fit(datasets["train"])
        result = evaluate(model, datasets, f"random_forest_m{m}")
        result["m"] = m
        if m == d:
            matches = {name: model.predict(rows) == bagging_models[10].predict(rows)
                       for name, rows in datasets.items()}
            print("m=d predictions match Bagging:", matches)
            assert all(matches.values())
            result["matches_bagging"] = matches
        results.append(result)
    return results


def run_weighted(datasets):
    # Identical bootstrap trees provide a control for the effect of vote weights.
    options = {"n_estimators": 20, "max_depth": 4, "random_state": SEED}
    uniform = BaggingEnsemble(**options)
    uniform.fit(datasets["train"])
    weighted = ValidationWeightedEnsemble(**options)
    weighted.fit(datasets["train"], datasets["validation"])
    interpretable = DecisionTree(max_depth=3, random_state=SEED, use_numeric=True)
    interpretable.fit(datasets["train"])
    print("Vote weights (validation accuracy):", [round(w, 4) for w in weighted.weights])
    results = [evaluate(uniform, datasets, "shallow_bagging_depth4"),
               evaluate(weighted, datasets, "validation_weighted_depth4"),
               evaluate(interpretable, datasets, "single_mixed_depth3")]
    results[1]["weights"] = weighted.weights
    print("Validation metrics reflect weight selection; test labels do not set weights.")
    print("Interpretable comparison: first three levels of the mixed-feature tree")
    interpretable.print_tree(max_depth=2)
    return results


def run_early_stopping(datasets):
    results = []
    for name, m in (("bagging", None), ("random_forest", 3)):
        for enabled in (False, True):
            print(f"{name}: early stopping = {enabled}", flush=True)
            model = BaggingEnsemble(
                n_estimators=20, feature_subsample_size=m, random_state=SEED,
                min_samples_split=20 if enabled else 2,
                min_gain=0.01 if enabled else 0.0,
            )
            start = time.perf_counter()
            model.fit(datasets["train"])
            elapsed = time.perf_counter() - start
            result = evaluate(model, datasets, f"{name}_early_{enabled}", elapsed)
            result["early_stopping"] = enabled
            result["min_samples_split"] = model.min_samples_split
            result["min_gain"] = model.min_gain
            results.append(result)
    return results


def save_results(results, output_dir=None):
    output_dir = Path(output_dir) if output_dir else BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["section", "model", "training_seconds", "actual_depth", "average_depth",
                  "internal_nodes", "split", "accuracy", "f1", "baseline_accuracy", "baseline_f1"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for section in ("depths", "bagging", "forests", "weighted", "early_stopping"):
            for model in results[section]:
                for split in ("train", "validation", "test"):
                    row = {key: model.get(key, "") for key in fieldnames[:6]}
                    row.update(section=section, split=split, **model[split],
                               baseline_accuracy=results["baseline"][split]["accuracy"],
                               baseline_f1=results["baseline"][split]["f1"])
                    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run all churn prediction experiments")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    random.seed(SEED)
    datasets = load_datasets(args.data_dir)
    print_summary(datasets)
    results = {"seed": SEED, "baseline": baseline_metrics(datasets),
               "depths": run_depths(datasets), "thresholds": run_thresholds(datasets)}
    results["bagging"], models = run_bagging(datasets)
    results["forests"] = run_forests(datasets, models)
    results["weighted"] = run_weighted(datasets)
    results["early_stopping"] = run_early_stopping(datasets)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
