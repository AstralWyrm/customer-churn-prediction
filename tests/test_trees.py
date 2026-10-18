# Yi Han z5614040
import math
import random
import unittest

from trees import (BaggingEnsemble, DecisionTree, ValidationWeightedEnsemble,
                   entropy, f1_score_positive)


class TreeTests(unittest.TestCase):
    def test_entropy_and_information_gain(self):
        rows = [{"plan": "A", "churned": 0}, {"plan": "B", "churned": 1}]
        self.assertAlmostEqual(entropy(rows), 1.0)
        self.assertAlmostEqual(DecisionTree().information_gain(rows, [[rows[0]], [rows[1]]]), 1.0)

    def test_f1_zero_and_positive_cases(self):
        self.assertEqual(f1_score_positive([1, 0], [0, 1]), 0.0)
        self.assertEqual(f1_score_positive([0, 0], [0, 0]), 0.0)
        self.assertAlmostEqual(f1_score_positive([1, 1, 0], [1, 0, 0]), 2 / 3)

    def test_numeric_midpoints_against_exhaustive_information_gain(self):
        rng = random.Random(3411)
        tree = DecisionTree()
        for _ in range(80):
            rows = [{"tenure_months": rng.randrange(7), "churned": rng.randrange(2)}
                    for _ in range(20)]
            split = tree.best_numeric_split(rows, "tenure_months")
            values = sorted({row["tenure_months"] for row in rows})
            best_threshold = None
            best_score = -math.inf
            for first, second in zip(values, values[1:]):
                threshold = (first + second) / 2
                left = [r for r in rows if r["tenure_months"] <= threshold]
                right = [r for r in rows if r["tenure_months"] > threshold]
                score = tree.information_gain(rows, [left, right])
                if score > best_score:
                    best_threshold, best_score = threshold, score
            self.assertEqual(split.threshold, best_threshold)
            self.assertAlmostEqual(split.score, best_score)
        self.assertIsNone(tree.best_numeric_split([{"tenure_months": 1, "churned": 0}] * 3, "tenure_months"))

    def test_leaf_stopping_majority_and_unseen_category(self):
        tree = DecisionTree(use_numeric=False)
        rows = [{"plan": "A", "churned": 0}, {"plan": "A", "churned": 0},
                {"plan": "B", "churned": 1}]
        tree.fit(rows)
        self.assertEqual(tree.predict_one({"plan": "unseen"}), 0)
        pure = DecisionTree()
        pure.fit([{"plan": "A", "churned": 1}] * 3)
        self.assertTrue(pure.root.is_leaf)
        no_gain = DecisionTree()
        no_gain.fit([{"plan": "A", "churned": 0}, {"plan": "A", "churned": 1}])
        self.assertTrue(no_gain.root.is_leaf)
        self.assertEqual(no_gain.root.prediction, 0)

    def test_early_stopping_rules(self):
        rows = [{"tenure_months": x, "churned": int(x > 4)} for x in range(10)]
        tree = DecisionTree(min_samples_split=20)
        tree.fit(rows)
        self.assertTrue(tree.root.is_leaf)
        tree = DecisionTree(min_gain=1.1)
        tree.fit(rows)
        self.assertTrue(tree.root.is_leaf)
        tree = DecisionTree()
        tree.fit(rows)
        self.assertFalse(tree.root.is_leaf)

    def test_forest_all_features_and_repeat_fit_match_bagging(self):
        rows = [{"tenure_months": i, "region": str(i % 3), "churned": int(i % 4 == 0)}
                for i in range(30)]
        bagging = BaggingEnsemble(n_estimators=5, random_state=3411)
        forest = BaggingEnsemble(n_estimators=5, feature_subsample_size=2, random_state=3411)
        bagging.fit(rows)
        forest.fit(rows)
        predictions = bagging.predict(rows)
        self.assertEqual(predictions, forest.predict(rows))
        bagging.fit(rows)
        self.assertEqual(predictions, bagging.predict(rows))

    def test_weighted_vote_and_tie_breaking(self):
        class FixedTree:
            def __init__(self, label):
                self.label = label
            def predict(self, rows):
                return [self.label] * len(rows)
        model = ValidationWeightedEnsemble()
        model.trees = [FixedTree(0), FixedTree(0), FixedTree(1)]
        model.weights = [0.1, 0.1, 0.9]
        self.assertEqual(model.predict([{}]), [1])
        model.weights = [0.4, 0.3, 0.7]
        self.assertEqual(model.predict([{}]), [0])


if __name__ == "__main__":
    unittest.main()
