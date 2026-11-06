import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from ml_core import (
    FEATURES, MODEL_FILES, evaluate, load_models, load_split, parse_test_csv,
    save_models, test_csv, train_models,
)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.X, cls.y = train_models()

    def test_stratified_split_and_train_only_scaling(self):
        X_train, X_test, y_train, y_test = load_split()
        self.assertEqual((len(X_train), len(X_test)), (1437, 360))
        self.assertFalse(set(X_train.index) & set(X_test.index))
        self.assertEqual(set(y_train), set(range(10)))
        self.assertEqual(set(y_test), set(range(10)))
        scaler = self.models["Logistic Regression"].named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, X_train.mean())

    def test_all_five_models_have_valid_metrics(self):
        self.assertEqual(len(self.models), 5)
        for name, model in self.models.items():
            with self.subTest(model=name):
                metrics, matrix, report, predicted = evaluate(model, self.X, self.y)
                self.assertEqual(len(metrics), 6)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                self.assertTrue(all(0 <= value <= 1 for key, value in metrics.items() if key != "MCC"))
                self.assertTrue(-1 <= metrics["MCC"] <= 1)
                self.assertEqual(matrix.shape, (10, 10))
                self.assertEqual(matrix.to_numpy().sum(), len(self.y))
                self.assertEqual(len(predicted), len(self.y))
                self.assertIn("macro avg", report.index)

    def test_csv_roundtrip_and_reordered_columns(self):
        X, y = parse_test_csv(test_csv(self.X, self.y))
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(y, self.y)
        frame = self.X.assign(target=np.asarray(self.y))
        X, _ = parse_test_csv(frame[frame.columns[::-1]].to_csv(index=False).encode())
        self.assertEqual(list(X), FEATURES)

    def test_upload_validation(self):
        base = self.X.head(3).assign(target=np.asarray(self.y.head(3)))
        invalid = [base.drop(columns=[FEATURES[0]]), base.assign(extra=0),
                   base.assign(target=10), base.assign(target=1.5),
                   base.assign(**{FEATURES[0]: np.nan}),
                   base.assign(**{FEATURES[0]: np.inf}),
                   base.assign(**{FEATURES[0]: 17}),
                   base.assign(**{FEATURES[0]: 0.5}),
                   base.assign(**{FEATURES[0]: "text"}), base.iloc[:0]]
        for frame in invalid:
            with self.assertRaises(ValueError):
                parse_test_csv(frame.to_csv(index=False).encode())
        for raw in [b"", b"\xff", b"target,target\n0,0", b"x" * (5 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                parse_test_csv(raw)

    def test_missing_class_auc_is_unavailable(self):
        mask = self.y != 9
        metrics, matrix, _, _ = evaluate(
            self.models["Support Vector Machine"], self.X[mask], self.y[mask])
        self.assertTrue(np.isnan(metrics["AUC (macro OvR)"]))
        self.assertEqual(matrix.shape, (10, 10))

    def test_saved_models_preserve_preprocessing_predictions_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            save_models(self.models, directory)
            self.assertTrue(all((Path(directory) / filename).is_file()
                                for filename in MODEL_FILES.values()))
            loaded = load_models(directory)
            for name, original in self.models.items():
                with self.subTest(model=name):
                    np.testing.assert_array_equal(original.predict(self.X), loaded[name].predict(self.X))
                    np.testing.assert_allclose(original.predict_proba(self.X),
                                               loaded[name].predict_proba(self.X))
            metadata_path = Path(directory) / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sklearn_version"] = "incompatible"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Saved models use scikit-learn"):
                load_models(directory)


if __name__ == "__main__":
    unittest.main()
