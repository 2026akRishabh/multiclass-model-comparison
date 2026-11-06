"""Generate comparison results and a labeled held-out CSV for submission."""

import json
from pathlib import Path
import pandas as pd
from ml_core import MODEL_DIR, evaluate, save_models, test_csv, train_models


def main():
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    models, X_test, y_test = train_models()
    save_models(models)
    rows = {}
    for name, model in models.items():
        metrics, matrix, report, _ = evaluate(model, X_test, y_test)
        rows[name] = metrics
        slug = name.lower().replace(" ", "_").replace("-", "_")
        matrix.to_csv(output / f"{slug}_confusion_matrix.csv")
        report.to_csv(output / f"{slug}_classification_report.csv")
    comparison = pd.DataFrame.from_dict(rows, orient="index")
    comparison.index.name = "Model"
    comparison.to_csv(output / "model_comparison.csv")
    (output / "digits_test.csv").write_bytes(test_csv(X_test, y_test))
    import sklearn
    import numpy
    metadata = {"dataset": "UCI digits: scikit-learn 1797-row subset",
                "features": 64, "classes": 10, "train_rows": 1437,
                "test_rows": 360, "seed": 42, "test_fraction": 0.2,
                "stratified": True, "sklearn": sklearn.__version__,
                "numpy": numpy.__version__, "pandas": pd.__version__}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved evaluation outputs to {output}")
    print(f"Saved five fitted models and preprocessing metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
