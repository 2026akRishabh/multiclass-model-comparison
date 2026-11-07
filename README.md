# Multiclass classification comparison

## Problem statement

Build a multiclass classification system that predicts a handwritten digit (0-9)
from 64 pixel features. Train five classification models on the same public dataset
and compare their performance using Accuracy, AUC, Precision, Recall, F1 and
Matthews Correlation Coefficient (MCC). The dataset must contain at least 12
features and 500 instances.

Provide a Streamlit application with test CSV upload, model selection, evaluation
metrics, a confusion matrix and a classification report. Training happens locally;
the deployed app loads the five saved classifiers and their fitted preprocessing.

## GitHub repository link

[Project repository](https://github.com/2026akRishabh/multiclass-model-comparison)

## Dataset description

- Source: [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
- This project uses the **1,797-row subset distributed with scikit-learn** through
  [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
  The full UCI dataset has 5,620 instances; this project does not use the full dataset.
- 64 numeric features: an 8 × 8 grid of integer pixel values between 0 and 16.
- 10 target classes: digits 0–9. No missing values.
- Meets the assignment requirements: at least 12 features, at least 500 instances,
  and more than two classes.

The subset is split into 1,437 training rows and 360 test rows using stratification
and random seed 42. This is a new split of the scikit-learn subset, not the original
UCI train/test partition. Models use identical splits. StandardScaler is fitted
only on the training partition for Logistic Regression, KNN and SVM. The tree and
Gaussian NB use raw features. There is no test-set tuning or feature selection.

## Project structure

```text
MLModels/
|-- app.py                          # Streamlit application; loads saved models
|-- ml_core.py                      # Data split, preprocessing, models, metrics, CSV validation
|-- train.py                        # Train, save models and generate evaluation files
|-- requirements.txt
|-- README.md
|-- .streamlit/config.toml          # Upload limit and theme
|-- model/
|   |-- logistic_regression.joblib  # Fitted StandardScaler + LogisticRegression pipeline
|   |-- decision_tree.joblib        # Fitted DecisionTreeClassifier
|   |-- knn.joblib                  # Fitted StandardScaler + KNeighborsClassifier pipeline
|   |-- naive_bayes.joblib          # Fitted GaussianNB
|   |-- svm.joblib                  # Fitted StandardScaler + SVC pipeline
|   `-- metadata.json              # Feature order, classes, preprocessing and library versions
|-- results/
|   |-- model_comparison.csv
|   |-- digits_test.csv            # Labeled held-out test data for app upload
|   |-- metadata.json
|   `-- *_classification_report.csv / *_confusion_matrix.csv
`-- tests/test_ml_core.py
```

The fitted StandardScaler artifacts for Logistic Regression, KNN and SVM are
embedded in their `.joblib` pipelines. Loading a pipeline applies exactly the
preprocessing learned during training, so separate scaler files are unnecessary.
Decision Tree and Gaussian NB require no scaling. All training and preprocessing
source code is included in `ml_core.py` and `train.py`.

## Commands to run locally

Use Python 3.14 (the version used for core validation). In PowerShell, from this
project's directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

If `python` is unavailable in PowerShell on the current development machine,
create the virtual environment with the existing interpreter instead:

```powershell
E:\Environment\ml-env\Scripts\python.exe -m venv .venv
```

Then run the remaining `.venv` commands above. No activation command is required.

`train.py` writes `results/model_comparison.csv`, each model's classification
report and confusion matrix, `results/metadata.json`, and `results/digits_test.csv`.
It also saves all five fitted classifiers to `model/`. Re-running `train.py`
replaces these generated files. The app loads the saved models once per running
process and caches them with `st.cache_resource`; it does not train models.
Commit `model/` to GitHub before deploying. No dataset download or credentials
are required at runtime. Use the pinned scikit-learn version in `requirements.txt`
to load the saved artifacts. After regenerating artifacts, restart the app to
clear cached models. Only load `.joblib` files produced by trusted project code.

## Models used

| Model | Settings |
| --- | --- |
| Logistic Regression | StandardScaler; max_iter=3000; seed=42 |
| Decision Tree | Default unpruned tree; seed=42 |
| K-Nearest Neighbors | StandardScaler; k=5; uniform weights; Euclidean distance |
| Gaussian Naive Bayes | Default GaussianNB, var_smoothing=1e-9 |
| Support Vector Machine | StandardScaler; RBF kernel; C=1; gamma=scale; probability=True; seed=42 |

Other parameters retain the scikit-learn defaults. Probability estimation is
enabled for SVM so all five models can be evaluated with multiclass ROC AUC.

## Metrics

| Metric | Definition used |
| --- | --- |
| Accuracy | Correct predictions / test rows |
| AUC | Macro one-vs-rest ROC AUC from predicted class probabilities |
| Precision | Unweighted mean of per-class precision |
| Recall | Unweighted mean of per-class recall |
| F1 | Unweighted mean of per-class F1 scores |
| MCC | Multiclass Matthews correlation coefficient |

Precision, recall and F1 always include all ten labels. Undefined ratios are set
to zero. AUC is displayed as N/A if any of the ten classes is missing from an
uploaded test set. MCC ranges from -1 to 1; other metrics range from 0 to 1.
Higher values are better. Confusion matrices use actual labels as rows and
predicted labels as columns. Report support is the number of true test examples.

The comparison is a single-split baseline, not a claim of statistically significant
superiority. Macro metrics are appropriate when each digit should have equal
importance. Gaussian NB assumes conditionally independent features, which may
limit its performance on neighboring pixels. A single tree can overfit; KNN and
RBF SVM can capture nonlinear boundaries. Assess the generated results before
drawing conclusions about these models.

## Comparison table

All values below are calculated on the same 360 held-out rows. AUC is macro
one-vs-rest; Precision, Recall and F1 use macro averaging.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9722 | 0.9991 | 0.9721 | 0.9719 | 0.9719 | 0.9692 |
| Decision Tree | 0.8250 | 0.9024 | 0.8236 | 0.8242 | 0.8230 | 0.8057 |
| KNN | 0.9639 | 0.9950 | 0.9646 | 0.9637 | 0.9634 | 0.9600 |
| Naive Bayes (Gaussian) | 0.8111 | 0.9705 | 0.8463 | 0.8103 | 0.8137 | 0.7940 |
| Support Vector Machine | 0.9750 | 0.9995 | 0.9760 | 0.9748 | 0.9748 | 0.9723 |

## Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Second-highest accuracy (97.22%) and macro F1 (0.9719), with AUC 0.9991. A strong linear baseline on the scaled features; it correctly predicts 350 of 360 test rows. |
| Decision Tree | Accuracy 82.50%, macro F1 0.8230 and MCC 0.8057. It performs below the scaled linear and distance-based models. An unpruned tree can overfit; this experiment does not establish the cause without training or cross-validation comparisons. |
| KNN | Accuracy 96.39%, macro F1 0.9634 and AUC 0.9950. It performs well with scaled pixel distances, although its accuracy is lower than SVM and Logistic Regression. |
| Naive Bayes | Gaussian NB has the lowest accuracy (81.11%) and macro F1 (0.8137), despite AUC 0.9705. Its probability ranking is stronger than its final class predictions. Correlated neighboring pixels may challenge its conditional independence assumption. |
| Support Vector Machine | Highest accuracy (97.50%), macro F1 (0.9748), AUC (0.9995) and MCC (0.9723). The RBF kernel provides a strong nonlinear classifier on scaled features, correctly predicting 351 of 360 test rows. |
| **Overall, winner for this dataset?** | **Support Vector Machine**, leading all six reported metrics on this held-out split. Its accuracy advantage over Logistic Regression is only one correct prediction; this single split does not establish a statistically significant advantage. |

Exact unrounded values and environment versions are recorded in `results/`.

## Upload format

Upload **test data only**, using `results/digits_test.csv` or the sample downloaded
from the sidebar. The required header contains `pixel_0_0` through `pixel_7_7`
(row-major order) and `target`. Column order may vary. The label column is required
to calculate evaluation metrics. Additional columns, missing values, nonnumeric
values, out-of-range pixels, and unknown targets are rejected. Uploads are limited
to 5 MB and 5,000 rows.

This app evaluates pretrained digits classifiers; it does not train classifiers
on arbitrary uploaded datasets. Use independent test data with the same pixel
representation. The app cannot establish whether arbitrary uploaded rows were
previously used for training. Uploaded files are not saved by application code.

## Deploy on Streamlit Community Cloud

1. Run `train.py` locally, then push this directory to a GitHub repository, including
   `app.py`, `ml_core.py`, `train.py`, `requirements.txt`, `.streamlit/config.toml`,
   and **all files in `model/`**. Include `results/` if you want
   the generated comparison files available in the repository. Exclude `.venv/`.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose
   **Create app**, and select the repository and branch.
3. Set the entrypoint to `app.py` (include a folder prefix if the project is nested).
4. Select Python 3.14 in the deployment's advanced settings and deploy.
5. Check the deployment logs, then test the model dropdown, sample CSV upload,
   metric comparison and downloads.

See Streamlit's official [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [dependency guidance](https://docs.streamlit.io/deploy/concepts/dependencies).
Deployment requires your GitHub/Streamlit account. The project is published on
GitHub; deployment to Streamlit Community Cloud is pending.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m py_compile app.py ml_core.py train.py
```

Tests cover split isolation, training-only scaling, all five models' metrics,
CSV roundtrips, reordered columns, invalid uploads, missing-class AUC, and saved
model roundtrips (identical predictions and probabilities, including preprocessing).
All six core tests passed in the project virtual environment. Automated Streamlit
AppTest checks passed with Streamlit 1.67.0 for all five model selections, rendered
metrics and tables, evaluation with a simulated sample CSV upload, and rejection
of an invalid CSV. These checks exercise the app without a browser; browser-based
file selection, downloads and the hosted deployment still require manual checks.
`requirements.txt` pins the numeric packages used for core validation and allows
Streamlit 1.51 or newer within major version 1.
