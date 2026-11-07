"""Run with: streamlit run app.py"""

import numpy as np
import pandas as pd
import streamlit as st

from ml_core import evaluate, load_models, load_split, parse_test_csv, test_csv

st.set_page_config(page_title="Multiclass Model Comparison", page_icon="📊", layout="wide")
st.title("Multiclass Model Comparison")
st.write("Compare five classifiers on handwritten digits: 1,797 samples · "
         "64 pixel features · 10 classes (0–9).")
st.caption("UCI Optical Recognition of Handwritten Digits, 1,797-row scikit-learn subset. "
           "Fixed stratified 80/20 split, seed 42; 1,437 training and 360 test rows.")


@st.cache_resource(show_spinner="Loading saved classifiers…")
def fitted_models():
    return load_models()


try:
    models = fitted_models()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load saved models. Run `python train.py` before launching the app. {exc}")
    st.stop()
_, X_test, _, y_test = load_split()
with st.sidebar:
    st.header("Evaluation settings")
    selected = st.selectbox("Model", list(models))
    source = st.radio("Test data", ["Built-in held-out test set", "Upload test CSV"])
    st.download_button("Download sample test CSV", test_csv(X_test, y_test),
                       file_name="digits_test.csv", mime="text/csv")
    st.caption("Upload test data only, up to 5 MB and 5,000 rows. "
               "Uploads are used only for evaluation.")
    st.markdown("[Dataset source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)")

with st.expander("CSV format and metric definitions"):
    st.write("Use the sample CSV format: pixel_0_0 through pixel_7_7 (64 features), "
             "plus target (the true digit, 0–9). Pixel values are integers from 0 to 16. "
             "Column order may vary; missing or extra columns are rejected. "
             "These trained models accept only this digits schema.")
    st.write("Precision, recall and F1 are macro averages over all 10 classes, "
             "giving each class equal weight. Undefined class ratios are set to zero. "
             "AUC uses predicted probabilities and macro one-vs-rest averaging; "
             "it is unavailable when the test CSV omits any class. MCC uses the "
             "multiclass definition. All preprocessing is fitted on training data only.")
    st.caption("Use independent test rows for meaningful evaluation. The app cannot "
               "verify the provenance of arbitrary uploads. These are fixed baseline "
               "models; the test set is not used for hyperparameter tuning.")

if source == "Upload test CSV":
    uploaded = st.file_uploader("Upload labeled test data (.csv)", type=["csv"])
    if uploaded is None:
        st.info("Download the sample test CSV, then upload it here to evaluate a model.")
        st.stop()
    try:
        X_test, y_test = parse_test_csv(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

st.subheader(f"{selected} — {len(X_test):,} test rows")
metrics, matrix, report, predicted = evaluate(models[selected], X_test, y_test)
for column, (name, value) in zip(st.columns(6), metrics.items()):
    column.metric(name, "N/A" if np.isnan(value) else f"{value:.4f}")
if np.isnan(metrics["AUC (macro OvR)"]):
    st.info("AUC is unavailable: include at least one test example of every digit (0–9).")

comparison_tab, confusion_tab, report_tab, data_tab = st.tabs(
    ["Compare all models", "Confusion matrix", "Classification report", "Test data & predictions"])
with comparison_tab:
    comparison = pd.DataFrame({name: evaluate(model, X_test, y_test)[0]
                               for name, model in models.items()}).T
    comparison.index.name = "Model"
    st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
    st.download_button("Download metric comparison", comparison.to_csv(),
                       file_name="model_comparison.csv", mime="text/csv")
with confusion_tab:
    st.caption("Rows: actual digit. Columns: predicted digit.")
    st.dataframe(matrix.style.background_gradient(cmap="Blues", axis=None),
                 width="stretch")
with report_tab:
    st.dataframe(report.style.format("{:.4f}"), width="stretch")
with data_tab:
    predictions = X_test.assign(target=np.asarray(y_test), prediction=predicted)
    st.dataframe(predictions, width="stretch")
    st.download_button("Download predictions", predictions.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")
