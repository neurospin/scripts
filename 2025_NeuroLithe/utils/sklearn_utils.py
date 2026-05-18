import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef)

from utils.plot_utils import plot_roc


# %% ──────────────────────────────────────────────────────────────────────────
#  Sklearn Utilities
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_split(pipeline, step=-1):
    """Split pipeline into two pipelines body[:step] and head[step:].
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    step : int
        The step where to split the pipeline
    Returns
    -------
    body : Pipeline
        A new Pipeline object containing "step" fist steps.
    head : Pipeline
        A new Pipeline object containing the remaining steps.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")


    body = Pipeline(pipeline.steps[:step])
    head = Pipeline(pipeline.steps[step:])
    
    return body, head
    
    
def pipeline_behead(pipeline):
    """Separate preprocessing transformers from the prediction head of a pipeline estimator.
    This function assumes that the last step of the pipeline is the prediction head
    (e.g., a classifier or regressor) and all previous steps are preprocessing steps.
    
    Parameters
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object.
    Returns
    -------
    transformers : Pipeline
        A new Pipeline object containing only the preprocessing steps.
    prediction_head : object
        The last step of the original pipeline, which is the prediction head (e.g., classifier
        Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', LogisticRegression())
    ... ])
    >>> pipe.fit(X, y)
    Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression())])
    >>> transformers, prediction_head = pipeline_behead(pipe)
    >>> # Apply preprocessing to data
    >>> X_preprocessed = transformers.transform(X)
    >>> # Use prediction head on preprocessed data
    >>> prediction_head.predict(X_preprocessed)
    array([0, 0, 1, 1])
    >>> pipe.fit(X, y)
    >>> pipe.predict(X)
    array([0, 0, 1, 1])
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Estimator must be a Pipeline instance")

    #pipeline = clone(pipeline)  # Clone the estimator to avoid modifying the original
    preprocessing_steps = pipeline.steps[:-1]
    # Get the last step (the predictor)
    _, prediction_head = pipeline.steps[-1]
    # Create a new pipeline with only the preprocessing steps
    transformers = Pipeline(preprocessing_steps)
    
    return transformers, prediction_head

def get_predictor(estimator):
    """Unwrap a fitted meta-estimator and return the underlying predictor.

    Handles any object exposing ``best_estimator_`` (e.g. GridSearchCV,
    RandomizedSearchCV). Returns the estimator itself when it is already
    a plain fitted predictor.
    """
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_
    return estimator


def get_coef(estimator):
    """Extract a 1-D coefficient array from a fitted predictor.

    Tries ``coef_`` (linear models) then ``feature_importances_`` (tree
    models). Raises ``AttributeError`` if neither attribute exists.
    """
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        return coef[0] if coef.ndim > 1 else coef
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    raise AttributeError(
        f"{type(estimator).__name__} exposes neither 'coef_' nor "
        "'feature_importances_' — cannot extract coefficients."
    )


def oof_arrays_from_cv(vals_cv: list, cv, X: np.ndarray, y: np.ndarray, split: str= 'test') -> np.ndarray:
    """Reconstruct a full out-of-fold array from per-fold results.

    Inverse of the accumulation loop used in cross-validation: places each
    fold's values back at the test indices to produce a single array aligned
    with the original sample order.

    Parameters
    ----------
    vals_cv : list of ndarray, one per fold, each of shape (n_test_i, ...)
    cv      : CV splitter (same instance used during fitting)
    X       : feature matrix (used only for cv.split)
    y       : target array  (used only for cv.split)
    split   : str, either 'test' or 'train' indicating which indices to reconstruct
    Returns
    -------
    out : ndarray of shape (n_samples, ...) with fold results placed at their
          original test indices
    """
    if split is 'test':
        n_samples = np.sum([len(te) for _, te in cv.split(X, y)])
    else:
        n_samples = np.sum([len(tr) for tr, _ in cv.split(X, y)])
    first    = vals_cv[0]
    out      = np.zeros((n_samples, *first.shape[1:]), dtype=first.dtype)
    
    for (tr, te), vals in zip(cv.split(X, y), vals_cv):
        if split is 'test':
            out[te] = vals
        else:
            out[tr] = vals
    return out

# %% ── Model Evaluation ──────────────────────────────────────────────────────────
def plot_classification_report(X: np.ndarray,
                     y: np.ndarray,
                     fitted_estimators_cv: list,
                     cv) -> pd.DataFrame:
    """
    Derive out-of-fold predictions from the CV estimators, plot the confusion
    matrix and ROC curve, and return a tidy metrics DataFrame.

    Reports each metric with two strategies:
      - Average-of-folds : Average-of-folds / "score-then-average": compute the metric within each fold,
      then average across folds. This is what cross_val_score and cross_validate do — sklearn states
      "The function cross_val_score takes an average over cross-validation folds", and 
      "The performance measure reported by k-fold cross-validation is then the average of the values
      computed in the loop".

      - Pooled (OOF)     : Pooled / "predict-then-score" (sometimes called "aggregate" or "concatenated" evaluation):
      concatenate Out-Of-Fold predictions, then score once on the full vector.
      This is what you get by passing the output of cross_val_predict to a metric.
      In the literature it's often called aggregate or pooled cross-validation, in contrast to the averaged version.
      The terms you'll see most often are simply "average across folds" vs. "pooled (out-of-fold) predictions".

    Parameters
    ----------
    X                    : raw feature matrix
    y                    : true binary labels
    fitted_estimators_cv : list of fitted pipelines from fit_pipeline
    cv                   : CV splitter used in fit_pipeline

    Returns
    -------
    metrics : pd.DataFrame
        Columns: metric, avg_folds, pooled_oof, fold_values (list per row).
    """
    y_pred_cv  = np.empty(len(y), dtype=int)
    y_proba_cv = np.empty(len(y))
    fold_ba, fold_auc, fold_prec, fold_rec, fold_f1, fold_mcc = [], [], [], [], [], []

    for (_, te), pipe in zip(cv.split(X, y), fitted_estimators_cv):
        y_pred_cv[te]  = pipe.predict(X[te])
        y_proba_cv[te] = pipe.predict_proba(X[te])[:, 1]
        fold_ba.append(balanced_accuracy_score(y[te], y_pred_cv[te]))
        fold_auc.append(roc_auc_score(y[te], y_proba_cv[te]))
        fold_prec.append(precision_score(y[te], y_pred_cv[te], zero_division=0))
        fold_rec.append(recall_score(y[te], y_pred_cv[te], zero_division=0))
        fold_f1.append(f1_score(y[te], y_pred_cv[te], zero_division=0))
        fold_mcc.append(matthews_corrcoef(y[te], y_pred_cv[te]))

    rows = [
        ("Balanced Accuracy", fold_ba,   balanced_accuracy_score(y, y_pred_cv)),
        ("ROC-AUC",           fold_auc,  roc_auc_score(y, y_proba_cv)),
        ("Precision",         fold_prec, precision_score(y, y_pred_cv, zero_division=0)),
        ("Recall",            fold_rec,  recall_score(y, y_pred_cv, zero_division=0)),
        ("F1",                fold_f1,   f1_score(y, y_pred_cv, zero_division=0)),
        ("MCC",               fold_mcc,  matthews_corrcoef(y, y_pred_cv)),
    ]
    metrics = pd.DataFrame(
        [(name, np.mean(folds), np.std(folds), pooled, folds)
         for name, folds, pooled in rows],
        columns=["metric", "avg_folds", "std_folds", "pooled_oof", "fold_values"],
    )

    print("\n━━━  Model Performance  ━━━")
    print(f"  {'Metric':<22}  {'Avg-of-folds':>13}  {'±std':>6}  {'Pooled OOF':>11}")
    print(f"  {'-'*22}  {'-'*13}  {'-'*6}  {'-'*11}")
    for _, row in metrics.iterrows():
        print(f"  {row.metric:<22}  {row.avg_folds:>13.3f}  {row.std_folds:>6.3f}  {row.pooled_oof:>11.3f}")

    print("\n━━━  Classification Report (Pooled OOF)  ━━━")
    print(classification_report(y, y_pred_cv))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(y, y_pred_cv, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix  (pooled OOF)", fontweight="bold")

    plot_roc(y, y_proba_cv,
             title="ROC Curve  (pooled OOF)",
             ax=axes[1])

    plt.suptitle("Model Evaluation — Nested Cross-Validation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved model_evaluation.png")

    return metrics

