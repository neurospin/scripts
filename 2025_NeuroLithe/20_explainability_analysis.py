"""
Explainability Analysis Pipeline
=================================
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn import linear_model as lm, preprocessing
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer

import shap
import warnings
warnings.filterwarnings("ignore")

from utils.sklearn_utils import (pipeline_behead, get_predictor, get_coef,
                                 oof_arrays_from_cv, plot_classification_report)



# %% ── A. SHAP — LinearExplainer ─────────────────────────────────────────────────
def shap_analysis(estimators: list,
                  X: np.ndarray,
                  y: np.ndarray,
                  features: list,
                  cv) -> tuple:
    """
    Compute out-of-fold SHAP values with LinearExplainer across the CV loop.
    Each fold's test samples are scaled with that fold's scaler and explained
    with that fold's best LogisticRegression — no leakage.

    Parameters
    ----------
    estimators : list of fitted pipelines (one per CV fold)
    X          : raw (unscaled) feature matrix
    y          : binary response array (used only for cv.split)
    features   : ordered list of feature names
    cv         : CV splitter used during fitting

    Returns
    -------
    shapvals_cv   : list of ndarray (n_test_i, n_features), one per fold —
                    raw per-fold SHAP values for downstream analysis
    X_trn_cv      : list of ndarray (n_test_i, n_features), one per fold —
                    scaled input features (body-transformed) at test time
    shap_stats_df : DataFrame with columns [Feature, Mean, Std] sorted by
                    |Mean| desc — Mean and Std of |SHAP| computed over all OOF samples
    """
    print("\n━━━  SHAP Analysis (LinearExplainer, out-of-fold)  ━━━")
    shapvals_cv = []
    X_trn_cv    = []

    for fold, ((_, te), estimator) in enumerate(zip(cv.split(X, y), estimators), 1):
        if isinstance(estimator, Pipeline):
            body, head = pipeline_behead(estimator)
            X_trn_te   = body.transform(X[te])
        else:
            X_trn_te = X[te]
            head     = estimator
        predictor = get_predictor(head)
        X_trn_cv.append(X_trn_te)
        masker    = shap.maskers.Independent(X_trn_te, max_samples=min(200, len(te)))
        explainer = shap.LinearExplainer(predictor, masker=masker)
        shapvals_cv.append(explainer.shap_values(X_trn_te))
        print(f"   fold {fold} done")

    shap_arr      = oof_arrays_from_cv(shapvals_cv, cv, X, y)
    shap_stats_df = (pd.DataFrame({
        "Feature": features,
        "Mean":    np.abs(shap_arr).mean(axis=0),
        "Std":     np.abs(shap_arr).std(axis=0),
    }).sort_values("Mean", ascending=False)
     .reset_index(drop=True))

    return shapvals_cv, X_trn_cv, shap_stats_df


def plot_shap_analysis(shapvals_cv: list,
                       X_trn_cv: list,
                       features: list,
                       cv,
                       X: np.ndarray,
                       y: np.ndarray) -> None:
    """
    Plot SHAP beeswarm, global bar chart, and dependence plots for the top-2
    features.

    Parameters
    ----------
    shapvals_cv : list of ndarray (n_test_i, n_features), one per fold — from shap_analysis
    X_trn_cv    : list of ndarray (n_test_i, n_features), one per fold — from shap_analysis
    features    : ordered list of feature names
    cv          : CV splitter used during fitting
    X           : raw feature matrix (used for OOF reconstruction)
    y           : response array (used for OOF reconstruction)
    """
    shap_vals = oof_arrays_from_cv(shapvals_cv, cv, X, y)
    X_trn     = oof_arrays_from_cv(X_trn_cv, cv, X, y)

    shap.summary_plot(shap_vals, X_trn, feature_names=features, show=False, plot_type="dot")
    plt.title("SHAP Beeswarm  —  log-odds contribution per patient (OOF)", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_beeswarm.png")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0),
                         index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(features))))
    mean_abs.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (log-odds units)", fontsize=11)
    ax.set_title("SHAP Global Importance — Lithium Response\n"
                 "Correlated features share credit — no double-counting", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_bar.png")

    top2 = mean_abs.nlargest(min(2, len(features))).index.tolist()
    fig, axes = plt.subplots(1, len(top2), figsize=(6 * len(top2), 4), squeeze=False)
    for ax, feat in zip(axes[0], top2):
        idx = features.index(feat)
        ax.scatter(X_trn[:, idx], shap_vals[:, idx],
                   c=shap_vals[:, idx], cmap="coolwarm", alpha=0.6, s=20)
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel(f"{feat}  (standardised)")
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence — {feat}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✔  Saved shap_dependence.png")


# %% ── B. Permutation Importance (via cv_test) ───────────────────────────────────
def permutation_importance_analysis(estimators: list,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    features: list,
                                    cv) -> tuple:
    """
    Permutation importance evaluated out-of-fold using pre-fitted CV estimators:
    for each test fold we permute features and measure the drop in balanced
    accuracy using the pipeline already fitted on that fold's training data.

    Parameters
    ----------
    estimators : list of fitted pipelines (one per CV fold)
    X          : raw (unscaled) feature matrix
    y          : binary response array
    features   : ordered list of feature names
    cv         : CV splitter used during fitting

    Returns
    -------
    imps_cv      : list of ndarray (n_repeats, n_features), one per fold —
                   raw per-repeat importances for downstream analysis
    imps_stats_df : DataFrame with columns [Feature, Mean, Std] sorted by Mean
                    desc — Mean and Std computed across folds then repeats
    """
    print("\n━━━  Permutation Importance (per cv fold, pre-fitted estimators)  ━━━")
    scorer   = make_scorer(balanced_accuracy_score)
    imps_cv = []

    for fold, ((_, te), pipe) in enumerate(zip(cv.split(X, y), estimators), 1):
        res = permutation_importance(
            pipe, X[te], y[te],
            n_repeats=30, random_state=RANDOM_STATE,
            scoring=scorer, n_jobs=-1,
        )
        imps_cv.append(res.importances.T)
        print(f"   fold {fold} done")

    imps_stats = np.vstack([imps.mean(axis=0) for imps in imps_cv])
    imps_stats_df = (pd.DataFrame({
        "Feature":    features,
        "Mean":       imps_stats.mean(axis=0),
        "Std":          imps_stats.std(axis=0),
    }).sort_values("Mean", ascending=False).reset_index(drop=True))

    print(imps_stats_df.round(4).to_string(index=False))
    return imps_cv, imps_stats_df


# %% ── D. Bootstrap Coefficient Stability ────────────────────────────────────────
def bootstrap_stability(estimator,
                        X: np.ndarray,
                        y: np.ndarray,
                        features: list,
                        B: int = 200) -> tuple:
    """
    Bootstrap coefficient stability by cloning and refitting the estimator on
    B resampled datasets, then extracting coefficients each time.

    If estimator is a Pipeline, pipeline_behead splits it into preprocessing
    body and prediction head; get_predictor unwraps any meta-estimator wrapper
    (e.g. GridSearchCV); get_coef extracts coef_ or feature_importances_.

    Parameters
    ----------
    estimator  : single unfitted sklearn estimator or Pipeline (cloned each resample)
    X          : raw feature matrix (any preprocessing lives inside estimator)
    y          : binary response array
    features   : ordered list of feature names
    B          : number of bootstrap resamples (default 200)

    Returns
    -------
    coefs_boot   : list of B ndarrays, each (1, n_features) — raw per-resample
                   coefficients, mirroring the structure of imps_cv from
                   permutation_importance_analysis
    boot_stats_df : DataFrame with columns [Feature, Mean, Std,
                    CI_low (2.5%), CI_high (97.5%), Sign consistency (%)]
                    sorted by |Mean| desc
    """
    print(f"\n━━━  Bootstrap Coefficient Stability  (B={B})  ━━━")
    rng        = np.random.default_rng(RANDOM_STATE)
    coefs_boot = []

    for _ in range(B):
        idx    = rng.choice(len(X), size=len(X), replace=True)
        fitted = clone(estimator).fit(X[idx], y[idx])

        if isinstance(fitted, Pipeline):
            _, head = pipeline_behead(fitted)
        else:
            head = fitted

        coefs_boot.append(get_coef(get_predictor(head)).reshape(1, -1))

    boot_arr = np.vstack(coefs_boot)                          # (B, n_features)
    boot_mean = boot_arr.mean(axis=0)

    boot_stats_df = (pd.DataFrame({
        "Feature":              features,
        "Mean":                 boot_mean,
        "Std":                  boot_arr.std(axis=0),
        "CI_low (2.5%)":        np.quantile(boot_arr, 0.025, axis=0),
        "CI_high (97.5%)":      np.quantile(boot_arr, 0.975, axis=0),
        "Sign consistency (%)": (100 * (np.sign(boot_arr) == np.sign(boot_mean)).mean(axis=0)).round(1),
    }).sort_values("Mean", key=lambda s: s.abs(), ascending=False)
     .reset_index(drop=True))

    print("\n  Bootstrap stability summary:")
    print(boot_stats_df.round(4).to_string(index=False))
    return coefs_boot, boot_stats_df


# ── E. Comparison Dashboard ────────────────────────────────────────────────────
def comparison_dashboard(stats_dict: dict) -> None:
    """
    Bar chart comparing normalised feature importances across methods.

    Parameters
    ----------
    stats_dict : dict mapping method name -> DataFrame [Feature, Mean, ...]
        Each DataFrame must have 'Feature' and 'Mean' columns (as returned by
        shap_analysis, permutation_importance_analysis, bootstrap_stability).
        The dictionary key is used as the legend label.
    """
    def norm(s):
        s = s.abs()
        return s / s.sum() if s.sum() > 0 else s

    features = next(iter(stats_dict.values()))["Feature"].tolist()

    all_imp = pd.DataFrame({
        label: norm(df.set_index("Feature")["Mean"]).reindex(features)
        for label, df in stats_dict.items()
    })

    rank_order = all_imp.mean(axis=1).sort_values(ascending=False).index
    all_imp = all_imp.loc[rank_order]

    fig, ax = plt.subplots(figsize=(max(11, 0.6 * len(features)), 5))
    all_imp.plot.bar(ax=ax, width=0.72, edgecolor="white")
    ax.set_title("Feature Importance — Method Comparison", fontsize=12)
    ax.set_ylabel("Normalised importance")
    ax.set_xticklabels(all_imp.index, rotation=30, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax



# %%
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
from config import data, config
from sklearn.impute import SimpleImputer

# %% Input data
input = data[config['clinical_vars']].copy()
input.Catatonie.sum()


    
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(input)
Xdf = pd.DataFrame(X_imp, columns=config['clinical_vars'])
y = data['response']
y        = np.asarray(y)
features = list(Xdf.columns)
X        = Xdf.values
# %% Fit
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — matches the brief exactly
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 8
PALETTE      = "coolwarm"
C_GRID       = 10.0 ** np.arange(-3, 1)                       # [1e-3, 1e-2, 1e-1, 1]
cv_test      = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
cv_val       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = make_pipeline(
        preprocessing.StandardScaler(),
        GridSearchCV(
            lm.LogisticRegression(
                fit_intercept=False,
                class_weight="balanced",
                max_iter=5000,
                solver="saga",
            ),
            param_grid={"C": C_GRID},
            cv=cv_val,
            n_jobs=5,
            scoring="accuracy",
            refit=True,
        ),
    )

cv_results = cross_validate(
    model, X, y, cv=cv_test,
    scoring=["balanced_accuracy", "roc_auc"],
    return_train_score=True, return_estimator=True, n_jobs=5,
)
# DEBUG
estimators = cv_results['estimator']
cv = cv_test

# %% Analysis
metrics                          = plot_classification_report(X, y, cv_results['estimator'], cv_test)

# SHAP analysis.
shapvals_cv, X_trn_cv, shap_stats_df = shap_analysis(cv_results['estimator'], X, y, features, cv_test)
plot_shap_analysis(shapvals_cv, X_trn_cv, features, cv_test, X, y)
shapvals_cv_df = pd.DataFrame(np.vstack([np.abs(imps).mean(axis=0) for imps in shapvals_cv]), columns=features)
shapvals_cv_df   = shapvals_cv_df[shapvals_cv_df.mean(axis=0).sort_values(ascending=False).index]  # largest → top
#ax = sns.boxplot(data=shapvals_cv_df, orient="h", palette="Reds_r")
ax = sns.barplot(data=shapvals_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("SHAP importance averaged within cv folds")
ax.set_xlabel("Mean absolute SHAP value")
plt.tight_layout()
plt.savefig("reports/importance_SHAP.png", dpi=150, bbox_inches="tight")
plt.show()


# Permutation importance is more computationally efficient, so we run it before bootstrap stability (which is more intensive), and we can optionally run it without bootstrap stability if desired.
perm_imps_cv, perm_stats_df = permutation_importance_analysis(cv_results['estimator'], X, y, features, cv_test)

#perm_imps_cv_df = pd.DataFrame(np.vstack(perm_imps_cv), columns=features)
perm_imps_cv_df = pd.DataFrame(np.vstack([imps.mean(axis=0) for imps in perm_imps_cv]), columns=features)
perm_imps_cv_df   = perm_imps_cv_df[perm_imps_cv_df.mean(axis=0).sort_values(ascending=False).index]  # largest → top
#ax = sns.boxplot(data=perm_imps_cv_df, orient="h", palette="Reds_r")
ax = sns.barplot(data=perm_imps_cv_df, orient="h", palette="Reds_r", capsize=.1, errorbar="se")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Permutation importance averaged within cv folds")
ax.set_xlabel("Mean drop in balanced accuracy when permuted")
plt.tight_layout()
plt.savefig("reports/importance_permutation.png", dpi=150, bbox_inches="tight")
plt.show()

# Bootstrap stability is more computationally intensive, so we run it last and optionally
coefs_boot, boot_stats_df = bootstrap_stability(model, X, y, features)

boot_df   = pd.DataFrame(np.vstack(coefs_boot), columns=features)
boot_df   = boot_df[boot_df.mean().sort_values(ascending=False).index]  # largest → top
ax = sns.boxplot(data=boot_df, orient="h", palette="vlag_r")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_title(f"Bootstrap Coefficient Stability  (B={len(coefs_boot)} resamples)")
ax.set_xlabel("Bootstrap coefficient (standardised)")
plt.tight_layout()
plt.savefig("reports/importance_bootstrap.png", dpi=150, bbox_inches="tight")
plt.show()


# Comparison dashboard
fig, ax = comparison_dashboard({
    "SHAP":             shap_stats_df,
    "Permutation":      perm_stats_df,
    "Bootstrap |coef|": boot_stats_df,
})
fig.savefig("reports/importance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# %%
