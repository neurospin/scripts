import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc
from scipy import stats


def plot_roc(y_true_lab, y_pred_score,
             label=None,
             ci=True, ci_n_bootstrap=1000, ci_alpha=0.95,
             ax=None,
             color="#2166ac",
             title="ROC Curve",
             rndseed=42):
    """
    Plot a publication-quality ROC curve with optional bootstrap confidence interval.

    Parameters
    ----------
    y_true_lab      : array-like, true binary labels (0/1)
    y_pred_score    : array-like, predicted scores or probabilities
    label           : str, legend label (default: auto from AUC)
    ci              : bool, whether to plot bootstrap CI band (default: True)
    ci_n_bootstrap  : int, number of bootstrap iterations (default: 1000)
    ci_alpha        : float, confidence level (default: 0.95)
    ax              : matplotlib Axes, if None a new figure is created
    color           : str, color for the ROC curve
    title           : str, plot title

    Returns
    -------
    ax : matplotlib Axes
    """
    y_true  = np.asarray(y_true_lab)
    y_score = np.asarray(y_pred_score)

    # ── Main ROC ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    # ── Bootstrap CI ────────────────────────────────────────────────────────
    if ci:
        rng          = np.random.default_rng(rndseed)
        tpr_grid     = np.linspace(0, 1, 200)   # common FPR grid
        tpr_boots    = []
        auc_boots    = []
        n            = len(y_true)

        for _ in range(ci_n_bootstrap):
            idx          = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
            tpr_interp       = np.interp(tpr_grid, fpr_b, tpr_b)
            tpr_boots.append(tpr_interp)
            auc_boots.append(auc(fpr_b, tpr_b))

        tpr_boots = np.array(tpr_boots)
        alpha     = (1 - ci_alpha) / 2
        tpr_lo    = np.quantile(tpr_boots, alpha,     axis=0)
        tpr_hi    = np.quantile(tpr_boots, 1 - alpha, axis=0)
        auc_lo    = np.quantile(auc_boots, alpha)
        auc_hi    = np.quantile(auc_boots, 1 - alpha)
        ci_str    = f"  95% CI [{auc_lo:.2f}–{auc_hi:.2f}]"
    else:
        ci_str = ""

    # ── Figure ──────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Diagonal chance line
    ax.plot([0, 1], [0, 1],
            linestyle="--", linewidth=1, color="#aaaaaa",
            label="Chance (AUC = 0.50)", zorder=1)

    # CI band
    if ci:
        ax.fill_between(tpr_grid, tpr_lo, tpr_hi,
                        alpha=0.15, color=color, zorder=2)

    # ROC curve
    curve_label = label if label else f"AUC = {roc_auc:.2f}{ci_str}"
    ax.plot(fpr, tpr,
            linewidth=2, color=color,
            label=curve_label, zorder=3)

    # ── Aesthetics ──────────────────────────────────────────────────────────
    ax.set_xlabel("False Positive Rate (1 − Specificity)",
                  fontsize=11, labelpad=6)
    ax.set_ylabel("True Positive Rate (Sensitivity)",
                  fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", length=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(loc="lower right", fontsize=9,
              frameon=True, framealpha=0.9,
              edgecolor="#cccccc")

    ax.grid(True, which="major", linestyle=":", linewidth=0.5,
            color="#dddddd", zorder=0)

    plt.tight_layout()
    return ax


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    X, y = make_classification(n_samples=300, n_features=10,
                                random_state=0)
    clf    = LogisticRegression()
    scores = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]

    ax = plot_roc(y, scores, title="Logistic Regression — ROC Curve")
    plt.savefig("roc_curve.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

"""
Key design choices for publication quality:

- **Bootstrap CI band** (1000 iterations by default) gives honest uncertainty on the AUC, reported as `AUC = 0.XX  95% CI [lo–hi]` directly in the legend
- **Clean spine style** — top/right spines removed, thin remaining borders, no heavy grid
- **Square aspect ratio** — mandatory for ROC curves in journals
- **Minor ticks** at 0.1 intervals for precise reading without clutter
- **`.pdf` output** — vector format preferred by most journals; `.png` at 300 dpi as fallback
- **`ax` parameter** — makes it composable, e.g. for multi-model comparison:

```python
fig, ax = plt.subplots(figsize=(5, 5))
plot_roc(y, scores_model_a, ax=ax, color="#2166ac", label="Model A")
plot_roc(y, scores_model_b, ax=ax, color="#d6604d", label="Model B", ci=False)
plt.savefig("roc_comparison.pdf", bbox_inches="tight")
"""