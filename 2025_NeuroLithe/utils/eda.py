
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import warnings
warnings.filterwarnings("ignore")

PALETTE = "coolwarm"
_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#17becf",
]


def _pub(methods: str, results: str) -> None:
    print("\n── Publication Text ──────────────────────────────────────────────────")
    print(f"[Methods] {methods}")
    print(f"[Results] {results}")


# ══════════════════════════════════════════════════════════════════════════════
#  PART (i) — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _var_type(s: pd.Series, max_cat_unique: int = 2) -> str:
    """
    Classify a series as 'constant', 'binary', 'multicategory', or 'quantitative'.

    Float columns are always quantitative.
    Integer / object columns with <= max_cat_unique distinct values are categorical.
    """
    n = s.nunique(dropna=True)
    if n <= 1:
        return 'constant'
    if n == 2:
        return 'binary'
    if s.dtype.kind == 'f' or n > max_cat_unique:
        return 'quantitative'
    return 'multicategory'


# ── 1. Descriptive Statistics & Class Balance ──────────────────────────────
def descriptive_stats(X_df: pd.DataFrame,
                      max_cat_unique: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split columns into quantitative and categorical using _var_type, then
    return two DataFrames:
      - quant_df : describe().T + skewness + kurtosis (quantitative + constant)
      - cat_df   : long-form with columns var, level, count, prop (binary + multicategory)
    """
    types = {col: _var_type(X_df[col], max_cat_unique) for col in X_df.columns}
    quant_cols = [c for c, t in types.items() if t in ('quantitative', 'constant')]
    cat_cols   = [c for c, t in types.items() if t in ('binary', 'multicategory')]

    # ── Quantitative summary ───────────────────────────────────────────────
    if quant_cols:
        quant_df = X_df[quant_cols].describe().T
        quant_df["skewness"] = X_df[quant_cols].skew()
        quant_df["kurtosis"] = X_df[quant_cols].kurt()
        print("\n━━━  Descriptive Statistics — Quantitative  ━━━")
        print(quant_df.round(3).to_string())
    else:
        quant_df = pd.DataFrame()

    # ── Categorical summary ────────────────────────────────────────────────
    rows = []
    for col in cat_cols:
        s = X_df[col].dropna()
        vc = s.value_counts().sort_index()
        for level, cnt in vc.items():
            rows.append({"var": col, "level": level,
                         "count": cnt, "prop": cnt / len(s)})
    cat_df = pd.DataFrame(rows, columns=["var", "level", "count", "prop"])

    if not cat_df.empty:
        print("\n━━━  Descriptive Statistics — Categorical  ━━━")
        print(cat_df.to_string(index=False))

    # ── Publication text ───────────────────────────────────────────────────
    methods = (
        "Continuous variables were described using mean, standard deviation, median, "
        "interquartile range, skewness, and kurtosis. Categorical variables were "
        "described using absolute frequencies and proportions."
    )
    res_parts = []
    if not quant_df.empty:
        skewed = quant_df[quant_df["skewness"].abs() > 1].index.tolist()
        res_parts.append(
            f"{len(quant_cols)} continuous variable(s) were analysed"
            + (f"; {len(skewed)} showed marked skewness (|skewness| > 1): "
               + ", ".join(skewed) if skewed else "")
            + "."
        )
    if not cat_df.empty:
        cat_summaries = []
        for var, grp in cat_df.groupby("var"):
            parts = ", ".join(
                f"level {row.level}: {row.count} ({row.prop*100:.1f}%)"
                for row in grp.itertuples()
            )
            cat_summaries.append(f"{var} ({parts})")
        res_parts.append(
            f"{len(cat_cols)} categorical variable(s): " + "; ".join(cat_summaries) + "."
        )
    _pub(methods, " ".join(res_parts))

    return quant_df, cat_df


# ── 2. Correlation Matrices ────────────────────────────────────────────────
def plot_correlations(X_df: pd.DataFrame,
                      filename: str | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pearson, spearman) correlation matrices."""
    pearson  = X_df.corr(method="pearson")
    spearman = X_df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mat, title in zip(axes,
                              [pearson, spearman],
                              ["Pearson Correlation", "Spearman Correlation"]):
        mask = np.triu(np.ones_like(mat, dtype=bool))
        sns.heatmap(mat, mask=mask, annot=True, fmt=".2f",
                    cmap=PALETTE, center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax, annot_kws={"size": 9})
        ax.set_title(title, fontsize=13, fontweight="bold")
    plt.suptitle("Pairwise Feature Correlations", fontsize=15, y=1.01)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    # ── Publication text ───────────────────────────────────────────────────
    # Compute on upper triangle (excluding diagonal)
    p_vals = pearson.values
    mask_ut = np.triu(np.ones_like(p_vals, dtype=bool), k=1)
    upper   = np.abs(p_vals[mask_ut])
    n_strong = int((upper > 0.5).sum())
    n_pairs  = len(upper)
    idx_max  = np.unravel_index(np.argmax(np.abs(p_vals - np.eye(len(p_vals)))), p_vals.shape)
    # pick the strongest off-diagonal pair
    ut_idx = np.argmax(upper)
    feat_pairs = [(pearson.index[i], pearson.columns[j])
                  for i, j in zip(*np.where(mask_ut))]
    strongest  = feat_pairs[ut_idx]
    r_max      = pearson.loc[strongest[0], strongest[1]]

    methods = (
        "Pairwise Pearson and Spearman correlation coefficients were computed between "
        "all features to assess linear and monotonic associations, respectively."
    )
    results = (
        f"Among the {n_pairs} unique feature pairs, {n_strong} showed a strong Pearson "
        f"correlation (|r| > 0.50). The strongest correlation was observed between "
        f"{strongest[0]} and {strongest[1]} (r = {r_max:.2f})."
    )
    _pub(methods, results)
    return pearson, spearman


# ── 2b. Clustered Correlation Heatmap ─────────────────────────────────────
def plot_correlation_clustermap(X_df: pd.DataFrame,
                                filename: str | None = None,
                                cluster_color_threshold: float | None = 0.7,
                                ) -> pd.DataFrame:
    """
    Pearson correlation heatmap with features reordered by Ward hierarchical
    clustering so that highly correlated features appear adjacent.
    Uses seaborn.clustermap which draws the dendrogram alongside the heatmap.
    Returns the Pearson correlation matrix reindexed in clustering order.

    cluster_color_threshold : float or None
        Controls cluster coloring of dendrogram branches and tick labels.
        The Ward linkage tree is cut at  ``threshold × max_merge_height``,
        and every feature below the same sub-tree gets the same color from
        ``_CLUSTER_COLORS``.  This is also the threshold passed to scipy's
        ``hierarchy.dendrogram`` so branch colors match label colors exactly.
        Use 0.7 (default, scipy convention) to color ~30 % of the top of the
        tree with a single grey and split the lower clusters by color.
        Use a smaller value (e.g. 0.4, equivalent to |ρ| > 0.6 in correlation
        space) for finer cluster coloring.
        Pass ``None`` to disable all cluster coloring (monochrome dendrogram,
        black tick labels).
    """
    corr = X_df.corr(method="pearson")
    n = len(corr)
    cell_px = 72        # target cell size in points ≈ inches × dpi
    cell_in = cell_px / 72
    heatmap_in = n * cell_in
    dendro_in  = max(0.8, heatmap_in * 0.08)   # very narrow dendrogram
    fig_size   = heatmap_in + dendro_in + 1.2   # +margin for labels/colorbar
    font_size  = max(7, min(14, int(cell_px * 0.30)))

    label_size = max(9, min(16, int(cell_px * 0.22)))

    if cluster_color_threshold is not None:
        hierarchy.set_link_color_palette(_CLUSTER_COLORS)
    cg = sns.clustermap(
        corr,
        method="ward",
        metric="euclidean",
        cmap=PALETTE,
        center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": font_size, "weight": "bold"},
        linewidths=0.5,
        figsize=(fig_size, fig_size),
        dendrogram_ratio=dendro_in / fig_size,
        cbar_pos=(1.02, 0.4, 0.025, 0.3),
    )
    hierarchy.set_link_color_palette(None)

    # Thicken dendrogram branches
    for ax_dend in (cg.ax_col_dendrogram, cg.ax_row_dendrogram):
        for line in ax_dend.get_lines():
            line.set_linewidth(2.0)

    cg.ax_heatmap.set_xticklabels(
        cg.ax_heatmap.get_xticklabels(),
        rotation=45, ha="right", fontsize=label_size,
    )
    cg.ax_heatmap.set_yticklabels(
        cg.ax_heatmap.get_yticklabels(),
        rotation=0, fontsize=label_size,
    )

    if cluster_color_threshold is not None:
        lkg = cg.dendrogram_col.linkage
        ct = cluster_color_threshold * lkg[:, 2].max()
        cluster_ids = hierarchy.fcluster(lkg, t=ct, criterion="distance")
        feat_to_cid = {f: cluster_ids[i] for i, f in enumerate(corr.columns)}
        for lbl in list(cg.ax_heatmap.get_xticklabels()) + list(cg.ax_heatmap.get_yticklabels()):
            cid = feat_to_cid.get(lbl.get_text(), 1)
            lbl.set_color(_CLUSTER_COLORS[(cid - 1) % len(_CLUSTER_COLORS)])
    cg.figure.suptitle(
        "Pearson Correlation — Features Ordered by Ward Clustering",
        fontsize=13, fontweight="bold", y=1.02,
    )
    if filename is not None:
        cg.figure.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    # ── Publication text ───────────────────────────────────────────────────
    reordered = [corr.index[i] for i in cg.dendrogram_col.reordered_ind]
    methods = (
        "Pearson correlation coefficients were computed and features were reordered "
        "by Ward hierarchical clustering on the distance matrix (1 − r) to reveal "
        "groups of co-varying variables."
    )
    results = (
        f"Hierarchical clustering of {n} features revealed a structured correlation "
        f"pattern. The clustering-derived feature order was: {', '.join(reordered)}."
    )
    _pub(methods, results)
    return corr.loc[reordered, reordered]


# ── 3. Variance Inflation Factors ─────────────────────────────────────────
def plot_variance_inflation_factors(X_df: pd.DataFrame, filename: str | None = None) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer

    features = list(X_df.columns)
    X = X_df.values
    if X_df.isnull().values.any():
        X = SimpleImputer(strategy='median').fit_transform(X)

    vifs = []
    for j in range(X.shape[1]):
        X_others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(X_others, X[:, j]).score(X_others, X[:, j])
        vifs.append(1 / (1 - r2) if r2 < 1 else np.inf)
    vif_df = (pd.DataFrame({"Feature": features, "VIF": vifs})
              .sort_values("VIF", ascending=False).reset_index(drop=True))
    print("\n━━━  Variance Inflation Factors  ━━━")
    print(vif_df.round(2).to_string(index=False))
    print("  Threshold guide:  VIF > 10 -> severe  |  5-10 -> moderate")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(features))))
    colors = ["crimson" if v > 10 else "orange" if v > 5 else "steelblue"
              for v in vif_df["VIF"]]
    ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors, edgecolor="white")
    ax.axvline(5,  color="orange",  linestyle="--", lw=1.2, label="VIF = 5")
    ax.axvline(10, color="crimson", linestyle="--", lw=1.2, label="VIF = 10")
    ax.set_xlabel("VIF"); ax.set_title("Variance Inflation Factors", fontweight="bold")
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    # ── Publication text ───────────────────────────────────────────────────
    severe   = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()
    moderate = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]["Feature"].tolist()
    methods = (
        "Variance inflation factors (VIF) were computed for each predictor by "
        "regressing it on the remaining predictors. VIF > 10 indicates severe "
        "multicollinearity; VIF 5–10 indicates moderate multicollinearity."
    )
    if severe:
        res = (f"Severe multicollinearity (VIF > 10) was detected for: "
               f"{', '.join(severe)}. ")
    else:
        res = "No feature showed severe multicollinearity (VIF > 10). "
    if moderate:
        res += f"Moderate multicollinearity (VIF 5–10) was observed for: {', '.join(moderate)}."
    else:
        res += "No feature showed moderate multicollinearity (VIF 5–10)."
    _pub(methods, res)

    return vif_df


# ── 4. Hierarchical Feature Clustering ────────────────────────────────────
def plot_feature_dendrogram(X_df: pd.DataFrame,
                            filename: str | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns [feature, cluster] (Ward cut at distance 0.4)."""
    features = list(X_df.columns)
    corr = X_df.corr().abs()
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0)
    linkage = hierarchy.ward(squareform(dist.values))

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(features)), 5))
    hierarchy.set_link_color_palette(_CLUSTER_COLORS)
    ddata = hierarchy.dendrogram(linkage, labels=features,
                                 ax=ax, color_threshold=0.4,
                                 above_threshold_color="#bbbbbb")
    hierarchy.set_link_color_palette(None)

    # Thicken branches
    for line in ax.get_lines():
        line.set_linewidth(2.0)

    # Color x-tick labels to match their cluster
    cluster_ids = hierarchy.fcluster(linkage, t=0.4, criterion="distance")
    for lbl, leaf_idx in zip(ax.get_xticklabels(), ddata["leaves"]):
        cid = cluster_ids[leaf_idx]
        lbl.set_color(_CLUSTER_COLORS[(cid - 1) % len(_CLUSTER_COLORS)])
        lbl.set_fontweight("bold")

    ax.axhline(0.4, color="crimson", linestyle="--", lw=1.2,
               label="Cut  (|rho| > 0.6)")
    ax.set_title("Feature Dendrogram  —  Ward linkage on |1 - rho|",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance"); ax.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    # ── Publication text ───────────────────────────────────────────────────
    # Count clusters at the cut threshold (0.4)
    cluster_ids = hierarchy.fcluster(linkage, t=0.4, criterion='distance')
    n_clusters  = len(np.unique(cluster_ids))
    cluster_df  = (pd.DataFrame({"feature": features, "cluster": cluster_ids})
                   .sort_values("cluster").reset_index(drop=True))
    methods = (
        "Features were hierarchically clustered using Ward linkage applied to the "
        "distance matrix 1 − |ρ|, where ρ is the Pearson correlation coefficient. "
        "Clusters were identified by cutting the dendrogram at distance 0.4 "
        "(corresponding to |ρ| > 0.6)."
    )
    results = (
        f"Hierarchical clustering of {len(features)} features yielded {n_clusters} "
        f"cluster(s) at the distance threshold of 0.4 (|ρ| > 0.6)."
    )
    _pub(methods, results)
    return cluster_df


# ── 5. PCA Optimal Number of Components ───────────────────────────────────
def plot_pca_components(X_df: pd.DataFrame,
                        var_thresholds: list[float] = [0.80, 0.90, 0.95],
                        filename: str | None = None,
                        ) -> tuple[pd.DataFrame, int, dict]:
    """
    Fit full PCA on standardised data, plot the scree, and recommend the
    optimal number of components via two complementary criteria:
      - Cumulative variance thresholds (80 / 90 / 95 %)
      - Elbow detection: largest second-difference of explained-variance ratios
        (point where marginal gain drops most sharply)
    Returns:
      scree_df      : DataFrame [component, evr, cumulative_evr]
      elbow_idx     : int — elbow-based recommendation (1-based)
      thresh_results: dict {threshold: n_components}
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X = X_df.values
    if X_df.isnull().values.any():
        X = SimpleImputer(strategy='median').fit_transform(X)

    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA().fit(X_sc)
    evr  = pca.explained_variance_ratio_
    cumev = np.cumsum(evr)
    comps = np.arange(1, len(evr) + 1)

    elbow_idx     = int(np.argmax(np.diff(np.diff(cumev))) + 2)   # 1-based
    thresh_results = {t: int(np.searchsorted(cumev, t) + 1) for t in var_thresholds}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(comps, evr * 100, color="steelblue", edgecolor="white", label="Individual")
    axes[0].plot(comps, cumev * 100, "o-", color="crimson", lw=2, ms=5, label="Cumulative")
    axes[0].axvline(elbow_idx, color="darkorange", linestyle="--", lw=1.5,
                    label=f"Elbow  (n={elbow_idx})")
    for t, n in thresh_results.items():
        axes[0].axhline(t * 100, color="grey", linestyle=":", lw=1,
                        label=f"{int(t*100)}%  →  n={n}")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree Plot", fontweight="bold")
    axes[0].legend(fontsize=8)

    d2 = np.diff(np.diff(cumev))
    axes[1].bar(comps[1:-1], d2, color="mediumpurple", edgecolor="white")
    axes[1].axvline(elbow_idx, color="darkorange", linestyle="--", lw=1.5,
                    label=f"Max curvature  (n={elbow_idx})")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Δ² cumulative variance")
    axes[1].set_title("Elbow Detection  —  Second Difference", fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.suptitle("PCA — Optimal Number of Components", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    print("\n━━━  PCA — Component Selection  ━━━")
    for t, n in thresh_results.items():
        print(f"  {int(t*100)}% variance  →  {n} component(s)  "
              f"(cumulative = {cumev[n-1]*100:.1f} %)")
    print(f"  Elbow criterion      →  {elbow_idx} component(s)  "
          f"(cumulative = {cumev[elbow_idx-1]*100:.1f} %)")

    # ── Publication text ───────────────────────────────────────────────────
    thresh_str = "; ".join(
        f"{int(t*100)}% of variance explained by {n} component(s) "
        f"(cumulative = {cumev[n-1]*100:.1f}%)"
        for t, n in thresh_results.items()
    )
    methods = (
        "Principal component analysis (PCA) was applied to standardised features. "
        "The optimal number of components was determined using cumulative explained "
        "variance thresholds (80%, 90%, 95%) and an elbow criterion defined as the "
        "point of maximum curvature in the cumulative variance curve (largest second "
        "difference)."
    )
    results = (
        f"PCA was performed on {X_df.shape[1]} features (n = {X_df.shape[0]}). "
        f"{thresh_str}. "
        f"The elbow criterion suggested {elbow_idx} component(s) "
        f"(cumulative variance = {cumev[elbow_idx-1]*100:.1f}%)."
    )
    _pub(methods, results)

    scree_df = pd.DataFrame({
        "component":      comps,
        "evr":            evr,
        "cumulative_evr": cumev,
    })
    return scree_df, elbow_idx, thresh_results


# ── 6. Feature–Response Relationships ─────────────────────────────────────
def plot_feature_response(X_df: pd.DataFrame, y,
                          filename: str | None = None) -> pd.DataFrame:
    """Returns a DataFrame [feature, r_pb, p_mw] sorted by p_mw."""
    y = pd.Series(np.asarray(y), index=X_df.index, name="lithium_response")
    features = list(X_df.columns)
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.atleast_1d(axes).flatten()

    plot_df = X_df.copy()
    plot_df["__y__"] = y.values

    assoc = []
    for i, feat in enumerate(features):
        ax = axes[i]
        sns.violinplot(data=plot_df, x="__y__", y=feat,
                       palette="Set2", inner="box", ax=ax)

        r_pb, _ = stats.pointbiserialr(y.values, X_df[feat].values)
        g0 = X_df.loc[y.values == 0, feat]
        g1 = X_df.loc[y.values == 1, feat]
        _, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        assoc.append({"feature": feat, "r_pb": r_pb, "p_mw": p_mw})

        ax.set_title(f"{feat}\nrho_pb={r_pb:.2f}  |  MW p={p_mw:.3f}", fontsize=10)
        ax.set_xlabel("Lithium response  (0 / 1)")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature -> Lithium Response Relationships",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✔  Saved {filename}")
    plt.show()

    # ── Publication text ───────────────────────────────────────────────────
    assoc_df  = pd.DataFrame(assoc).sort_values("p_mw")
    sig       = assoc_df[assoc_df["p_mw"] < 0.05]
    trend     = assoc_df[(assoc_df["p_mw"] >= 0.05) & (assoc_df["p_mw"] < 0.10)]
    methods = (
        "The association between each feature and lithium response was evaluated "
        "using the point-biserial correlation coefficient (ρ_pb) and the "
        "Mann-Whitney U test (two-sided). No correction for multiple comparisons "
        "was applied at this exploratory stage."
    )
    if sig.empty:
        res = "No feature reached statistical significance (p < 0.05 by Mann-Whitney U test)."
    else:
        sig_str = "; ".join(
            f"{row.feature} (ρ_pb = {row.r_pb:.2f}, p = {row.p_mw:.3f})"
            for row in sig.itertuples()
        )
        res = f"The following feature(s) were significantly associated with lithium response: {sig_str}."
    if not trend.empty:
        trend_str = "; ".join(
            f"{row.feature} (p = {row.p_mw:.3f})" for row in trend.itertuples()
        )
        res += f" Trend-level associations (0.05 ≤ p < 0.10): {trend_str}."
    _pub(methods, res)
    return assoc_df

# %% Demo

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 120

    # Synthetic neuroimaging + clinical features
    X = pd.DataFrame({
        "age":           rng.normal(45, 12, n),
        "bmi":           rng.normal(24, 4, n),
        "gm_volume":     rng.normal(650, 50, n),
        "wm_volume":     rng.normal(450, 40, n),
        "hippo_l":       rng.normal(3.5, 0.4, n),
        "hippo_r":       rng.normal(3.5, 0.4, n),
        "amygdala":      rng.normal(1.8, 0.3, n),
        "fa_cingulum":   rng.beta(5, 2, n),
        "sex":           rng.integers(0, 2, n),
        "diagnosis":     rng.choice([0, 1, 2], n),
    })
    # Correlated pairs: hippo_l ~ hippo_r, gm_volume ~ wm_volume
    X["hippo_r"] = X["hippo_l"] + rng.normal(0, 0.1, n)
    X["wm_volume"] = 0.6 * X["gm_volume"] + rng.normal(0, 30, n)

    y = (0.3 * X["hippo_l"] - 0.2 * X["age"] / 50
         + rng.normal(0, 0.5, n)) > 0
    y = y.astype(int)

    quant_features = [c for c in X.columns if c not in ("sex", "diagnosis")]

    print("=" * 70)
    print("1. Descriptive statistics")
    print("=" * 70)
    quant_df, cat_df = descriptive_stats(X)

    print("\n" + "=" * 70)
    print("2. Correlation matrices")
    print("=" * 70)
    pearson, spearman = plot_correlations(X[quant_features])

    print("\n" + "=" * 70)
    print("2b. Clustered correlation heatmap")
    print("=" * 70)
    corr_reordered = plot_correlation_clustermap(X[quant_features])

    print("\n" + "=" * 70)
    print("3. Variance inflation factors")
    print("=" * 70)
    vif_df = plot_variance_inflation_factors(X[quant_features])

    print("\n" + "=" * 70)
    print("4. Feature dendrogram")
    print("=" * 70)
    cluster_df = plot_feature_dendrogram(X[quant_features])

    print("\n" + "=" * 70)
    print("5. PCA optimal components")
    print("=" * 70)
    scree_df, elbow_idx, thresh_results = plot_pca_components(X[quant_features])

    print("\n" + "=" * 70)
    print("6. Feature–response relationships")
    print("=" * 70)
    assoc_df = plot_feature_response(X[quant_features], y)



