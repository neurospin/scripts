"""
Clustering on CTQ / CDI scores and association with CAARMS outcomes
====================================================================

Input features (clustering)
----------------------------
  ctq_total  : Childhood Trauma Questionnaire — total score
  cdi_total  : Children's Depression Inventory — total score

Clustering
----------
  KMeans (k=2) on standardised inputs.
  Cluster stability is assessed by silhouette score.

Output variables (association test)
-------------------------------------
  caarms_utc : CAARMS — unusual thought content
  caarms_nbi : CAARMS — non-bizarre ideas
  caarms_pa  : CAARMS — perceptual abnormalities
  caarms_ds  : CAARMS — disorganised speech

Statistical test
----------------
  pairwise_stats (utils/stats_pairwise.py) between cluster label (binary)
  and each CAARMS score (quantitative — Welch t-test).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utils.stats_pairwise import pairwise_stats

# ==============================================================================
# %% Configuration
# ==============================================================================
DATA_FILE   = "data/NeuroPEP_AR.xlsx"
SHEET       = "Dataset"
RANDOM_STATE = 0

INPUT_VARS  = ["ctq_total", "cdi_total"]
OUTPUT_VARS = ["caarms_utc", "caarms_nbi", "caarms_pa", "caarms_ds"]

# ==============================================================================
# %% Load data
# ==============================================================================
data = pd.read_excel(DATA_FILE, sheet_name=SHEET)
print(f"Loaded: {data.shape[0]} participants, {data.shape[1]} variables")

# ==============================================================================
# %% Clustering on ctq_total / cdi_total
# ==============================================================================
cluster_df = data[["ID"] + INPUT_VARS].dropna()
print(f"\nClustering on {len(cluster_df)} participants (complete cases on inputs)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[INPUT_VARS])

kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)
labels = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, labels)
print(f"Silhouette score (k=2): {sil:.3f}")
print(f"Cluster sizes: {pd.Series(labels).value_counts().to_dict()}")

cluster_df = cluster_df.copy()
cluster_df["cluster"] = labels

# ==============================================================================
# %% Descriptive: cluster means on input features
# ==============================================================================
cluster_descriptive_df = (
    cluster_df.groupby("cluster")[INPUT_VARS]
    .agg(["count", "mean", "std"])
    .rename(columns={"std": "sd"})
    .round(2)
)
print("\nCluster descriptives (input features):")
print(cluster_descriptive_df)

# ==============================================================================
# %% Scatter plot of clusters
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
for cl, grp in cluster_df.groupby("cluster"):
    ax.scatter(grp["ctq_total"], grp["cdi_total"],
               label=f"Cluster {cl} (n={len(grp)})", alpha=0.7, s=50)
ax.set_xlabel("CTQ total")
ax.set_ylabel("CDI total")
ax.set_title("K-means clustering (k=2)")
ax.legend()
plt.tight_layout()
plt.savefig("reports/clustering_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ==============================================================================
# %% Association: cluster × CAARMS (Welch t-test via pairwise_stats)
# ==============================================================================
# Merge cluster labels with CAARMS scores on ID
df_assoc = cluster_df[["ID", "cluster"]].merge(data[["ID"] + OUTPUT_VARS], on="ID", how="inner")
print(f"\nAssociation dataset: {len(df_assoc)} participants")

# CAARMS scores (0-6 ordinal, ≤6 unique values) are treated as quantitative
# by setting max_cat_unique=5 so cluster (2 unique) → binary
# and CAARMS (7 unique) → quantitative → Welch t-test
stats_df = pairwise_stats(
    df_assoc,
    vars1=["cluster"],
    vars2=OUTPUT_VARS,
    max_cat_unique=5,
)
print("\nCluster × CAARMS association (Welch t-test):")
print(stats_df[["v1", "v2", "test", "stat", "dof", "pval", "descriptive"]].to_string(index=False))

# ==============================================================================
# %% Violin plots: CAARMS by cluster
# ==============================================================================
df_long = df_assoc.melt(id_vars="cluster", value_vars=OUTPUT_VARS,
                         var_name="CAARMS", value_name="score")
fig, axes = plt.subplots(1, len(OUTPUT_VARS), figsize=(4 * len(OUTPUT_VARS), 4), sharey=False)
for ax, var in zip(axes, OUTPUT_VARS):
    sns.violinplot(data=df_assoc, x="cluster", y=var, ax=ax,
                   palette={"0": "#4878d0", "1": "#ee854a"}, inner="box", cut=0)
    row = stats_df[stats_df["v2"] == var]
    if not row.empty:
        p = row["pval"].values[0]
        ax.set_title(f"{var}\np={p:.3f}")
    ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig("reports/clustering_caarms_violins.png", dpi=150, bbox_inches="tight")
plt.show()

# ==============================================================================
# %% Save results to Excel
# ==============================================================================
EXCEL_OUT = "reports/clustering_results.xlsx"
"""
Excel output — three sheets
─────────────────────────────────────────────────────────────────────────────
Sheet: cluster_assignments
  One row per participant. Columns: ID, ctq_total, cdi_total, cluster (0/1).

Sheet: cluster_descriptives
  Descriptive statistics of input features per cluster.
  MultiIndex columns: (variable, stat) where stat ∈ {count, mean, sd}.
  Rows: cluster 0, cluster 1.

Sheet: ttest_caarms
  One row per CAARMS subscore. Columns: v1 (cluster), v2 (CAARMS variable),
  test (welch_t), stat (t value), dof, pval, descriptive (mean±sd per group).
─────────────────────────────────────────────────────────────────────────────
"""
with pd.ExcelWriter(EXCEL_OUT) as writer:
    cluster_df.to_excel(writer, sheet_name="cluster_assignments", index=False)
    cluster_descriptive_df.to_excel(writer, sheet_name="cluster_descriptives")
    stats_df.to_excel(writer, sheet_name="ttest_caarms", index=False)
print(f"Saved: {EXCEL_OUT}")

# ==============================================================================
# %% Methods and Results text
# ==============================================================================
sizes  = cluster_df["cluster"].value_counts().sort_index()
n_tot  = len(cluster_df)

# Build per-cluster descriptive string: "mean ± sd" for each input variable
def _desc(cl):
    row = cluster_descriptive_df.loc[cl]
    parts = [f"{var}: {row[(var, 'mean')]:.1f} ± {row[(var, 'sd')]:.1f}"
             for var in INPUT_VARS]
    return "; ".join(parts)

# Significant associations
sig = stats_df[stats_df["pval"] < 0.05].copy()
sig_str = (
    ", ".join(
        f"{r.v2} (t={r.stat:.2f}, p={r.pval:.3f})"
        for _, r in sig.iterrows()
    ) or "none"
)

methods_text = f"""Methods

Participants with complete data on both input variables (CTQ total score and CDI total score)
were clustered using K-means (k=2) applied to standardised scores (zero mean, unit variance).
The number of clusters was fixed a priori to two to contrast high- versus low-trauma/depression
profiles. Cluster stability was quantified by the average silhouette coefficient.
Differences between clusters on CAARMS subscores (unusual thought content, non-bizarre ideas,
perceptual abnormalities, disorganised speech) were assessed with Welch two-sample t-tests
using the pairwise_stats routine (utils/stats_pairwise.py). Statistical significance was set
at α = 0.05 (uncorrected).
"""

results_text = f"""Results

Of the {n_tot} participants with complete CTQ and CDI data, K-means identified two clusters:
cluster 0 (n={sizes[0]}, {_desc(0)}) and cluster 1 (n={sizes[1]}, {_desc(1)}).
The average silhouette score was {sil:.2f}, indicating {'moderate' if sil >= 0.3 else 'weak'} separation.

Welch t-tests between clusters on CAARMS subscores yielded significant differences for:
{sig_str}.
Full statistics are reported in {EXCEL_OUT} (sheet: ttest_caarms).
"""

print(methods_text)
print(results_text)

# Optionally save to text file
with open("reports/methods_results.txt", "w") as fh:
    fh.write(methods_text + "\n" + results_text)
print("Saved: reports/methods_results.txt")

# %%
