
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

import warnings
warnings.filterwarnings("ignore")

PALETTE = "coolwarm"

from utils.eda import plot_correlations, plot_correlation_clustermap, plot_variance_inflation_factors, plot_feature_dendrogram, plot_pca_components, plot_feature_response, descriptive_stats


# %% Input data
from config import data, config

X_df = data[config['clinical_vars']].copy()
X_df.Catatonie.sum()


# ── Class balance ──────────────────────────────────────────────────────
y = data['response']
counts = y.value_counts()
print("\n━━━  Lithium Response — Class Balance  ━━━")
for cls, cnt in counts.items():
    print(f"  Class {cls}: {cnt}  ({100*cnt/len(y):.1f} %)")

quant_df, cat_df                    = descriptive_stats(X_df, max_cat_unique=2)
corr_reordered                      = plot_correlation_clustermap(X_df, cluster_color_threshold=None, filename="reports/eda_correlation_clustermap.png")
pearson, spearman                   = plot_correlations(X_df[corr_reordered.index], spearman=False, filename="reports/eda_correlation.png")  # same but reordered by clustering
vif_df                              = plot_variance_inflation_factors(X_df, filename="reports/eda_vif.png")
cluster_df                          = plot_feature_dendrogram(X_df, filename="reports/eda_dendrogram.png")
scree_df, elbow_idx, thresh_results = plot_pca_components(X_df, filename="reports/eda_pca_components.png")
assoc_df                            = plot_feature_response(X_df, y, filename="reports/eda_feature_response.png")
