
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

quant_df, cat_df, pub_desc             = descriptive_stats(X_df, max_cat_unique=2)
corr_reordered, pub_clust              = plot_correlation_clustermap(X_df, cluster_color_threshold=None, filename="reports/eda_correlation_clustermap.png")
pearson, spearman, pub_corr            = plot_correlations(X_df[corr_reordered.index], spearman=False, filename="reports/eda_correlation.png")
vif_df, pub_vif                        = plot_variance_inflation_factors(X_df, filename="reports/eda_vif.png")
cluster_df, pub_dend                   = plot_feature_dendrogram(X_df, filename="reports/eda_dendrogram.png")
scree_df, elbow_idx, thresh_results, pub_pca = plot_pca_components(X_df, filename="reports/eda_pca_components.png")
assoc_df, pub_resp                     = plot_feature_response(X_df, y, filename="reports/eda_feature_response.png")

# ── Save all results to Excel ──────────────────────────────────────────────
pub_df = pd.DataFrame([
    {"function": "descriptive_stats",               **pub_desc},
    {"function": "plot_correlation_clustermap",     **pub_clust},
    {"function": "plot_correlations",               **pub_corr},
    {"function": "plot_variance_inflation_factors", **pub_vif},
    {"function": "plot_feature_dendrogram",         **pub_dend},
    {"function": "plot_pca_components",             **pub_pca},
    {"function": "plot_feature_response",           **pub_resp},
])

excel_path = "reports/eda_results.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    quant_df.to_excel(writer,       sheet_name="desc_quantitative")
    cat_df.to_excel(writer,         sheet_name="desc_categorical",  index=False)
    corr_reordered.to_excel(writer, sheet_name="corr_clustermap")
    pearson.to_excel(writer,        sheet_name="corr_pearson")
    vif_df.to_excel(writer,         sheet_name="vif",               index=False)
    cluster_df.to_excel(writer,     sheet_name="feature_clusters",  index=False)
    scree_df.to_excel(writer,       sheet_name="pca_scree",         index=False)
    assoc_df.to_excel(writer,       sheet_name="feature_response",  index=False)
    pub_df.to_excel(writer,         sheet_name="publication_text",  index=False)
print(f"\n✔  Saved {excel_path}")

