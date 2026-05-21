"""
Exploratory Data Analysis (EDA) — Lithium Response (NeuroLithe)
===============================================================
Runs the full EDA pipeline on the clinical feature matrix and saves
figures and a summary Excel workbook to reports/.

Steps
-----
1. Class balance — prints lithium-response group sizes.
2. Descriptive statistics — mean/SD/skewness for continuous variables;
   frequencies/proportions for binary/categorical variables.
3. Clustered correlation heatmap — hierarchical clustering of features
   based on absolute Pearson correlation; saved to eda_correlation_clustermap.png.
4. Pearson correlation matrix — plotted in the cluster order from step 3;
   saved to eda_correlation.png.
5. Variance Inflation Factors (VIF) — multicollinearity diagnosis;
   saved to eda_vif.png.
6. Feature dendrogram — Ward linkage on the VIF-standardised matrix;
   saved to eda_dendrogram.png.
7. PCA scree plot — explained variance vs. number of components with
   elbow detection; saved to eda_pca_components.png.
8. Feature–response associations — per-feature comparison between
   responders and non-responders; saved to eda_feature_response.png.

Outputs
-------
reports/eda_results.xlsx   — one sheet per analysis step, plus
                              ready-to-paste Methods / Results text.
reports/eda_*.png          — individual figures (one per step).

Inputs (from config.py)
-----------------------
data                    : pd.DataFrame — full patient dataset
config['clinical_vars'] : list[str]    — clinical feature names to analyse
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.eda import (descriptive_stats, plot_correlation,
                       plot_correlation_clustermap,
                       plot_variance_inflation_factors,
                       plot_feature_dendrogram, plot_pca_components,
                       plot_feature_response)

from config import data, config

# %% Input data
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
pearson, spearman, pub_corr            = plot_correlation(X_df[corr_reordered.index], spearman=False, filename="reports/eda_correlation.png")
vif_df, pub_vif                        = plot_variance_inflation_factors(X_df, filename="reports/eda_vif.png")
cluster_df, pub_dend                   = plot_feature_dendrogram(X_df, filename="reports/eda_dendrogram.png")
scree_df, elbow_idx, thresh_results, pub_pca = plot_pca_components(X_df, filename="reports/eda_pca_components.png")
assoc_df, pub_resp                     = plot_feature_response(X_df, y, filename="reports/eda_feature_response.png")

# ── Save all results to Excel ──────────────────────────────────────────────
pub_df = pd.DataFrame([
    {"function": "descriptive_stats",               **pub_desc},
    {"function": "plot_correlation_clustermap",     **pub_clust},
    {"function": "plot_correlation",                **pub_corr},
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
