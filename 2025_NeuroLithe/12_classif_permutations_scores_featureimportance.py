from ml_utils import stack_features_dicts, features_statistics, features_statistics_pvalues
from statsmodels.stats.multitest import multipletests
from ml_utils import mean_sd_tval_pval_ci
from ml_utils import dict_to_frame
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from joblib import Memory
from ml_utils import ClassificationScorer
from ml_utils import run_parallel, fit_predict
from ml_utils import dict_cartesian_product, permutation
from ml_utils import make_models
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from config import *
pd.set_option('display.width', 1000)


################################################################################
# %%


config['n_splits_val'] = 5
config['n_splits_test'] = 5
config['n_jobs_grid_search'] = 5
config['n_jobs_grid_search'] = 5

config['output_predictions_scores_feature-importance'] = config['output_models'] + \
    'predictions_scores_feature-importance.xlsx'
config['cachedir'] = './cachedir'


cv_val = StratifiedKFold(
    n_splits=config['n_splits_val'], shuffle=True, random_state=42)
# cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=1) #std auc = 20%
# cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=22) #std auc=10%
cv_test = StratifiedKFold(
    # std auc=10%
    n_splits=config['n_splits_test'], shuffle=True, random_state=8)

################################################################################
# %% Input data
X = data[config['clinical_vars']].copy()
# X = data[['DSM_TDAH', 'TEMPSA-C', 'PQ16-A']].copy()

y = data['response']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)


################################################################################
# %% Configure models, CV and permutation scheme


# cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=1)

models = make_models(cv_val=cv_val, scoring='accuracy', n_jobs_grid_search=1)
models = {k: models[k] for k in ['model-lrl2cv']}

permutation_seed = np.arange(1000)
# permutation_seed = np.arange(10)
# permutation_seed = [0]

# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i' % fold):
                   (X, permutation(y, perm), train_index, test_index)
                   for perm in permutation_seed
                   for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

models_cv = dict_cartesian_product(models, cv_test_dict_Xy)


################################################################################
# %% Fit models

memory = Memory(config['cachedir'], verbose=0)
# To FIX disable cahe for the moment
# fit_predict_cached = memory.cache(fit_predict)
fit_predict_cached = fit_predict
res_cv = run_parallel(fit_predict_cached, models_cv, verbose=50, n_jobs=10)
# res_cv = run_sequential(fit_predict, models_cv, verbose=50)


################################################################################
# %% Classifications metrics

reducer = ClassificationScorer()

# Aggregate predictions in a single dataframe (one row per test sample per fold, permutation and model)
predictions_df = reducer.predictions_dict_to_frame(res_cv,
                                                   keys=['test_idx', 'y_test_pred_lab',
                                                         'y_test_pred_decision_function',
                                                         'y_test_pred_proba',
                                                         'y_test_true_lab'])

assert predictions_df.shape == (58000, 8)

# Compute metrics per model, permutation and fold. (one row per fold, permutation and model)
predictions_metrics_df = reducer.prediction_metrics(
    predictions_df, groupby=['model', 'perm', 'fold'])
assert predictions_metrics_df.shape == (5000, 5)

# Compute mean and std of metrics accross folds, for each model and permutation. (one row per permutation and model)
predictions_metrics_stats_df = reducer.prediction_metrics_stats(
    predictions_metrics_df, groupby=['model', 'perm'])
assert predictions_metrics_stats_df.shape == (1000, 6)


print(predictions_metrics_stats_df[predictions_metrics_stats_df['perm'] == 'perm-000'])
"""
          model      perm balanced_accuracy             roc_auc          
                                       mean       std      mean       std
0  model-lrl2cv  perm-000          0.651429  0.028302  0.702381  0.112687
"""


predictions_metrics_stats_df_ = predictions_metrics_stats_df[[
    ('model', ''), ('perm', ''), ('balanced_accuracy', 'mean'), ('roc_auc', 'mean')]]

predictions_metrics_stats_pvalues_df = reducer.prediction_metrics_pvalues(
    predictions_metrics_stats_df_)
print(predictions_metrics_stats_pvalues_df)

"""
          model balanced_accuracy   roc_auc balanced_accuracy_pval roc_auc_pval
                             mean      mean                   mean         mean
0  model-lrl2cv          0.651429  0.702381               0.033033     0.026026
"""

from plot_utils import plot_roc


# Plot ROC curve for the original (non-permuted) data, averaged across folds, with bootstrap CI band
predictions_df_ = predictions_df[predictions_df['perm'] == 'perm-000']

aucs = list()
for fold, df in predictions_df_.groupby("fold"):
    y_pred_lab, y_pred_score, y_true_lab = df["y_test_pred_lab"], df[
        "y_test_pred_decision_function"], df["y_test_true_lab"]
    aucs.append(metrics.roc_auc_score(y_true_lab, y_pred_score))

print("Mean AUC:", np.mean(aucs), "Std AUC:", np.std(aucs))

y_pred_lab = predictions_df_["y_test_pred_lab"]
y_pred_score = predictions_df_["y_test_pred_decision_function"]
y_true_lab = predictions_df_["y_test_true_lab"]
ax = plot_roc(y_true_lab, y_pred_score, label="L2-Regularized Logistic Regression", rndseed=31, ci=True, ci_alpha=0.95)

plt.savefig("reports/roc_curve.pdf", dpi=300, bbox_inches="tight")
plt.savefig("reports/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

################################################################################
# %% Feature importance


models_features_names = {mod: config['clinical_vars'] for mod in models.keys()}


features_df = stack_features_dicts(res_cv,
                                   models_features_names=models_features_names,
                                   importances=['coefs', 'forwd', 'feature_auc'])

features_stats = features_statistics(features_df)
features_stats_pvals = features_statistics_pvalues(features_stats)


################################################################################
# %% Save results

# , mode="a", if_sheet_exists="replace") as writer:
with pd.ExcelWriter(config['output_predictions_scores_feature-importance']) as writer:
    predictions_metrics_df.to_excel(writer, sheet_name='predictions_metrics', index=True)
    predictions_metrics_stats_df[predictions_metrics_stats_df['perm'] == 'perm-000'].to_excel(writer, sheet_name='predictions_metrics_stats', index=True)
    predictions_metrics_stats_pvalues_df.to_excel(writer, sheet_name='predictions_metrics_pvalues', index=True)
    predictions_df.to_excel(writer, sheet_name='predictions', index=False)
    features_df.to_excel(writer, sheet_name='feature_importance', index=False)
    for (mod, stat), df in features_stats_pvals.items():
        sheet_name = '__'.join([mod, stat])
        print(sheet_name)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# %%
