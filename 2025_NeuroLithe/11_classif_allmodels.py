#from ml_utils import stack_features_dicts, features_statistics, features_statistics_pvalues
#from statsmodels.stats.multitest import multipletests
#from ml_utils import mean_sd_tval_pval_ci
#from ml_utils import dict_to_frame
#from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt
#from sklearn import metrics
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

NSPLITS_VAL = 5
NSPLITS_TEST = 5
NJOBS_GRID_SEARCH = 5

config['n_splits_val'] = 5
config['n_splits_test'] = 5
config['n_jobs_grid_search'] = 5
config['n_jobs_grid_search'] = 5

OUTPUT = "./reports/classif_models.xlsx"

cv_val = StratifiedKFold(
    n_splits=NSPLITS_VAL, shuffle=True, random_state=42)
cv_test = StratifiedKFold(
    n_splits=NSPLITS_TEST, shuffle=True, random_state=8)

################################################################################
# %% Input data
X = data[config['clinical_vars']].copy()
y = data['response']
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)


################################################################################
# %% Configure models, CV and permutation scheme

models = make_models(cv_val=cv_val, scoring='accuracy', n_jobs_grid_search=1)
# models = {k: models[k] for k in ['model-lrl2cv']}

# Permutation ? : (no permutation 0 => (perm-000))
# permutation_seed = np.arange(1000)
# permutation_seed = np.arange(10)
permutation_seed = [0]

# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i' % fold):
                   (X, permutation(y, perm), train_index, test_index)
                   for perm in permutation_seed
                   for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

models_cv = dict_cartesian_product(models, cv_test_dict_Xy)


################################################################################
# %% Fit models

# memory = Memory(config['cachedir'], verbose=0)
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

assert predictions_df.shape == (348, 8)

# Compute metrics per model, permutation and fold. (one row per fold, permutation and model)
predictions_metrics_df = reducer.prediction_metrics(
    predictions_df, groupby=['model', 'perm', 'fold'])
assert predictions_metrics_df.shape == (30, 5)

# Compute mean and std of metrics accross folds, for each model and permutation. (one row per permutation and model)
predictions_metrics_stats_df = reducer.prediction_metrics_stats(
    predictions_metrics_df, groupby=['model', 'perm'])
assert predictions_metrics_stats_df.shape == (6, 6)

predictions_metrics_stats_df[('balanced_accuracy', 'se')] = predictions_metrics_stats_df[('balanced_accuracy', 'std')] / np.sqrt(NSPLITS_TEST)
predictions_metrics_stats_df[('roc_auc', 'se')] = predictions_metrics_stats_df[('roc_auc', 'std')] / np.sqrt(NSPLITS_TEST)

print(predictions_metrics_stats_df)


"""
           model      perm balanced_accuracy             roc_auc           balanced_accuracy   roc_auc
                                         mean       std      mean       std                se        se
0          mlp_cv  perm-000          0.505476  0.076883  0.578095  0.164806          0.034383  0.073703
1  model-forestcv  perm-000          0.560000  0.071891  0.500952  0.085691          0.032151  0.038322
2      model-gbcv  perm-000          0.588095  0.173238  0.590000  0.163632          0.077474  0.073179
3  model-lrenetcv  perm-000          0.500000  0.000000  0.500000  0.000000          0.000000  0.000000
4    model-lrl2cv  perm-000          0.651429  0.028302  0.702381  0.112687          0.012657  0.050395
5  model-svmrbfcv  perm-000          0.571429  0.079700  0.630952  0.140011          0.035643  0.062615
"""


with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    predictions_metrics_stats_df.to_excel(writer, sheet_name="metrics_stats")
    predictions_metrics_df.to_excel(writer, sheet_name="metrics_per_fold")
print(f"Saved {OUTPUT}")

# %%
