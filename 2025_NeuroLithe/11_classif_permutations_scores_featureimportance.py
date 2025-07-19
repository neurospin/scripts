from config import *

################################################################################
# %% 

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

config['n_splits_val'] = 5
config['n_splits_test'] = 5
config['n_jobs_grid_search'] = 5
config['n_jobs_grid_search'] = 5

config['output_predictions_scores_feature-importance'] = config['output_models'] + 'predictions_scores_feature-importance.xlsx'
config['cachedir'] = './cachedir'


cv_val = StratifiedKFold(n_splits=config['n_splits_val'], shuffle=True, random_state=42)
cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=1)

################################################################################
# %% Input data
X = data[config['clinical_vars']].copy()
#X = data[['DSM_TDAH', 'TEMPSA-C', 'PQ16-A']].copy()

y = data['response']

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)


################################################################################
# %% Configure models, CV and permutation scheme

from ml_utils import make_models
from ml_utils import dict_cartesian_product, permutation
from ml_utils import run_parallel, fit_predict
from ml_utils import ClassificationScorer

cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=1)

models = make_models(cv_val=cv_val, scoring='accuracy', n_jobs_grid_search=1)
models = {k:models[k] for k in  ['model-lrl2cv']}

permutation_seed = np.arange(1000)
#permutation_seed = np.arange(10)
#permutation_seed = [0]

# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
    (X, permutation(y, perm), train_index, test_index)
    for perm in permutation_seed
    for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

models_cv = dict_cartesian_product(models, cv_test_dict_Xy)


################################################################################
# %% Fit models

from joblib import Memory
memory = Memory(config['cachedir'], verbose=0)
# To FIX disable cahe for the moment
#fit_predict_cached = memory.cache(fit_predict)
fit_predict_cached = fit_predict
res_cv = run_parallel(fit_predict_cached, models_cv, verbose=50, n_jobs=10)
#res_cv = run_sequential(fit_predict, models_cv, verbose=50)


################################################################################
# %% Classifications metrics

reducer = ClassificationScorer()
predictions_df = reducer.predictions_dict_toframe(res_cv)
predictions_metrics_df = reducer.prediction_metrics(predictions_df)
print(predictions_metrics_df)
#predictions_metrics_df.to_csv(config['output_models'] + "classif_scores_models.csv")

predictions_metrics_pvalues_df = reducer.prediction_metrics_pvalues(predictions_metrics_df)
print(predictions_metrics_pvalues_df)


################################################################################
# %% Feature importance

from ml_utils import dict_to_frame
from ml_utils import mean_sd_tval_pval_ci
from statsmodels.stats.multitest import multipletests

models_features_names = {mod:config['clinical_vars'] for mod in models.keys()}

from ml_utils import predictions_dict_toframe, features_statistics, features_statistics_pvalues


features_df = predictions_dict_toframe(res_cv,
                models_features_names=models_features_names,
                importances=['coefs', 'forwd', 'feature_auc'])

features_stats = features_statistics(features_df)
features_stats_pvals = features_statistics_pvalues(features_stats)


################################################################################
# %% Save results

with pd.ExcelWriter(config['output_predictions_scores_feature-importance']) as writer:#, mode="a", if_sheet_exists="replace") as writer:
    predictions_metrics_pvalues_df.to_excel(writer, sheet_name='predictions_metrics_pvalues', index=False)
    predictions_df.to_excel(writer, sheet_name='predictions', index=False)
    for (mod, stat), df in features_stats_pvals.items():
        sheet_name = '__'.join([mod, stat])
        # print(sheet_name)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

