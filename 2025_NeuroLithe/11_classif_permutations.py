from config import *

################################################################################
# %% 

from sklearn.model_selection import StratifiedKFold#, GridSearchCV
#from sklearn.pipeline import Pipeline
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
#import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer

config['n_splits_val'] = 5
config['n_splits_test'] = 5
config['n_jobs_grid_search'] = 5

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
# %%

from ml_utils import make_models
from ml_utils import dict_cartesian_product, permutation
from ml_utils import run_parallel, fit_predict
from ml_utils import ClassificationScorer

cv_test = StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=1)

models = make_models(cv_val=cv_val, scoring='accuracy', n_jobs_grid_search=1)
models = {k:models[k] for k in  ['model-lrl2cv']}
# models = make_models(cv_val=cv_val, scoring='roc_auc', n_jobs_grid_search=1)
# models = make_models(cv_val=cv_val, scoring='balanced_accuracy', n_jobs_grid_search=1)

permutation_seed = np.arange(1000)

# {('perm-%03i' % perm, 'fold-%i'  % fold): 1 for perm in range(0, 50, 10) for fold in range(0, 5)}
cv_test_dict_Xy = {('perm-%03i' % perm, 'fold-%i'  % fold):
    (X, permutation(y, perm), train_index, test_index)
    for perm in permutation_seed
    for fold, (train_index, test_index) in enumerate(cv_test.split(X, y))}
print(cv_test_dict_Xy.keys())

models_cv = dict_cartesian_product(models, cv_test_dict_Xy)

# Fit models

res_cv = run_parallel(fit_predict, models_cv, verbose=50, n_jobs=10)
#res_cv = run_sequential(fit_predict, models_cv, verbose=50)
# res_cv[[k for k in res_cv.keys()][0]].keys()

# Classifications metrics
reducer = ClassificationScorer()
predictions_df = reducer.predictions_dict_toframe(res_cv)
predictions_metrics_df = reducer.prediction_metrics(predictions_df)
print(predictions_metrics_df)
#predictions_metrics_df.to_csv(config['output_models'] + "classif_scores_models.csv")

predictions_metrics_pvalues_df = reducer.prediction_metrics_pvalues(predictions_metrics_df)
print(predictions_metrics_pvalues_df)


# with pd.ExcelWriter(config['output_predictions_scores_feature-importance']) as writer:#, mode="a", if_sheet_exists="replace") as writer:
#     predictions_metrics_pvalues_df.to_excel(writer, sheet_name='predictions_metrics_pvalues', index=False)
#     predictions_df.to_excel(writer, sheet_name='predictions', index=False)
#     for (mod, stat), df in stat_pval.items():
#         sheet_name = '__'.join([mod, stat])
#         # print(sheet_name)
#         df.to_excel(writer, sheet_name=sheet_name, index=False)
            
# %%
