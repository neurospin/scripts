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
# %% Repeated Cross-Validation
from ml_utils import make_models

nseeds = 100
rep_cv = list()
from ml_utils import make_models
estimator = make_models(cv_val=cv_val, scoring='accuracy')['model-lrl2']

for seed in range(nseeds):
    cv_= cross_validate(estimator, X, y,
                        cv=StratifiedKFold(n_splits=config['n_splits_test'], shuffle=True, random_state=seed),
                        scoring=config["metrics"],
                        return_train_score=True, n_jobs=5)
    rep_cv.append([seed, cv_['test_roc_auc'].mean(), cv_['test_balanced_accuracy'].mean()])


rep_cv = pd.DataFrame(rep_cv, columns=['seed', 'test_roc_auc', 'test_balanced_accuracy'])
rep_cv.sort_values('test_roc_auc', ascending=False, inplace=True)
print(rep_cv)
rep_cv.to_csv(config['output_models']+"repeated_cv.csv", index=False)


