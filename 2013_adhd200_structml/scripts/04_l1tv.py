# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:28:34 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
from structured import LogisticRegressionL1TV
from sklearn import preprocessing
from sklearn.metrics import Scorer, recall_score 
from structured import loss_functions

# Scripts PATH
try:
    ADHD200_SCRIPTS_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
  ADHD200_SCRIPTS_BASE_PATH = os.path.join(os.environ["HOME"] , "git", "scripts", "2013_adhd200_structml")

#ADHD200_DATA_BASE_PATH = "/neurospin/adhd200"
ADHD200_DATA_BASE_PATH = "/volatile/duchesnay/data/2013_adhd_structml"
INPUT_PATH = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")

execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "data_api.py"))
import cPickle as pickle
tv_path = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "precomp", "tv_loss.pkl")

## ==========================================================================
feature = "mw"


# DATA PATH
#INPUT_PATH2 = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")

X_train , y_train = get_data(INPUT_PATH, feature, test=False)
X_train = X_train.astype(np.float)
y_train = y_train.astype(np.float)[:, np.newaxis]

#X_test , y_test = get_data(INPUT_PATH, feature, test=True)

# Scale data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# Flatten mask
mask = np.array(get_mask(INPUT_PATH) == 1, dtype=np.int)
shape = mask.shape
mask = mask.ravel()


# !! Gamma as float
l=1; gamma=1.; mu=None

## BUILD SAVE tv =============================================================
if False:
    tv = loss_functions.TotalVariation(gamma, shape, mu, mask)
    with open(tv_path, 'wb') as outfile: pickle.dump(tv, outfile, pickle.HIGHEST_PROTOCOL)

## SAVE Everything
#with open("X.npy", 'wb') as outfile: pickle.dump(tv, outfile, pickle.HIGHEST_PROTOCOL)
#np.save("X.npy", X_train)
#np.save("y.npy", y_train)


# RE-READ tv ============================================================
with open(tv_path, 'rb') as infile: tv = pickle.load(infile)

# save Ax, Ay, Az, 
clf = LogisticRegressionL1TV(l=l., gamma=gamma., shape=(5,5,5))


clf = LogisticRegressionL1TV(l=1, gamma=gamma, shape=(5,5,5))
self = clf

self._tv =  tv
self._tv.buff = np.zeros((tv.Ax.shape[1], 1))
self._tv.mu = 0.0001


self._combo = loss_functions.CombinedNesterovLossFunction(self._lr,
                                                          self._tv)
self.set_g(self._combo)
self._l1 = loss_functions.L1(l)
self.set_h(self._l1)

X = X_train
#y = np.asanyarray(y_train, dtype = 'int32')

y = np.asanyarray(y_train, dtype = X.dtype)[:, np.newaxis]

######### GO INTO ALGO
self.set_data(X, y)
clf.algorithm.g.a.lipschitz = 5029851.1238164632
#Out[69]: 5029851.1238164632
clf.algorithm.g.b.lambda_max = 88.849255330364926

self = clf.algorithm
t=None; tscale=0.95; early_stopping_mu=None
self.run(X, y)

t = 1.6051788336542154e-07
#######################

%time clf.fit(X, y)

#%time svmlin.fit(X_train, y_train)
# CPU times: user 17.58 s, sys: 1.50 s, total: 19.09 s

y_pred = clf.predict(X_test)


# Use Scorer
# scorer = Scorer(score_func=recall_score, pos_label=None, average='macro')
# scorer(estimator=svmlin, X=X_test, y=y_test)

# confusion_matrix(y_true=y_test, y_pred=y_pred, labels=None)
recall_scores = recall_score(y_true=y_test, y_pred=y_pred, pos_label=None, 
                             average=None, labels=[0,1])
bsr = recall_scores.mean()


print recall_scores, bsr

# Scaled    [ 0.61702128  0.68831169] 0.652666482454 19.09 s
# Unscaled  [ 0.81914894  0.41558442] 0.617366675877 319.13 s
