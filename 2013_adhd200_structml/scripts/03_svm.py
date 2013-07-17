# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:43:24 2013

@author: ed203246

"""
#import logging

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import Scorer, recall_score 

# Scripts PATH
try:
    ADHD200_SCRIPTS_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
  ADHD200_SCRIPTS_BASE_PATH = os.path.join(os.environ["HOME"] , "git", "scripts", "2013_adhd200_structml")

#ADHD200_DATA_BASE_PATH = "/neurospin/adhd200"
ADHD200_DATA_BASE_PATH = "/volatile/duchesnay/data/2013_adhd_structml"
INPUT_PATH = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")

execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "data_api.py"))

## ==========================================================================
feature = "mw"


# DATA PATH
#INPUT_PATH2 = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")

X_train , y_train = get_data(INPUT_PATH, feature, test=False)
X_test , y_test = get_data(INPUT_PATH, feature, test=True)

# Scale data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



svmlin = svm.LinearSVC(class_weight='auto')
%time svmlin.fit(X_train, y_train)
# CPU times: user 17.58 s, sys: 1.50 s, total: 19.09 s

y_pred = svmlin.predict(X_test)
# global accuracy
print svmlin.score(X_test, y_test)

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
