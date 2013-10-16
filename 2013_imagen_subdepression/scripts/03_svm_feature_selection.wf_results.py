# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:19:17 2013

@author: md238665

This script reload a workflow, reduce it and write results in a CSV file.

"""

# Standard library modules
import os, sys, argparse
import itertools, collections
# Numpy, scipy & friends
import numpy, scipy, pandas

import epac, epac.map_reduce.engine

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api, utils

TEST_MODE     = False
DEFAULT_WF_NAME    = "svm_wf"
MAX_N_FEATURES = 16384

if TEST_MODE:
    C_VALUES = [0.1, 1, 10]
else:
    C_VALUES = [0.1, 0.5, 1, 5, 10]

REGULARIZATION_METHODS = ['l1', 'l2']

if TEST_MODE:
    K_VALUES = [32, MAX_N_FEATURES]
else:
    K_VALUES = utils.range_log2(32, MAX_N_FEATURES)

parser = argparse.ArgumentParser(description='''Load a SVM+feature selection workflow and display results.''')

parser.add_argument('--wf_name',
      type=str, default=DEFAULT_WF_NAME,
      help='Name of the workflow (default: %s)'% (DEFAULT_WF_NAME))

args = parser.parse_args()

if TEST_MODE:
    DB_PATH='/volatile/DB/micro_subdepression/'
    LOCAL_PATH='/volatile/DB/cache/micro_subdepression.hdf5'
else:
    DB_PATH='/neurospin/brainomics/2013_imagen_subdepression'
    LOCAL_PATH='/volatile/DB/cache/imagen_subdepression.hdf5'

# Compute a priori probabilities
clinic_file_path = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(clinic_file_path)
N_SUBJECTS = float(df.shape[0])
counts = df['group_sub_ctl'].value_counts()
N_CONTROL = float(counts['control'])
P_CONTROL = N_CONTROL / N_SUBJECTS
N_SUBDEP = float(counts['sub'])
P_SUBDEP = N_SUBDEP / N_SUBJECTS

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm_feature_selection')
WORKFLOW_PATH= os.path.join(OUT_DIR, args.wf_name)
if not os.path.exists(WORKFLOW_PATH):
    raise Exception('{path} not found'.format(path=WORKFLOW_PATH))

svms_cv = epac.map_reduce.engine.SomaWorkflowEngine.load_from_gui(WORKFLOW_PATH)
print "Workflow loaded"
svms_cv_results = svms_cv.reduce()
print "Reduce done"
# Write CV results in a CSV file
epac.export_csv(svms_cv, svms_cv_results, os.path.join(WORKFLOW_PATH, "cv_results.csv"))

# Convert results to a pandas panel:
#  - panel item = class or mean
#  - row = parameters
#  - column = metrics
KEY_FORMAT='SelectKBest(k={k})/StandardScaler/LinearSVC(penalty={regularization},C={C})'
COLUMNS=['y/test/score_precision', 'y/test/score_recall', 'y/test/score_accuracy', 'y/test/score_f1', 'y/test/score_recall_mean']

indices = ['k', 'C', 'regularization']
n_rows = len(svms_cv_results)
columns = ['recall', 'recall_significance', 'precision', 'f1', 'accuracy']
n_col = len(columns)
col_map = {'y/test/score_precision': 'precision', 'y/test/score_recall': 'recall', 'y/test/score_f1': 'f1',
           'y/test/score_accuracy': 'accuracy', 'y/test/score_recall_mean': 'recall'}
param_set = list(itertools.product(K_VALUES, REGULARIZATION_METHODS, C_VALUES))
index = pandas.MultiIndex.from_tuples(param_set, names=indices)
panel_items = ['ctl', 'sub', 'mean']
n_items = len(panel_items)
# tmp arrays of NaN
nans = numpy.empty((n_rows, n_col))
nans.fill(numpy.NaN)
# We use this constructor to have NaN
panel = pandas.Panel(collections.OrderedDict.fromkeys(panel_items, pandas.DataFrame(nans, index = index, columns = columns)), copy = True)


for k, regularization, C in param_set:
    key = KEY_FORMAT.format(C=C, k=k, regularization=regularization)
    res = svms_cv_results[key]
    for res_column in COLUMNS:
        if isinstance(res[res_column], numpy.float64):
            panel['mean'][col_map[res_column]][k, regularization, C] = res[res_column]
        else:
            panel['ctl'][col_map[res_column]][k, regularization, C] = res[res_column][0]
            panel['sub'][col_map[res_column]][k, regularization, C] = res[res_column][1]
    # Compute significance of recall
    p_value_control = scipy.stats.binom_test(panel['ctl']['recall'][k, regularization, C]*N_CONTROL, n=N_CONTROL, p=P_CONTROL)
    panel['ctl']['recall_significance'][k, regularization, C] = p_value_control < 0.05
#    if (p_value_control < 0.05):
#        print "Control recall is significant"
#    else:
#        print "Control recall is not significant"
    p_value_subdep = scipy.stats.binom_test(panel['ctl']['recall'][k, regularization, C]*N_SUBDEP, n=N_SUBDEP, p=P_SUBDEP)
    panel['sub']['recall_significance'][k, regularization, C] = p_value_subdep < 0.05
#    if (p_value_subdep < 0.05):
#        print "Subdep recall is significant"
#    else:
#        print "Subdep recall is not significant"
    # Test mean recall
    p_value_mean = scipy.stats.binom_test(int(panel['mean']['recall'][k, regularization, C] * N_SUBJECTS), n=N_SUBJECTS, p=.5)
    panel['mean']['recall_significance'][k, regularization, C] = p_value_mean < 0.05
#    if (p_value_mean < 0.05):
#        print "Average recall is significant"
#    else:
#        print "Average recall is not significant"
panel.to_excel(os.path.join(WORKFLOW_PATH, 'results_with_significance.xls'))