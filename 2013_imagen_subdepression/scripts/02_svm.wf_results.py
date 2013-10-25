# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:19:17 2013

@author: md238665

This script reload a workflow and inspect the results and best parameters

"""

# Standard library modules
import os, sys, argparse
# Scipy
import scipy.stats

# EPAC
import epac, epac.map_reduce.engine

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api

TEST_MODE       = False
DEFAULT_WF_NAME = "svm_wf"

parser = argparse.ArgumentParser(description='''Load a SVM classification workflow and display results.''')

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

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm')
WORKFLOW_PATH= os.path.join(OUT_DIR, args.wf_name)
if not os.path.exists(WORKFLOW_PATH):
    raise Exception('{path} not found'.format(path=WORKFLOW_PATH))

svms_auto_cv = epac.map_reduce.engine.SomaWorkflowEngine.load_from_gui(WORKFLOW_PATH)
print "Workflow loaded"
svms_auto_cv_results = svms_auto_cv.reduce()
# Display CV results
print "Evaluation results:", svms_auto_cv_results
epac.export_csv(svms_auto_cv, svms_auto_cv_results, os.path.join(WORKFLOW_PATH, "cv_results.csv"))
# Test significance of recall rates
recalls = svms_auto_cv_results['CVBestSearchRefit']['y/test/score_recall']
p_value_control = scipy.stats.binom_test(recalls[0]*N_CONTROL, n=N_CONTROL, p=P_CONTROL)
if (p_value_control < 0.05):
    print "Control recall is significant"
else:
    print "Control recall is not significant"
p_value_subdep = scipy.stats.binom_test(recalls[1]*N_SUBDEP, n=N_SUBDEP, p=P_SUBDEP)
if (p_value_subdep < 0.05):
    print "Subdep recall is significant"
else:
    print "Subdep recall is not significant"
# Test mean recall
mean_recall = svms_auto_cv_results['CVBestSearchRefit']['y/test/score_recall_mean']
p_value_mean = scipy.stats.binom_test(int(mean_recall * N_SUBJECTS), n=N_SUBJECTS, p=.5)
if (p_value_mean < 0.05):
    print "Average recall is significant"
else:
    print "Average recall is not significant"

# Display best parameters for each fold & test significance of recall rates
for (i, leaf) in enumerate(svms_auto_cv.walk_leaves()):
    fold_results = leaf.load_results()
    epac.export_csv(svms_auto_cv, svms_auto_cv_results, os.path.join(WORKFLOW_PATH, "fold{i}_results.csv".format(i=i)))
    theta = fold_results['CVBestSearchRefit']['best_params'][1]
    print "Best SVM params for fold {i}: C={C}, regularization={reg}".format(i=i, C=theta['C'], reg=theta['penalty'])
