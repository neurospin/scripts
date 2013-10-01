# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:19:17 2013

@author: md238665

This script reload a workflow and inspect the results and best parameters

"""

# Standard library modules
import os, sys, argparse

import epac, epac.map_reduce.engine

TEST_MODE     = False
DEFAULT_WF_NAME    = "svm_wf"
DEFAULT_IMAGES_NAME = "masked_images"

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
# Display best parameters for each fold
for (i, leaf) in enumerate(svms_auto_cv.walk_leaves()):
    fold_results = leaf.load_results()
    epac.export_csv(svms_auto_cv, svms_auto_cv_results, os.path.join(WORKFLOW_PATH, "fold{i}_results.csv".format(i=i)))
    theta = fold_results['CVBestSearchRefit']['best_params'][1]
    print "Best SVM params for fold {i}: C={C}, regularization={reg}".format(i=i, C=theta['C'], reg=theta['penalty'])
