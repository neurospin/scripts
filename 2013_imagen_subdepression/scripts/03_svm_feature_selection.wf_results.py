# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:19:17 2013

@author: md238665

This script reload a workflow and inspect the results and best parameters

"""

# Standard library modules
import os, argparse

import epac, epac.map_reduce.engine

TEST_MODE     = False
DEFAULT_WF_NAME    = "svm_wf"

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

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm_feature_selection')
WORKFLOW_PATH= os.path.join(OUT_DIR, args.wf_name)
if not os.path.exists(WORKFLOW_PATH):
    raise Exception('{path} not found'.format(path=WORKFLOW_PATH))

svms_cv = epac.map_reduce.engine.SomaWorkflowEngine.load_from_gui(WORKFLOW_PATH)
print "Workflow loaded"
svms_cv_results = svms_cv.reduce()
print "Reduce done"
# Display CV results
print "Evaluation results:", svms_cv_results
epac.export_csv(svms_cv, svms_cv_results, os.path.join(WORKFLOW_PATH, "cv_results.csv"))
