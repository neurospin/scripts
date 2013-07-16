# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

DOC

"""

DEFAULT_C_VALUES = [0.1, 1, 10]
DEFAULT_REGULARIZATION_METHODS = ['l1', 'l2']
DEFAULT_K_VALUES = 'auto'

DEFAULT_FOLDS_NESTED  = 5
DEFAULT_FOLDS_EVAL    = 5

DEFAULT_NUM_PROCESSES = 5

# Standard library modules
import os, sys, argparse, ast
# Numpy and friends
import sklearn, sklearn.svm, sklearn.feature_selection
# For writing HDF5 files
import tables

import epac, epac.map_reduce.engine

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

def do_all(h5filename, workflow_dir,
           C_values = DEFAULT_C_VALUES, k_values = DEFAULT_K_VALUES, regularization_methods = DEFAULT_REGULARIZATION_METHODS,
           n_folds_nested = DEFAULT_FOLDS_NESTED, n_folds_eval = DEFAULT_FOLDS_EVAL,
           num_processes = DEFAULT_NUM_PROCESSES):
    '''Create EPAC workflow for SVM parameter selection'''

    # Open the output file
    h5file = tables.openFile(h5filename, mode = "r")
    X, Y, mask = data_api.get_data(h5file)
    n_features = X.shape[1]
    y = Y[:, 0]

    if k_values == 'auto':
        k_values = epac.range_log2(n_features, add_n=True)

    # Create the SVM parameter selection wf
    # Create all the classifiers
    pipelines = epac.Methods(*[
      epac.Pipe(sklearn.feature_selection.SelectKBest(k=k),
                sklearn.preprocessing.StandardScaler(),
                epac.Methods(*[sklearn.svm.LinearSVC(class_weight='auto',
                               C=C, penalty=penalty,
                               dual=False)
                               for C in C_values
                               for penalty in regularization_methods]))
                               for k in k_values])
    # Select the best with CV
    best_pipeline = epac.CVBestSearchRefit(pipelines,
                                           n_folds=n_folds_nested)
    # Evaluate it
    select_wf = epac.CV(best_pipeline,
                 n_folds=n_folds_eval,
                 keep=True)

    # Save it
    sfw_engine = epac.map_reduce.engine.SomaWorkflowEngine(tree_root=select_wf,
                                                           num_processes=num_processes)
    sfw_engine.export_to_gui(workflow_dir, X=X, y=y)

    h5file.close()
    return X, y, select_wf

if __name__ == '__main__':
    # Stupid type convert for k_values
    def convert_k_values(arg):
        try:
            k_values = ast.literal_eval(arg)
        except:
            k_values = arg
        return k_values

    # Parse CLI
    parser = argparse.ArgumentParser(description='''Create a workflow to select SVM parameters.''')

    parser.add_argument('h5filename',
      type=str,
      help='Read from filename')

    parser.add_argument('workflow_dir',
      type=str,
      help='Directory on which to save the workflow')

    parser.add_argument('--C_values',
      type=ast.literal_eval, default=DEFAULT_C_VALUES,
      help='Values for C (python list)')

    parser.add_argument('--k_values',
      type=convert_k_values, default=DEFAULT_K_VALUES,
      help='Values for k (python list or \'auto\')')

    parser.add_argument('--regularization_methods',
      type=ast.literal_eval, default=DEFAULT_REGULARIZATION_METHODS,
      help='Values for k (python string)')

    parser.add_argument('--n_folds_nested',
      type=int, default=DEFAULT_FOLDS_NESTED,
      help='Number of folds to use for model selection')

    parser.add_argument('--n_folds_eval',
      type=int, default=DEFAULT_FOLDS_EVAL,
      help='Number of folds to use for model evaluation')

    parser.add_argument('--num_processes',
      type=int, default=DEFAULT_NUM_PROCESSES,
      help='Number of processes to use')

    args = parser.parse_args()
    X, y, select_wf = do_all(**vars(args))
