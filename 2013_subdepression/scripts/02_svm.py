# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

DOC

"""

# Standard library modules
import os, sys, argparse
# Numpy and friends
import numpy
import sklearn, sklearn.svm, sklearn.feature_selection
# For writing HDF5 files
import tables

import epac, epac.map_reduce.engine

select = sklearn.feature_selection.SelectKBest
scale  = sklearn.preprocessing.StandardScaler()
# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

def apply_svm(h5filename, n_folds_nested, n_folds_eval, n_cores):
    # Open the output file
    h5file = tables.openFile(h5filename, mode = "r")
    X, y, mask = data_api.get_data(h5file)
    #Xnp = numpy.array(X)
    #Ynp = numpy.array(y)
    Xnp = X
    Ynp = y
    n_features = X.shape[1]

    # EPAC workflow
    C_values = [0.1, 1, 10]
    k_values = epac.range_log2(n_features, add_n=True)
    regularization_methods = ['l1', 'l2']
    # Create all the classifiers
    pipelines = epac.Methods(*[
      epac.Pipe(select(k=k), scale, epac.Methods(*[
                                      sklearn.svm.LinearSVC(class_weight='auto',
                                                            C=C, penalty=penalty,
                                                            dual=False)
                                      for C in C_values
                                      for penalty in regularization_methods]))
       for k in k_values])
    # Select the best with CV
    best_pipeline = epac.CVBestSearchRefit(pipelines,
                                           n_folds=n_folds_nested)
    # Evaluate it
    wf = epac.CV(best_pipeline,
                 n_folds=n_folds_eval)
    # Run the workflow
    print 'Running'
    epac.conf.TRACE_TOPDOWN=True
    #local_engine = epac.map_reduce.engine.LocalEngine(tree_root=wf, num_processes=n_cores)
    #wf = local_engine.run(X=Xnp, y=Ynp[:, 0])
    wf.run(X=Xnp, y=Ynp[:, 0])
    print 'Finished'
    
    h5file.close()
    return wf.reduce()

if __name__ == '__main__':
    DEFAULT_FOLDS_NESTED = 5
    DEFAULT_FOLDS_EVAL   = 5
    DEFAULT_CORES = 5

    # Parse CLI
    parser = argparse.ArgumentParser(description='''Load the data in HDF5 and apply SVM.''')

    parser.add_argument('h5filename',
      type=str,
      help='Write to outfilename')

    parser.add_argument('--n_folds_nested',
      type=int, default=DEFAULT_FOLDS_NESTED,
      help='Number of folds to use for model selection')
    
    parser.add_argument('--n_folds_eval',
      type=int, default=DEFAULT_FOLDS_EVAL,
      help='Number of folds to use for model evaluation')
      
    parser.add_argument('--n_cores',
      type=int, default=DEFAULT_CORES,
      help='Number of processes to use')
      
    args = parser.parse_args()
    reduced_wf = apply_svm(**vars(args))
