# -*- coding: utf-8 -*-


import time
import numpy as np

def A_from_structure(structure_filepath):
    """User defined function, to build the A matrices.

    Parameters
    ----------
    structure : string, filepath to the structure

    Return
    ------
    A, structure
    Those two objects will be accessible via global variables: A and STRUCTURE
    """
    import parsimony.functions.nesterov.tv as tv
    import nibabel
    structure = nibabel.load(structure_filepath)
    A, _ = tv.A_from_mask(structure.get_data())
    return A, structure


def mapper(key, output_collector):
    """User defined mapper function. Run the learning algo using key
    (list of parameters). At the end call output_collector.collect(result_dict)
    Nothing is returned.

    Parameters
    ----------
    key : (primary key) list of parameters.

    Global variables
    ----------------
    DATA : list(len == file matched by --data) of list(len == 2) of numpy arr.
    Typically: {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}

    STRUCTURE : as returned by A_from_structure

    A : as returned by A_from_structure

    Output
    ------
    None but call output_collector.collect(result_dict)
    """
    import nibabel
    from parsimony.estimators import RidgeLogisticRegression_L1_TV
    from parsimony.algorithms.explicit import StaticCONESTA
    from parsimony.utils import LimitedDict, Info
    Xtr = DATA["X.center"][0]
    Xte = DATA["X.center"][1]
    ytr = DATA["y.center"][0]
    yte = DATA["y.center"][1]
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto",
                                    algorithm=StaticCONESTA(info=LimitedDict(Info.num_iter)))
    time_curr = time.time()
    mod.fit(Xtr, ytr)
    y_pred = mod.predict(Xte)
    print "Time :",key, ":", time.time() - time_curr, "ite:%i" % mod.algorithm.info.get(Info.num_iter)
    time_curr = time.time()
    mod.A = None
    structure_data = STRUCTURE.get_data() != 0
    arr = np.zeros(structure_data.shape)
    arr[structure_data] = mod.beta.ravel()
    beta3d = nibabel.Nifti1Image(arr, affine=STRUCTURE.get_affine())
    ret = dict(model=mod, y_pred=y_pred, y_true=yte, beta3d=beta3d)
    output_collector.collect(key, ret)

    
def reducer(key, values):
    """ Get a bag of all the values associated with key. Produces a table 
    of final values.
    
    Parameters
    ----------
    key : string of intermediary key

    values: list of dict.
        list of all the values associated with intermediary key.

    Return
    ------
    pandas.DataFrame
    """
    from sklearn.metrics import precision_recall_fscore_support
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]     
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = np.mean([len(item["model"].info["t"]) for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores

