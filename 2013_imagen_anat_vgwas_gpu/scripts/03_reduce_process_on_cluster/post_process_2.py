"""
post processing for pa prace big dataset.
"""
import sys
import joblib
import numpy as np
from glob import glob
import os.path as path

if __name__ == "__main__":
    workdir = sys.argv[1]
    flist = glob(path.join(workdir, "red*.joblib"))
    n_f = len(flist)
    cpt = 0
    for resfile in flist[0:1]:
        cpt += 1
        print "%.2f : %s" % (100 * cpt / float(n_f) , resfile)
        sys.stdout.flush()
        ar = joblib.load(resfile)
        h1 = ar['h1']
        h0 = ar['h0']

    for resfile in flist[1:]:
        cpt += 1
        print "%.2f : %s" % (100 *cpt / float(n_f) , resfile)
        sys.stdout.flush()
        ar = joblib.load(resfile)
        h1 = np.concatenate((h1, ar['h1']))
        h0 = np.max([h0, ar['h0']], axis=0)
    joblib.dump({'h1' : h1, 'h0' : h0}, path.join(workdir, "result.joblib"), compress=1)
