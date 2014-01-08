import joblib
import numpy as np
from glob import glob
import os.path as path
import sys

if __name__ == "__main__":
    workdir = sys.argv[1]
    wf_id = int(sys.argv[2])
    other_id = int(sys.argv[3])
    res_dir = sys.argv[4]
    #flist = glob(path.join(workdir, "*/*.joblib"))
    # n_voxels = 336188
    n_perm = 10000
    h0 = np.zeros((n_perm - 1))
    if other_id > 0:
        flist = glob(path.join(workdir, "wf_%d/result_%d?_*.joblib" % (wf_id, other_id)))
    else:
        flist = glob(path.join(workdir, "wf_%d/result_?_*.joblib" % wf_id))
    n_f = len(flist)
    cpt = 0
    for resfile in flist[0:1]:
        cpt += 1
        print "%.2f : %s" % (100 * cpt / float(n_f) , resfile)
        sys.stdout.flush()
        ar = joblib.load(resfile)
        sizes = ar['n_results_per_perm']
        data = ar['sparse_results']
        h1 = data[:sizes[0]]
        start = 0
        end = sizes[0]
        for i in range(len(sizes) - 1):
            try:
                h0[i] = np.max(data[start:end]['score'])
            except:
                pass
            start = end
            end += sizes[i + 1]

    for resfile in flist[1:]:
        cpt += 1
        print "%.2f : %s" % (100 *cpt / float(n_f) , resfile)
        sys.stdout.flush()
        ar = joblib.load(resfile)
        sizes =ar['n_results_per_perm']
        data = ar['sparse_results']
        h1 = np.concatenate((h1, data[:sizes[0]]))
        start =0
        end = sizes[0]
        for i in range(len(sizes) - 1):
            try:
                h0[i] = max(h0[i], np.max(data[start:end]['score']))
            except:
                pass
            start = end
            end += sizes[i + 1]
    joblib.dump({'h1' : h1, 'h0' : h0}, path.join(res_dir, "red_%d_%d.joblib" % (wf_id,other_id)),compress=1)

