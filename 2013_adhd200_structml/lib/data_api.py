from os.path import join, exists
import tables
from numpy import where, array

_base_path = join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")
#_base_path = "/neurospin/adhd200/python_analysis/data"


def get_h5_path(feature,test=False):
    if test:
        tt = "test"
    else:
        tt = "train"
    pp = join(_base_path,feature+"_"+tt+".h5")
    if exists(pp):
        return pp
    else:
        raise Exception("feature does not seem to exist")
        return

def get_csv_path(feature,test=False):
    if test:
        tt = "test"
    else:
        tt = "train"
    pp = join(_base_path,feature+"_"+tt+".csv")
    if exists(pp):
        return pp
    else:
        raise Exception("feature does not seem to exist")
        return

def get_csv(feature,test=False,test_filtered=True):
    f = open(get_csv_path(feature,test))
    rows = []
    import csv
    reader = csv.reader(f)
    if test and test_filtered:
        for r in reader:
            if r[1] != "pending":
                rows.append(r)
    else:
        for r in reader:
            rows.append(r)
    f.close()
    return rows

def get_data(feature,test=False,test_filtered=True):
    h5_path = get_h5_path(feature,test)
    f = tables.openFile(h5_path)
    X = f.root.X.read()
    Y = array(f.root.DX.read())
    if test and test_filtered:
        X = X[where(Y>-1)]
        Y = Y[where(Y>-1)]
    return (X,Y)

def get_mask_path():
    return join(_base_path,"mask_t0.1_sum0.8_closing.nii")

def get_mask():
    import nibabel
    return nibabel.load(get_mask_path()).get_data()