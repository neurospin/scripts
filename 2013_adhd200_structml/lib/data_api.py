from os.path import join, exists
import tables
from numpy import where, array

#base_path = "/neurospin/adhd200/python_analysis/data"


def get_h5_path(base_path, feature, test=False, test_exist=True):
    if test:
        tt = "test"
    else:
        tt = "train"
    pp = join(base_path, feature+"_"+tt+".h5")
    if not test_exist or exists(pp):
        return pp
    else:
        raise Exception("feature does not seem to exist")
        return

def get_csv_path(base_path, feature, test=False, test_exist=True):
    if test:
        tt = "test"
    else:
        tt = "train"
    pp = join(base_path, feature+"_"+tt+".csv")
    if not test_exist or exists(pp):
        return pp
    else:
        raise Exception("feature does not seem to exist")
        return

def get_csv(base_path, feature,test=False,test_filtered=True):
    f = open(get_csv_path(base_path, feature,test))
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

def get_data(base_path, feature, test=False, test_filtered=True):
    h5_path = get_h5_path(base_path, feature, test)
    f = tables.openFile(h5_path)
    X = f.root.X.read()
    Y = array(f.root.DX.read())
    if test and test_filtered:
        X = X[where(Y>-1)]
        Y = Y[where(Y>-1)]
    return (X,Y)

def write_data(X, y, base_path, feature, test=False):
    file_path = get_h5_path(base_path, feature, test=test, test_exist=False)
    print file_path
    out_hf5 = tables.openFile(file_path, "w")
    out_hf5.createArray(out_hf5.root, "X", X)
    out_hf5.createArray(out_hf5.root, "DX", y)
    out_hf5.flush()
    out_hf5.close()

def get_mask_path(base_path):
    return join(base_path,"mask_t0.1_sum0.8_closing.nii")

def get_mask(base_path="/neurospin/adhd200/python_analysis/data"):
    import nibabel
    return nibabel.load(get_mask_path(base_path)).get_data()