# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:16:43 2013

@author: ed203246

This script performs MULM analysis:
 1) fit various LM and store t-values and p-values maps to find biomarkers
    this should give the same results
 2) generate a workflow to perform a MULM cluster-level analysis with permutation

"""

# Standard library modules
import os, sys
import numpy
import scipy, scipy.ndimage
import tables
import nibabel

import epac
from mulm import LinearRegression
# X(nxp), y(nx1), mask => MULMStats => pval(px1) => ClusterStat() => sizes(kx1)

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_subdepression", "lib"))
import data_api, utils

DB_PATH='/home/md238665/DB/micro_subdep'
LOCAL_PATH='/volatile/micro_subdepression.hdf5'
DEFAULT_N_PERMS  = 1000
DEFAULT_THRESH   = 0.01

class MULMStats(epac.BaseNode):
    def __init__(self):
        epac.BaseNode.__init__(self)
        self.lm = LinearRegression()

    def transform(self, **kwargs):
        X = kwargs['design_matrix']
        Y = kwargs['Y']
        contrast = kwargs['contrast']
        print 'Fitting LM'
        self.lm.fit(X=X, Y=Y)
        print 'Computing p-values'
        tval, pval = self.lm.stats(X, Y, contrast=contrast, pval=True)
        kwargs["tval"] = tval
        kwargs["pval"] = pval
        return kwargs

class ClusterStats(epac.BaseNode):
    def transform(self, **kwargs):
        print 'Clustering'
        mask   = kwargs['mask']
        pval   = kwargs['pval']
        thresh = kwargs['thresh']
        # Convert mask to numpy binary array
        binary_mask = mask!=0
        # Create an image and fill it with values
        img_data = numpy.zeros(mask.shape, dtype=bool)
        img_data[binary_mask] = pval < thresh;
        # Compute connected components
        (labeled_array, num_features) = scipy.ndimage.label(img_data)
        clust_sizes = scipy.ndimage.measurements.histogram(labeled_array, 1, num_features, num_features)
        out = dict(clust_sizes=clust_sizes, min_pval=pval.min())
        kwargs.update(out)
        return kwargs

#########################
# 0th part: access data #
#########################

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()

fd = tables.openFile(LOCAL_PATH)
masked_images = data_api.get_images(fd)

#####################################################
# 1st part: fit MULM models and store p-values maps #
#####################################################
OUT_DIR='results/mulm/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Construct EPAC workflow
pipeline = epac.Pipe(MULMStats(), ClusterStats())

# 1st model
MODEL = ["group_sub_ctl", "Gender", "pds", "VSF", "Scanner_Type"]
MODEL_OUT = os.path.join(OUT_DIR, "-".join(MODEL))
if not os.path.exists(MODEL_OUT):
    os.makedirs(MODEL_OUT)
design_mat = utils.make_design_matrix(df, regressors=MODEL, scale=True)
contrast = numpy.zeros(design_mat.shape[1])
contrast[0] = 1

pipeline_res = pipeline.run(design_matrix=design_mat, Y=masked_images, mask=mask,
                            contrast=contrast, thresh=0.01)
# Make a map
outimg = utils.make_image_from_array(pipeline_res['pval'], babel_mask)
nibabel.save(outimg, os.path.join(MODEL_OUT, 'pval.nii'))

# 2nd model
MODEL = ["group_sub_ctl", "Gender", "Age", "VSF", "Scanner_Type"]
MODEL_OUT = os.path.join(OUT_DIR, "-".join(MODEL))
if not os.path.exists(MODEL_OUT):
    os.makedirs(MODEL_OUT)
design_mat = utils.make_design_matrix(df, regressors=MODEL, scale=True)
contrast = numpy.zeros(design_mat.shape[1])
contrast[0] = 1

pipeline_res = pipeline.run(design_matrix=design_mat, Y=masked_images, mask=mask,
                            contrast=contrast, thresh=0.01)
# Make a map
outimg = utils.make_image_from_array(pipeline_res['pval'], babel_mask)
nibabel.save(outimg, os.path.join(MODEL_OUT, 'pval.nii'))

#######
# End #
#######

fd.close()


#def mulm_stat(h5filename, workflow_dir,
#              thresh=DEFAULT_THRESH, contrast=DEFAULT_CONTRAST,
#              n_perms=DEFAULT_N_PERMS,
#              num_processes=DEFAULT_NUM_PROCESSES):
#
#    # Load the file
#    h5file = tables.openFile(h5filename, mode = "r")
#    images, regressors, mask, mask_affine = data_api.get_data(h5file)
#    Y_dummy = data_api.get_dummy(h5file)
#    n_useful_voxels = images.shape[1]
#    n_obs           = images.shape[0]
#    n_regressors    = Y_dummy.shape[1]
#    if regressors.shape[0] != n_obs:
#        print 'You stupid'
#        sys.exit()
#
#    # Create design matrix: add an intercept column to Y_dummy & normalize
#    design_mat = numpy.ones((n_obs, n_regressors+1))
#    design_mat[:, 0:-1] = Y_dummy
#    design_mat = sklearn.preprocessing.scale(design_mat)
#
#    if contrast == 'auto':
#        contrast = numpy.zeros((n_regressors+1))
#        contrast[0] = 1
#    if num_processes == 'auto':
#        num_processes = n_perms
#
#    pipeline = epac.Pipe(MULMStats(), ClusterStats())
#
#    permutation_wf = epac.Perms(pipeline,
#                                permute="design_matrix",
#                                n_perms=n_perms)
#
#    # Save the workflow
#    sfw_engine = epac.map_reduce.engine.SomaWorkflowEngine(tree_root=permutation_wf,
#                                                           num_processes=num_processes)
#    sfw_engine.export_to_gui(workflow_dir,
#                             Y=images, design_matrix=design_mat, mask=numpy.asarray(mask),
#                             thresh=thresh, contrast=contrast)
#
#    h5file.close()

#if __name__ == '__main__':
#    # Stupid type convert for contrast
#    def convert_contrast(arg):
#        try:
#            contrast = ast.literal_eval(arg)
#        except:
#            contrast = arg
#        return contrast
#
#    def convert_num_processes(arg):
#        try:
#            num_processes = int(arg)
#        except:
#            num_processes = arg
#        return num_processes
#
#    # Parse CLI
#    parser = argparse.ArgumentParser(description='''Create a workflow for MULM and cluster level stat''')
#
#    parser.add_argument('h5filename',
#      type=str,
#      help='Read from filename')
#
#    parser.add_argument('workflow_dir',
#      type=str,
#      help='Directory on which to save the workflow')
#
#    parser.add_argument('--n_perms',
#      type=int, default=DEFAULT_N_PERMS,
#      help='Number of permutations')
#
#    parser.add_argument('--thresh',
#      type=float, default=DEFAULT_THRESH,
#      help='p-values threshold')
#
#    parser.add_argument('--contrast',
#      type=convert_contrast, default=DEFAULT_CONTRAST,
#      help='Contrast (python list or ''auto''; by default 1 with as many 0 as needed')
#
#    parser.add_argument('--num_processes',
#      type=convert_num_processes, default=DEFAULT_NUM_PROCESSES,
#      help='Number of processes to use or ''auto'' (defaut number of permutations)')
#
#    args = parser.parse_args()
#    mulm_stat(**vars(args))
