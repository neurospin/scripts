# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:49:36 2013

Neuroimaging anatomical simulated datasets based on hierarchical latent variable
model. We define a hierarchical brain model based on lobes (l) and structures (s)
and voxels (j).

Each voxel x_lkj is defined as:
x_lkj = b_l * y_l + b_s * z_s + b_e * e_j + GM_j
Where:
y_l ~ N(0, 1) is the latent random variable associated to lobe l.
z_s ~ N(0, 1) is the latent random variable associated to structure s.
e_j ~ N(0, 1) is the random noize associated to voxel j.
GM_j is a fixed value of grey matter at voxel j

b_l, b_s and b_e are the mixing coreficients.

Lobes are defined by TD_lobe atlas.
Structures  are defined by aal_MNI_V4 atlas.
"""
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil

def add_latents(Xim, label_im, coef=1):
    for label in np.unique(label_im):
        if label == 0:
            continue
        print label
        latent = np.random.normal(0, 1, Xim.shape[0])
        mask = label_im == label
        Xim[:, mask] += (coef * latent[:, np.newaxis])
    return Xim

def spatial_smoothing(Xim, sigma):
    for i in xrange(Xim.shape[0]):
        Xim[i, :] = ndimage.gaussian_filter(Xim[i, :],
            sigma=sigma)
    X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
    # Spatial smoothing reduced the std-dev, reset it to 1
    return Xim

##############################################################################
beta_on_structures = dict(Amygdala=1./100, Hippocampus=-1./100)
output_dir = "/neurospin/brainomics/neuroimaging_ressources/dataset_simulated_ni_500"
n_samples = 500
snr = 2

b_l = b_s = b_e = 1
sigma_smoothing=1

gm_filepath = "/i2bm/local/spm8-standalone/spm8_mcr/spm8/apriori/grey.nii"
mask_filepath = "/i2bm/local/spm8-standalone/spm8_mcr/spm8/apriori/brainmask.nii"
aal_filepath= "/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/aal_MNI_V4"
tdlobe_filepath= "/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/TD_lobe"

gm_arr = nib.load(gm_filepath).get_data()
mask_image = nib.load(mask_filepath)
mask_arr = mask_image.get_data()
lobe_arr = nib.load(tdlobe_filepath+".nii").get_data()
struct_arr = nib.load(aal_filepath+".nii").get_data()


shape = mask_arr.shape
n_features = np.prod(shape)
nx, ny, nz = shape

##############################################################################
# x_lkj = b_l * y_l + b_s * z_s + b_e * e_j + GM_j

print "== 1. Build images with noize => e_j"
E = b_e * np.random.normal(0, 1, n_samples * n_features).reshape(n_samples, n_features)
Xim = E.reshape(n_samples, nx, ny, nz)

print "== 2. Add GM cte => GM_j"
Xim += 10 * gm_arr

print "== 3. Add lobe latent variable => y_l"
Xim = add_latents(Xim=Xim, label_im=lobe_arr, coef=b_l)

print "== 4. Add structures latent variable => z_s"
Xim = add_latents(Xim=Xim, label_im=struct_arr, coef=b_s)
#plt.matshow(Xim.std(axis=0)[:, :, 40], cmap=plt.cm.gray)

print "== 5. spatial smoothing"
Xim = spatial_smoothing(Xim, sigma=sigma_smoothing)

print "== 6. betas"
strutures = [l.split() for l in open(aal_filepath+".txt").readlines()[1:]]

beta = np.zeros(shape)

for s in strutures:
    for k in beta_on_structures:
        if s[1].count(k) > 0:
            mask = struct_arr == int(s[0])
            print k, int(s[0]), mask.sum()
            beta[mask] = beta_on_structures[k]

X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
beta_flat = beta.ravel()
y_true = np.dot(X, beta_flat)
y = y_true + np.random.normal(0, y_true.std() / snr, y_true.shape[0])
#plt.matshow(beta[:, :, 25], cmap=plt.cm.gray)
#plt.show()

print "== 7. save to %s" % output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, "X.npy"), Xim)
np.save(os.path.join(output_dir, "y.npy"), y[:, np.newaxis])
np.save(os.path.join(output_dir, "brainmask.npy"), (mask_arr != 0))
np.save(os.path.join(output_dir, "beta.npy"), beta)
beta_im = nib.Nifti1Image(beta, affine=mask_image.get_affine())
beta_im.to_filename(os.path.join(output_dir, "beta.nii"))


##############################################################################
## Evaluate prediction results with enet
input_dir = "/neurospin/brainomics/neuroimaging_ressources/dataset_simulated_ni_500"
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing

from sklearn.metrics import r2_score

Xim = np.load(os.path.join(input_dir, "X.npy"))
y = np.load(os.path.join(input_dir, "y.npy"))
mask = np.load(os.path.join(input_dir, "brainmask.npy"))
beta = np.load(os.path.join(input_dir, "beta.npy"))
image = nib.load(os.path.join(input_dir, "beta.nii"))

#shutil.copyfile(mask_filepath, os.path.join(output_dir, "brainmask.nii"))
scaler = preprocessing.StandardScaler()
n_samples, nx, ny, nz = Xim.shape
shape = nx, ny, nz
X = Xim[:, mask]

if True:
    X = scaler.fit_transform(X)
    y -= y.mean()
    y /= y.std()

n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha_g = 10.

alpha = alpha_g * 1. / (2. * n_train)
l1_ratio = .9
l1l2 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
%time pred = l1l2.fit(Xtr, ytr).predict(Xte)

#alpha = alpha_g
#l2 = Ridge(alpha=alpha)
#pred = l2.fit(Xtr, ytr).predict(Xte)
#
#l1 = Lasso(alpha=alpha)
#pred = l1.fit(Xtr, ytr).predict(Xte)
    
algo = l1l2
# Vizualize beta
beta = np.zeros(shape)
beta[mask] = algo.coef_.ravel()
beta_im = nib.Nifti1Image(beta, affine=image.get_affine())
beta_im.to_filename("/tmp/beta_hat_l1:%.2f_scaled.nii" % l1_ratio)
#beta_im.to_filename("/tmp/beta_hat_cv.nii" % l1_ratio)
r2_score(yte, pred)

## CV
l1l2cv = ElasticNetCV()
%time pred = l1l2cv.fit(Xtr, ytr).predict(Xte)
