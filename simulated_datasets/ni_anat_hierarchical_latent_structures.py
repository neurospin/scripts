# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:49:36 2013

Neuroimaging anatomical simulated datasets based on hierarchical latent variable
model.

We simulated a realistic neuroimaging dataset of 500 anatomical (three dimensional $91 \times 109 \times 91$ voxels) images.
Each image was sampled according to a hierarchical latent variables model where latent variables correspond to brain regions at difference scale.
The hierarchical brain model is based on 12 large regions (lobes $l$) defined by the TD\_lobe atlas \cite{Maldjian2003}, and 116  smaller regions (structures $s$) defined by the aal atlas \cite{Tzourio-Mazoyer2002}.
Each voxel $j$ within lobe $l$ and structure $s$ is defined by:
$ x_{jls} = \text{lobe}_l + \text{struct}_s + e_j + \text{gm}_j $
Where: $\text{lobe}_l$ and $\text{struct}_s \sim \mathcal{N}(0, 1)$ are  latent
random variables associated to lobe $l$ and structure $s$, creating large scale and smaller scale covariance structure between voxels. $e_i \sim \mathcal{N}(0, 1)$ is the random variable associated to voxel $i$ and $\text{gm}_i$ is the grey matter mean of voxel $i$ defined from \cite{Mazziotta2001}. Then each image was smoothed with an isotropic Gaussian filter with one  standard-deviation inducing a local covariance structure mimicking realigned structural brain images. $X$ was obtained by masking the images using $p=???$ voxels corresponding to grey matter. The causal model from $X$ to the target variable $y$ was obtained  using a $p$-dimensional $\beta$ vector with null values except for $???$ voxels within bilateral Hippocampi ($\beta_i=1$) and Amygdalas ($\beta_i=0.2$). Then the image of $\beta$ was smoothed using the same Gaussian filter. Finally, the target variable was obtained adding some noise $e$ to the linear combination of voxels: $y = X \beta + e, e_i \sim \mathcal{N}(0, \sigma)$ where $\sigma$ ensure a signal to noise ratio equal to 2.

"""
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil

def add_latents(Xim, label_im, std=1.):
    for label in np.unique(label_im):
        if label == 0:
            continue
        print label
        latent = np.random.normal(0, std, Xim.shape[0])
        mask = label_im == label
        Xim[:, mask] += latent[:, np.newaxis]
    return Xim

def spatial_smoothing(Xim, sigma):
    for i in xrange(Xim.shape[0]):
        Xim[i, :] = ndimage.gaussian_filter(Xim[i, :],
            sigma=sigma)
    X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
    # Spatial smoothing reduced the std-dev, reset it to 1
    return Xim

##############################################################################
beta_on_structures = dict(Amygdala=1./100, Hippocampus=.2/100)
output_dir = "/neurospin/brainomics/neuroimaging_ressources/dataset_simulated_ni/dataset_original"
n_samples = 1000
snr = 2

#b_l = b_s = b_e = b_gm = 1.
std_e = .1; std_l = .1; std_s = .1

sigma_smoothing=1

gm_filepath = "/neurospin/brainomics/neuroimaging_ressources/spm8/apriori/grey.nii"
#mask_filepath = "/i2bm/local/spm8-standalone/spm8_mcr/spm8/apriori/brainmask.nii"
aal_filepath= "/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/aal_MNI_V4"
tdlobe_filepath= "/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/TD_lobe"

gm_image = nib.load(gm_filepath)
gm_arr = gm_image.get_data()
#mask_image = nib.load(mask_filepath)
gm_mask_arr = gm_arr >= .1
tmp = nib.Nifti1Image(gm_mask_arr.astype(np.int16), affine=gm_image.get_affine())
tmp.to_filename(os.path.join(output_dir, "mask_gm.nii"))
# mask_arr.sum(): 260 024 voxels
lobe_arr = nib.load(tdlobe_filepath+".nii").get_data()
struct_arr = nib.load(aal_filepath+".nii").get_data()


shape = gm_mask_arr.shape
n_features = np.prod(shape)
nx, ny, nz = shape

##############################################################################
# x_lkj = b_l * y_l + b_s * z_s + b_e * e_j + GM_j

print "== 1. Build images with random variable => e_j"
E = np.random.normal(0, std_e, n_samples * n_features).reshape(n_samples, n_features)
Xim = E.reshape(n_samples, nx, ny, nz)
# scale down std-dev of voxels out of the maske
Xim[:, np.logical_not(gm_mask_arr)] /= 100.

#plt.matshow(Xim.std(axis=0)[:, :, 40], cmap=plt.cm.gray)

print "== 2. Add GM cte => GM_j"
Xim += gm_arr
#cax=plt.matshow(Xim[5, :, :, 40], cmap=plt.cm.gray); plt.colorbar(cax); plt.show()

print "== 3. Add lobe latent variable => y_l"
Xim = add_latents(Xim=Xim, label_im=lobe_arr, std=std_l)
#cax=plt.matshow(Xim[5, :, :, 40], cmap=plt.cm.gray); plt.colorbar(cax); plt.show()

print "== 4. Add structures latent variable => z_s"
Xim = add_latents(Xim=Xim, label_im=struct_arr, std=std_l)
#cax=plt.matshow(Xim[5, :, :, 40], cmap=plt.cm.gray); plt.colorbar(cax); plt.show()

print "== 5. spatial smoothing"
Xim = spatial_smoothing(Xim, sigma=sigma_smoothing)
#cax=plt.matshow(Xim[5, :, :, 40], cmap=plt.cm.gray); plt.colorbar(cax); plt.show()

print "== 6. betas"
strutures = [l.split() for l in open(aal_filepath+".txt").readlines()[1:]]

beta_no_smooth_nomask = np.zeros(shape)

for s in strutures:
    for k in beta_on_structures:
        if s[1].count(k) > 0:
            mask = struct_arr == int(s[0])
            print k, int(s[0]), mask.sum()
            beta_no_smooth_nomask[mask] = beta_on_structures[k]
#Hippocampus 37 932
#Hippocampus 38 946
#Amygdala 41 220
#Amygdala 42 248
# Smooth the beta
print "== Set negative values to zero"
Xim[Xim < 0] = 0

print "---- 6.1 save beta"
beta_smoothed_nomask = ndimage.gaussian_filter(beta_no_smooth_nomask, sigma=sigma_smoothing)
beta_smoothed_zeros_outmask = beta_smoothed_nomask.copy()
beta_smoothed_zeros_outmask[np.logical_not(gm_mask_arr)] = 0.
#beta_smoothed_zeros_outmask = beta_smoothed_nomask[mask_arr]
tmp = nib.Nifti1Image(beta_no_smooth_nomask, affine=gm_image.get_affine())
tmp.to_filename(os.path.join(output_dir, "beta_no_smooth_nomask.nii"))
tmp = nib.Nifti1Image(beta_smoothed_nomask, affine=gm_image.get_affine())
tmp.to_filename(os.path.join(output_dir, "beta_smoothed_nomask.nii"))
tmp = nib.Nifti1Image(beta_smoothed_zeros_outmask, affine=gm_image.get_affine())
tmp.to_filename(os.path.join(output_dir, "beta_smoothed_zeros_outmask.nii"))
os.symlink(os.path.join(output_dir, "beta_smoothed_zeros_outmask.nii"), os.path.join(output_dir, "beta.nii"))

beta_smoothed_out_of_mask = (beta_smoothed_nomask != 0) & (gm_mask_arr == 0)
beta_smoothed_out_of_mask.sum()
#Out[75]: memmap(402)
beta_smoothed_out_of_mask = beta_smoothed_out_of_mask.astype(int).astype(float)
tmp = nib.Nifti1Image(beta_smoothed_out_of_mask, affine=gm_image.get_affine())
tmp.to_filename(os.path.join(output_dir, "beta_smoothed_out_of_mask.nii"))

print "== 7. y = X beta + noise"
beta = beta_smoothed_zeros_outmask
np.sum(beta!=0)
#Out[136]: 21371

X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
beta_flat = beta.ravel()
print "---- 7.1 Xbeta = np.dot(X, beta_flat)"

Xbeta = np.dot(X, beta_flat)
y = Xbeta + np.random.normal(0, Xbeta.std() / snr, Xbeta.shape[0])
#plt.matshow(beta[:, :, 25], cmap=plt.cm.gray)
#plt.show()

print "== 8. save to %s" % output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#np.save(os.path.join(output_dir, "X.npy"), Xim)
np.save(os.path.join(output_dir, "X_one.npy"), Xim[0:(n_samples/2),:])
np.save(os.path.join(output_dir, "X_two.npy"), Xim[(n_samples/2):,:])

np.save(os.path.join(output_dir, "y.npy"), y[:, np.newaxis])
np.save(os.path.join(output_dir, "mask_gm.npy"), (gm_mask_arr != 0))
np.save(os.path.join(output_dir, "beta_smoothed_zeros_outmask.npy"), beta)
os.symlink(os.path.join(output_dir, "beta_smoothed_zeros_outmask.npy"), os.path.join(output_dir, "beta.npy") )
#beta_im = nib.Nifti1Image(beta, affine=gm_image.get_affine())
#beta_im.to_filename(os.path.join(output_dir, "beta.nii"))

print "== 8. save some samples %s" % output_dir
selection = np.random.randint(0,n_samples,10)
for index in selection:
        img_arr = Xim[index, :, :, :]
        tmp = nib.Nifti1Image(img_arr, gm_image.get_affine())
        nib.save(tmp, os.path.join(output_dir, 'samples', 'sample%i' % index))

if False:
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
    #%time pred = l1l2.fit(Xtr, ytr).predict(Xte)
    
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
    #%time pred = l1l2cv.fit(Xtr, ytr).predict(Xte)
