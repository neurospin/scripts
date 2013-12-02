import nibabel as ni
import numpy as np
import joblib

ar = joblib.load('result.joblib')
h1 = ar['h1']
h0 = ar['h0']
snp_of_interest = [122664, 379105]
n_voxels = 336188
mask_file = "mask.nii"
mask = ni.load(mask_file).get_data().astype(bool)
aff = ni.load(mask_file).get_affine()
print mask.sum()
# print image1d, image3d
for snp in snp_of_interest:
    image3d = np.zeros(mask.shape) * np.nan
    image1d = np.zeros((n_voxels))
    sp_h1_snp = h1[h1['x_id'] == snp]
    image1d[sp_h1_snp['y_id']] = -np.log10((10000 - (
            sp_h1_snp['score'].reshape((1, -1))
            >= h0.reshape((-1, 1))).sum(0)) / 10000.)
    image3d[mask] = image1d
    img = ni.Nifti1Image(image3d, aff)
    ni.save(img, 'snp_%d_perm_pcorr.nii.gz' % snp)
