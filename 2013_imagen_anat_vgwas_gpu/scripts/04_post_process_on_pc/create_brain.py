import nibabel as ni
import numpy as np
import joblib
import os

BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
PATH_RES_RESULT = os.path.join(BASE_DIR,
                               'data',
                               'tmp_reduce',
                               'result.joblib')
DATA_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data')
MASK_FILE = os.path.join(DATA_DIR, 'mask', 'mask.nii')
mask_file = MASK_FILE
OUT_IMAGES = os.path.join(BASE_DIR,
                          'interesting_snp_brain_img')

ar = joblib.load(PATH_RES_RESULT)
h1 = ar['h1']
h0 = ar['h0']
snp_of_interest = [122664, 379105]
n_voxels = 336188

mask = ni.load(mask_file).get_data().astype(bool)
aff = ni.load(mask_file).get_affine()
print mask.sum()
# print image1d, image3d
for snp in snp_of_interest:
    image3d = np.zeros(mask.shape) * np.nan
    image1d = np.zeros((n_voxels))
    sp_h1_snp = h1[h1['x_id'] == snp]
    image1d[sp_h1_snp['y_id']] = sp_h1_snp['score']
    image3d[mask] = image1d
    img = ni.Nifti1Image(image3d, aff)
    img_path = os.path.join(OUT_IMAGES, 'snp_%d.nii.gz' % snp)
    ni.save(img, img_path)
