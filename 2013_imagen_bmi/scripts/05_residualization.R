#############################################################
## Read the images, remove covariates effect and save them ##
#############################################################

rm(list=ls()) ; gc()

require(rhdf5)
require(gdata)
require(oro.nifti)

DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
DATASET_FILE = file.path(DATA_PATH, "subsampled_non_smoothed_images.hdf5")
OUTPUT_NAME="images_residualized_gender_center_TIV_pds"
setwd(DATA_PATH)

# Read subjects ID
bmi_subjects <- h5read(DATASET_FILE, "subject_id")

# Read images
GROUP_NAME="standard_mask"
IMG = t(h5read(DATASET_FILE, file.path(GROUP_NAME, "masked_images")))
n = dim(IMG)[1]
p = dim(IMG)[2]
rownames(IMG) <- as.character(bmi_subjects)

# Read covariates
tab <- read.xls(file.path("clinic", "1534bmi-vincent2.xls"))
covar <- tab[,c("Gender.de.Feuil2","ImagingCentreCity","tiv_gaser","mean_pds")]

# Reindex
rownames(covar) <- as.character(tab$Subject)

# Subsample covar
bmi_covar <- covar[as.character(bmi_subjects), ]

# Residualize
IMG_covar <- matrix(0.0,n,p)
pb <- txtProgressBar(style=3)
for (j in 1:ncol(IMG)) {
  IMG_covar[,j] <- lm(img~.,data=data.frame(img=IMG[,j], bmi_covar))$residuals
  setTxtProgressBar(pb, j/ncol(IMG))
}
close(pb)

# I don't set names here because I don't know how it will be stored
#dimnames(IMG_covar) <- dimnames(IMG)

# Save residualized data in HDF5
name=file.path(GROUP_NAME, OUTPUT_NAME)
h5write(t(IMG_covar), DATASET_FILE, name)
