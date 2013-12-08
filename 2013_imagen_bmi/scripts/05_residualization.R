#############################################################
## Read the images, remove covariates effect and save them ##
#############################################################

# Warning: R stores data in column-major order so when reading data from HDF5 file we should transpose them.
# In order to avoid that we use the opposite convention (features in line, individuals in columns).
# This makes the script faster.

rm(list=ls()) ; gc()

require(rhdf5)
require(gdata)
require(oro.nifti)

#
# Parameters
#

DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
setwd(DATA_PATH)

BMI_SUBJECTS_FILE=file.path(DATA_PATH, "subjects_id.csv")

# Input file of each image type
DATASET_FILES = list(
  file.path(DATA_PATH, "non_smoothed_images.hdf5"),
  file.path(DATA_PATH, "smoothed_images.hdf5"),
  file.path(DATA_PATH, "smoothed_images_sigma=2.hdf5"),
  file.path(DATA_PATH, "smoothed_images_sigma=4.hdf5"),
  file.path(DATA_PATH, "smoothed_images_sigma=6.hdf5"))

GROUP_NAME="standard_mask"

INPUT_NAME="masked_images"
FULL_INPUT_NAME=file.path(GROUP_NAME, INPUT_NAME)

OUTPUT_NAME="residualized_images_gender_center_TIV_pds"
FULL_OUTPUT_NAME=file.path(GROUP_NAME, OUTPUT_NAME)

#
# Processing
#

# Read subjects ID
bmi_subjects <- read.csv(BMI_SUBJECTS_FILE)$subject_id

# Read covariates
tab <- read.xls(file.path("clinic", "1534bmi-vincent2.xls"))
covar <- tab[,c("Gender.de.Feuil2","ImagingCentreCity","tiv_gaser","mean_pds")]

# Reindex covar
rownames(covar) <- as.character(tab$Subject)

# Subsample covar
bmi_covar <- covar[as.character(bmi_subjects), ]

for (dataset_file in DATASET_FILES) {
  msg = sprintf("Processing %s", dataset_file)
  print(msg)
  # Read images
  # Warning: to save transposition, we use the reversed order
  # However, we use the usual meaning of n and p
  IMG = h5read(dataset_file, file.path(FULL_INPUT_NAME))
  p = dim(IMG)[1]
  n = dim(IMG)[2]
  colnames(IMG) <- as.character(bmi_subjects)
  print("Data read")
  
  # Residualize
  # Warning: we use in place residualization
  pb <- txtProgressBar(style=3)
  for (j in 1:p) {
    IMG[j,] <- lm(img~.,data=data.frame(img=IMG[j,], bmi_covar))$residuals
    setTxtProgressBar(pb, j/p)
  }
  close(pb)
  print("Residualization done")
  
  # Save residualized data in HDF5
  h5write(IMG, dataset_file, FULL_OUTPUT_NAME, level=5)
  print("Residualized data written")
  
  rm(list=c("IMG")); gc()
}
