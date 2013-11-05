#############################################################
## Read the images, remove covariates effect and save them ##
#############################################################

rm(list=ls()) ; gc()

require(rhdf5)
require(gdata)
require(oro.nifti)

DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
IMG_PATH = file.path(DATA_PATH, "subsampled_images/smooth")
DATASET_FILE = file.path(DATA_PATH, "dataset.hdf5")
OUTPUT_GROUP="smoothed_images_subsampled_residualized_gender_center_TIV_pds"
h5createGroup(DATASET_FILE, OUTPUT_GROUP)
setwd(DATA_PATH)

# Read images
# GROUP_NAME="smoothed_images"
# smoothed_images = t(h5read(DATASET_FILE, paste(GROUP_NAME, "masked_images", sep="/")))

# Read covariates
tab <- read.xls(file.path("clinic", "1534bmi-vincent2.xls"))
covar <- tab[,c("Gender.de.Feuil2","ImagingCentreCity","tiv_gaser","mean_pds")]

### Reduced images, with the provided reduced mask
id <- paste0("rsmwp1",sprintf("%012i",tab$Subject))
img_vec <- dir(IMG_PATH, pattern="^rsmw.*nii") # Image names in IMG_PATH (not full path)
Mask <- readNIfTI(file.path(IMG_PATH, "rmask.nii"))
mask <- Mask@.Data
n <- length(id)
ind <- mask!=0
p <- sum(ind)
IMG <- matrix(NA,n,p)
IMG_covar <- matrix(NA,n,p)

for (i in 1:length(id)) {
  img_id <- img_vec[pmatch(id[i],img_vec)]
  if (is.na(img_id)) stop("Image not found")
  full_file = file.path(IMG_PATH, img_id)
  img <- readNIfTI(full_file)
  IMG[i,] <- as.vector(img@.Data[ind]) ### No data in the images
  cat(i,full_file,"\n")
  rm(img_id,img) ; gc()
}

# Reindex
rownames(IMG) <- as.character(tab$Subject)
rownames(covar) <- as.character(tab$Subject)

pb <- txtProgressBar(style=3)
for (j in 1:ncol(IMG)) {
  IMG_covar[,j] <- lm(img~.,data=data.frame(img=IMG[,j],covar))$residuals
  setTxtProgressBar(pb, j/ncol(IMG))
}
close(pb)

dimnames(IMG_covar) <- dimnames(IMG)

######################################################
## Read BMI subjects ID and save residualized data in HDF5 ##
######################################################

bmi_subjects <- h5read(DATASET_FILE, "subject_id")

bmi_IMG_covar <- IMG_covar[as.character(bmi_subjects), ]
name=file.path(OUTPUT_GROUP, "masked_images")
h5write(t(bmi_IMG_covar), DATASET_FILE, name)

bmi_covar <- covar[as.character(bmi_subjects), ]
name=file.path(OUTPUT_GROUP, "covar")
h5write(t(bmi_covar), DATASET_FILE, name)

name=file.path(OUTPUT_GROUP, "mask")
h5write(aperm(mask), DATASET_FILE, name)
