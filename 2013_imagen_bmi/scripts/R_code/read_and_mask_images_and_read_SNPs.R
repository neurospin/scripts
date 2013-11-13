###################################################################
## Read the images, remove covariates effect, mask and save them ##
###################################################################

rm(list=ls()) ; gc()

require(oro.nifti)
require(gdata)


DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
IMG_PATH = file.path(DATA_PATH, "subsampled_images/smooth")
DATASET_FILE = file.path(DATA_PATH, "dataset.hdf5")
OUTPUT_GROUP="smoothed_images_subsampled_residualized_gender_center_TIV_pds"

setwd(DATA_PATH)

### Reduced images, with the provided reduced mask
tab <- read.xls(file.path("clinic", "1534bmi-vincent2.xls"))
id <- paste0("rsmwp1",sprintf("%012i",tab$Subject))
img_vec <- dir(IMG_PATH,pattern="^rsmw.*nii")
Mask <- readNIfTI("rmask.nii")
mask <- Mask@.Data
n <- length(id)
ind <- mask!=0
p <- sum(ind)
IMG <- matrix(NA,n,p)
IMG_covar <- matrix(NA,n,p)

for (i in 1:length(id)) {
  img_id <- img_vec[pmatch(id[i],img_vec)]
  if (is.na(img_id)) stop("Image not found")
  img <- readNIfTI(file.path(IMG_PATH, img_id))
  IMG[i,] <- as.vector(img@.Data[ind]) ### No data in the images
  cat(i,img_id,"\n")
  rm(img_id,img) ; gc()
}

rownames(IMG) <- as.character(tab$Subject)

## read covariates
covar <- tab[,c("Gender.de.Feuil2","ImagingCentreCity","tiv_gaser","mean_pds")]

pb <- txtProgressBar(style=3)
for (j in 1:ncol(IMG)) {
  IMG_covar[,j] <- lm(img~.,data=data.frame(img=IMG[,j],covar))$residuals
  setTxtProgressBar(pb, j/ncol(IMG))
}
close(pb)

dimnames(IMG_covar) <- dimnames(IMG)

save.image(file="save_session_IMG_reduced_residualized_smoothed.RData")

### Below: another more brutal version of the reducing process
## Data folder: 
#setwd("/neurospin/brainomics/2013_imagen_bmi/data")
## List of individuals
#tab <- read.xls("1534bmi-vincent2.xls")
#id <- paste0("/neurospin/brainomics/2012_imagen_subdepression/data_normalized_segmented/gaser_vbm8//mwp1",sprintf("%012i",tab$Subject))
## List of images : TODO : changes for subsampled data
#img_vec <- dir("/neurospin/brainomics/2012_imagen_subdepression/data_normalized_segmented/gaser_vbm8/",pattern="^mw.*nii",full.names=TRUE)
#Mask <- readNIfTI("mask.nii")@.Data
#a <- dim(Mask)[1]
#b <- dim(Mask)[2]
#c <- dim(Mask)[3]
#mask <- Mask[1:a%%2==0, 1:b%%2==0, 1:c%%2==0]
#n <- length(id)
#ind <- mask!=0
#p <- sum(ind)
#IMG <- matrix(NA,n,p)
#for (i in 1:length(id)) {
#  img_id <- img_vec[pmatch(id[i],img_vec)]
#  if (is.na(img_id)) stop("Image not found")
#  img <- readNIfTI(img_id)@.Data[1:a%%2==0, 1:b%%2==0, 1:c%%2==0]
#  IMG[i,] <- as.vector(img[ind])
#  cat(i,img_id,"\n")
#  rm(img_id,img) ; gc()
#}
#rownames(IMG) <- as.character(tab$Subject)
#save.image(file="save_session_IMG_reduced.RData")

#################################
## Read the SNPs and save them ##
#################################

rm(list=ls()) ; gc()

require(snpStats)
require(gdata)

## Data folder: 
setwd("/neurospin/brainomics/2013_imagen_bmi/data")

## List of individuals
tab <- read.xls("1534bmi-vincent2.xls")
id <- paste0("rsmwp1",sprintf("%012i",tab$Subject))

## read SNPs data
snps_raw <- data.matrix(read.plink("bmi_snp")$genotypes@.Data)
SNPs <- apply(snps_raw,2,as.numeric)
impute <- function(v) {
   v[v==0] <- median(v[v!=0])
   return(v) 
}
SNPs <- apply(SNPs,2,impute)

dimnames(SNPs) <- dimnames(snps_raw)

save(SNPs,file="save_SNPs.RData")

######################################################
## Read both, create multiblock dataset and save it ##
######################################################

rm(list=ls()) ; gc()

## Data folder: 
setwd("/neurospin/brainomics/2013_imagen_bmi/data")

## Load data
load("save_SNPs.RData")
#load("save_session_IMG_reduced.RData")
load("save_session_IMG_reduced_residualized_smoothed.RData")
#load("save_session_IMG_reduced_residualized.RData")

QC <- subset(tab,quality_control != "C", select=c("Subject",
"quality_control"))
qc <- as.character(QC$Subject)
zz <- Reduce("intersect",list(rownames(IMG),rownames(SNPs),qc))

block_SNPs <- SNPs[zz,]
block_IMGs <- IMG_covar[zz,]
tmp <- subset(tab,as.character(Subject) %in% zz, 
               select=c("Subject","NI_MASS","NI_HEIGHT","BMI") )
rownames(tmp) <- as.character(tmp$Subject)
block_pheno <- tmp[zz,-1]
colnames(block_pheno) <- c("mass","height","BMI")

A <- list(SNPs=block_SNPs,IMGs=block_IMGs,BMI=block_pheno[,"BMI",drop=F])

# save(A,file="multiblock_dataset_residualized_images.RData")
save(A,file="multiblock_dataset_residualized_smoothed_images_BUG.RData")


