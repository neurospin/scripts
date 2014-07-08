#
# This scripts creates the list of subjects (subjects_id.csv)
# the BMI (BMI.csv) and the SNPs (SNPs.csv)
# This is the information that can be extracted from the old .Rdata files.
# TODO: write covariates?
#

rm(list=ls()) ; gc()

## Data folder:
DATA_FOLDER="/neurospin/brainomics/2013_imagen_bmi/data"
setwd(DATA_FOLDER)

#########################################
# Read subject list, BMI and covariates #
# for all subjects                      #
#########################################
require(gdata)

IN_CLINIC_FILE=file.path(DATA_FOLDER, "clinic", "1534bmi-vincent2.xls")
tab <- read.xls(IN_CLINIC_FILE)

## List of individuals for which we have images
## This is a string vector
clinic_subjects_id <- as.character(tab$Subjects)

## BMI
## We create a matrix indexed by clinic_subjects_id
## so elements can be accessed by BMI["75717", ]
BMI <- data.matrix(tab$BMI)
colnames(BMI) <- "BMI"
rownames(BMI) <- clinic_subjects_id

###############################
# Read the SNPs and save them #
###############################
require(snpStats)

IN_PLINK_FILE=file.path(DATA_FOLDER, "genetics", "bmi_snp")

## read SNPs data
snps_raw <- data.matrix(read.plink(IN_PLINK_FILE)$genotypes@.Data)
SNPs <- apply(snps_raw,2,as.numeric)
impute <- function(v) {
   v[v==0] <- median(v[v!=0])
   return(v) 
}
SNPs <- apply(SNPs,2,impute)

dimnames(SNPs) <- dimnames(snps_raw)

## List of individuals for which we have genetics
SNPs_subjects_id = rownames(SNPs)

###########################
# Subsample and save data #
###########################

# We select subjects for which
#  - we have the genetics data
#  - image quality is A or B
subjects_id_gen_image = intersect(clinic_subjects_id, SNPs_subjects_id)
subjects_id = subset(tab, Subjects%in%subjects_id_gen_image & quality_control != 'C')$Subjects
subjects_id_char = as.character(subjects_id)

## Create a data frame to have a nicer output
subjects_id_df <- as.matrix(subjects_id)
colnames(subjects_id_df) <- "subject_id"
OUT_ID_FILE=file.path(DATA_FOLDER, "subjects_id.csv")
write.csv(subjects_id_df, OUT_ID_FILE, row.names=FALSE)

OUT_BMI_FILE=file.path(DATA_FOLDER, "BMI.csv")
# R don't want to write the "BMI" columns so we try to fuck it.
BMI_subset = as.matrix(BMI[subjects_id_char, ])
colnames(BMI_subset) <- "BMI"
write.csv(BMI_subset, OUT_BMI_FILE, row.names=TRUE)

OUT_SNP_FILE=file.path(DATA_FOLDER, "SNPs.csv")
write.csv(SNPs[subjects_id_char, ], OUT_SNP_FILE, row.names=TRUE)
