# Plot some graphs
# We use Group.ADNI as the reference group: it is identical to the one used by Cuingnet et al. 2010
# but with an extra subject (027_S_1082).
library(ADNIMERGE)

INPUT_CLINIC_PATH="/neurospin/brainomics/2013_adni/clinic"
INPUT_BL_GROUP_QC=file.path(INPUT_CLINIC_PATH, "adni510_bl_groups_qc.csv")
INPUT_M18_GROUP_QC=file.path(INPUT_CLINIC_PATH, "adni510_m18_groups_qc.csv")

INPUT_QC_PATH="/neurospin/cati/ADNI/ADNI_510/qualityControlSPM/QC"
INPUT_QC_GRADE=file.path(INPUT_QC_PATH, "final_grade.csv")

OUTPUT_CLINIC_PATH=INPUT_CLINIC_PATH
OUTPUT_BL_FIGS=file.path(OUTPUT_CLINIC_PATH, "bl_scores.pdf")
OUTPUT_M18_FIGS=file.path(OUTPUT_CLINIC_PATH, "m18_scores.pdf")

OUTPUT_BL_MCIcAD_FIGS=file.path(OUTPUT_CLINIC_PATH, "bl_scores.MCIc-AD.pdf")
OUTPUT_M18_MCIcAD_FIGS=file.path(OUTPUT_CLINIC_PATH, "m18_scores.MCIc-AD.pdf")

#############
# Functions #
#############

data_plots <- function(data) {
  boxplot(AGE~Group.ADNI, data=data, main="Age")
  boxplot(MMSE~Group.ADNI, data=data, main="MMSE")
  boxplot(ADAS11~Group.ADNI, data=data, main="ADAS11")
  boxplot(ADAS13~Group.ADNI, data=data, main="ADAS13")
  plot(data$ADAS11, data$ADAS13, main="ADAS13 as a function of ADAS11",
       xlab="ADAS11", ylab="ADAS13")
  plot(data$ADAS11, data$MMSE, main="MMSE as a function of ADAS11",
       xlab="ADAS11", ylab="MMSE")
}

####################################
# Read data & reorder group factor #
####################################

adni510_bl_groups_qc <- read.csv(INPUT_BL_GROUP_QC)
adni510_bl_groups_qc$Group.ADNI <- factor(adni510_bl_groups_qc$Group.ADNI, levels=c("control", "MCInc", "MCIc", "AD"))
adni510_m18_groups_qc <- read.csv(INPUT_M18_GROUP_QC)
adni510_m18_groups_qc$Group.ADNI <- factor(adni510_m18_groups_qc$Group.ADNI, levels=c("control", "MCInc", "MCIc", "AD"))

#########################################
# Scores for the whole ADNI 510 dataset #
#########################################

# Baseline scores
pdf(file=OUTPUT_BL_FIGS)
data_plots(adni510_bl_groups_qc)
dev.off()

# m18 scores
pdf(file=OUTPUT_M18_FIGS)
data_plots(adni510_m18_groups_qc)
dev.off()

######################################################
# Scores for MCIc & AD subjects that succeeded in CQ #
######################################################

# A, B indices
is_ab = adni510_bl_groups_qc$Grade %in% c("A", "B")
ab_indices = which(is_ab)

# MCIc, AD indices
is_MCIcAD = adni510_bl_groups_qc$Group.ADNI %in% c("MCIc", "AD")
MCIcAD_indices = which(is_MCIcAD)

# Logical AND (and store just to be sure)
is_cool = is_ab & is_MCIcAD
cool_indices = which(is_cool)
#write.csv2(adni510_bl_groups_qc$PTID[cool_indices], "test.csv", quote=FALSE, row.names=FALSE)

# Subsample and refactor
MCIcAD_adni510_bl_groups_qc = adni510_bl_groups_qc[cool_indices, ]
MCIcAD_adni510_bl_groups_qc$Group.ADNI <- factor(MCIcAD_adni510_bl_groups_qc$Group.ADNI, levels=c("MCIc", "AD"))

MCIcAD_adni510_m18_groups_qc = adni510_m18_groups_qc[cool_indices, ]
MCIcAD_adni510_m18_groups_qc$Group.ADNI <- factor(MCIcAD_adni510_m18_groups_qc$Group.ADNI, levels=c("MCIc", "AD"))

# Plots
pdf(file=OUTPUT_BL_MCIcAD_FIGS)
data_plots(MCIcAD_adni510_bl_groups_qc)
dev.off()

pdf(file=OUTPUT_M18_MCIcAD_FIGS)
data_plots(MCIcAD_adni510_m18_groups_qc)
dev.off()
