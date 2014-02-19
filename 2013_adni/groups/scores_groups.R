# Plot some graphs
# We use Group.ADNI as the reference group: it is identical to the one used by Cuingnet et al. 2010
# but with an extra subject (027_S_1082).
library(ADNIMERGE)

INPUT_CLINIC_PATH="/neurospin/brainomics/2013_adni_preprocessing/clinic"
INPUT_GROUPS=file.path(INPUT_CLINIC_PATH, "groups.csv")

INPUT_QC_PATH="/neurospin/cati/ADNI/ADNI_510/qualityControlSPM/QC"
INPUT_QC_GRADE=file.path(INPUT_QC_PATH, "final_grade.csv")

OUTPUT_CLINIC_PATH=INPUT_CLINIC_PATH
OUTPUT_BL_EXAM=file.path(OUTPUT_CLINIC_PATH, "adni510_bl_groups.csv")
OUTPUT_BL_FIGS=file.path(OUTPUT_CLINIC_PATH, "bl_scores.pdf")

OUTPUT_M18_EXAM=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_groups.csv")
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

######################################################
# Extract baseline and m18 examinations in adnimerge #
######################################################
adnimerge <- adnimerge

# Baseline indexes (in adnimerge)
bl_indices = which(adnimerge$EXAMDATE == adnimerge$EXAMDATE.bl)

# Extract baseline examinations (and index by PTID)
adnimerge_bl = adnimerge[bl_indices, ]
rownames(adnimerge_bl) <- adnimerge_bl$PTID

# Indexes of last examination before 18 months (in adnimerge)
# This is slow
m18_indices = c()
PTID = unique(adnimerge$PTID)
for (ptid in PTID) {
  #print(ptid)
  subject_lines = adnimerge[adnimerge['PTID'] == ptid, ]
  #print(rownames(subject_lines))
  subject_lines_before18m = subject_lines[subject_lines['M'] <= 18, ]
  #print(rownames(subject_lines_before18m))
  m18_indices = append(m18_indices, as.numeric(max(rownames(subject_lines_before18m))))
}

# Extract examination <= 18 months (and index by PTID)
adnimerge_m18 = adnimerge[m18_indices, ]
rownames(adnimerge_m18) <- adnimerge_m18$PTID

####################################
# Extract information for ADNI 510 #
####################################

# Open ADNI 510 groups
adni510_groups <- read.csv(INPUT_GROUPS)
adni510_groups$Group.ADNI <- factor(adni510_groups$Group.ADNI, levels=c("control", "MCInc", "MCIc", "AD"))
adni510_subjects = adni510_groups$PTID

# Merge & write
adni510_bl_groups = merge(adnimerge_bl, adni510_groups)
write.csv(adni510_bl_groups, OUTPUT_BL_EXAM)
adni510_m18_groups = merge(adnimerge_m18, adni510_groups)
write.csv(adni510_m18_groups, OUTPUT_M18_EXAM)

#########################################
# Scores for the whole ADNI 510 dataset #
#########################################

# Baseline scores
pdf(file=OUTPUT_BL_FIGS)
data_plots(adni510_bl_groups)
dev.off()

# m18 scores
pdf(file=OUTPUT_M18_FIGS)
data_plots(adni510_m18_groups)
dev.off()

######################################################
# Scores for MCIc & AD subjects that succeeded in CQ #
######################################################

# Open QC
qc <- read.csv(INPUT_QC_GRADE)
rownames(qc) = qc$PTID

# Merge (order is the same in those tables)
adni510_bl_groups_qc = merge(adni510_bl_groups, qc)
adni510_m18_groups_qc = merge(adni510_m18_groups, qc)

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
