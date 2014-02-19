# Plot some graphs
library(ADNIMERGE)
adnimerge <- adnimerge

INPUT_PATH="/neurospin/brainomics/2013_adni_preprocessing/clinic"
INPUT_GROUPS=file.path(INPUT_PATH, "groups.csv")

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

# Open ADNI groups
adni_510_groups <- read.csv(INPUT_GROUPS)
adni_510_groups$Group.article <- factor(adni_510_groups$Group.article, levels=c("control", "MCInc", "MCIc", "AD"))
adni_510_subjects = adni_510_groups$PTID

# Baseline scores
adnimerge_bl_groups = merge(adnimerge_bl, adni_510_groups)
pdf(file="baseline_scores.pdf")
boxplot(AGE~Group.article, data=adnimerge_bl_groups, main="Age")
boxplot(MMSE~Group.article, data=adnimerge_bl_groups, main="MMSE")
boxplot(ADAS11~Group.article, data=adnimerge_bl_groups, main="ADAS11")
boxplot(ADAS13~Group.article, data=adnimerge_bl_groups, main="ADAS13")
plot(adnimerge_bl_groups$ADAS11, adnimerge_bl_groups$ADAS13, main="ADAS13 as a function of ADAS11",
     xlab="ADAS11", ylab="ADAS13")
plot(adnimerge_bl_groups$ADAS11, adnimerge_bl_groups$MMSE, main="MMSE as a function of ADAS11",
    xlab="ADAS11", ylab="MMSE")
dev.off()

# m18 scores
adnimerge_m18_groups = merge(adnimerge_m18, adni_510_groups)
pdf(file="m18_scores.pdf")
boxplot(AGE~Group.article, data=adnimerge_m18_groups, main="Age")
boxplot(MMSE~Group.article, data=adnimerge_m18_groups, main="MMSE")
boxplot(ADAS11~Group.article, data=adnimerge_m18_groups, main="ADAS11")
boxplot(ADAS13~Group.article, data=adnimerge_m18_groups, main="ADAS13")
plot(adnimerge_m18_groups$ADAS11, adnimerge_m18_groups$ADAS13, main="ADAS13 as a function of ADAS11",
     xlab="ADAS11", ylab="ADAS13")
plot(adnimerge_m18_groups$ADAS11, adnimerge_m18_groups$MMSE, main="MMSE as a function of ADAS11",
     xlab="ADAS11", ylab="MMSE")
dev.off()
