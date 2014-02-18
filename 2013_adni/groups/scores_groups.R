library(ADNIMERGE)
adnimerge <- adnimerge

INPUT_PATH="/neurospin/brainomics/2013_adni_preprocessing/clinic"
INPUT_GROUPS=file.path(INPUT_PATH, "groups.csv")

# Open ADNI groups
adni_groups <- read.csv(INPUT_GROUPS)
adni_groups$Group.article <- factor(adni_groups$Group.article, levels=c("control", "MCInc", "MCIc", "AD"))

# Merge tables
adnimerge_groups <- merge(adnimerge, adni_groups)

pdf(file="scores.pdf")
boxplot(AGE~Group.article, data=adnimerge_groups, main="Age")
boxplot(MMSE~Group.article, data=adnimerge_groups, main="MMSE")
boxplot(ADAS11~Group.article, data=adnimerge_groups, main="ADAS11")
boxplot(ADAS13~Group.article, data=adnimerge_groups, main="ADAS13")
plot(adnimerge_groups$ADAS11, adnimerge_groups$ADAS13, main="ADAS13 as a function of ADAS11",
     xlab="ADAS11", ylab="ADAS13")
plot(adnimerge_groups$ADAS11, adnimerge_groups$MMSE, main="MMSE as a function of ADAS11",
     xlab="ADAS11", ylab="MMSE")
dev.off()
