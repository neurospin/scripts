# Extract baseline and last examination before 18 months from adnimerge for the ADNI 510 subjects
# Also merge the groups
# This a bit long and should only be done once.
library(ADNIMERGE)

INPUT_CLINIC_PATH="/neurospin/brainomics/2013_adni/clinic"
INPUT_GROUPS=file.path(INPUT_CLINIC_PATH, "adni510_groups.csv")

OUTPUT_CLINIC_PATH=INPUT_CLINIC_PATH
OUTPUT_BL_EXAM=file.path(OUTPUT_CLINIC_PATH, "adni510_bl_groups.csv")
OUTPUT_M18_EXAM=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_groups.csv")

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
write.csv(adni510_bl_groups, OUTPUT_BL_EXAM, row.names=FALSE)
adni510_m18_groups = merge(adnimerge_m18, adni510_groups)
write.csv(adni510_m18_groups, OUTPUT_M18_EXAM, row.names=FALSE)
