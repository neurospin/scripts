# Extract baseline and last examination before 18 months from adnimerge for the ADNI 510 subjects
# Also merge with the groups and QC
# This a bit long and should only be done once.
library(ADNIMERGE)

INPUT_CLINIC_PATH="/neurospin/brainomics/2013_adni/clinic"
INPUT_GROUPS=file.path(INPUT_CLINIC_PATH, "adni510_groups.csv")

INPUT_ADNI510_CATI_PATH="/neurospin/cati/ADNI/ADNI_510"
INPUT_QC_PATH=file.path(INPUT_ADNI510_CATI_PATH,
                        "qualityControlSPM", "QC")
INPUT_QC_GRADE=file.path(INPUT_QC_PATH, "final_grade.csv")

OUTPUT_CLINIC_PATH=INPUT_CLINIC_PATH
OUTPUT_BL_GROUPS=file.path(OUTPUT_CLINIC_PATH, "adni510_bl_groups.csv")
OUTPUT_M18_GROUPS=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_groups.csv")
OUTPUT_M18_NONNULL_GROUPS=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_nonnull_groups.csv")
OUTPUT_BL_GROUPS_QC=file.path(OUTPUT_CLINIC_PATH, "adni510_bl_groups_qc.csv")
OUTPUT_M18_GROUPS_QC=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_groups_qc.csv")
OUTPUT_M18_NONNULL_GROUPS_QC=file.path(OUTPUT_CLINIC_PATH, "adni510_m18_nonnull_groups_qc.csv")

######################################################
# Extract baseline and m18 examinations in adnimerge #
######################################################
adnimerge <- adnimerge

# Baseline indexes (in adnimerge)
bl_indices = which(adnimerge$EXAMDATE == adnimerge$EXAMDATE.bl)

# Extract baseline examinations (and index by PTID)
adnimerge_bl = adnimerge[bl_indices, ]
rownames(adnimerge_bl) <- adnimerge_bl$PTID

# Find indexes of last examination before 18 months (this is how groups in Cuingnet et al. 2010 are found)
# and indexes of examination closest to 18 months with non-null MMSE
# This is slow
m18_indices = c()
m18_nonnull_indices = c()
PTID = unique(adnimerge$PTID)
for (ptid in PTID) {
  #print(ptid)
  subject_lines = adnimerge[adnimerge['PTID'] == ptid, ]
  #print(rownames(subject_lines))

  # Examinations before 18 months
  subject_lines_before18m = subject_lines[subject_lines['M'] <= 18, ]
  #print(rownames(subject_lines_before18m))
  m18_indices = append(m18_indices, as.numeric(max(rownames(subject_lines_before18m))))

  # Lines with non null MMSE
  subject_lines_nonnull = subject_lines[!is.na(subject_lines['MMSE']), ]
  M = subject_lines_nonnull$M
  M0 = 18
  m = M0
  found = FALSE
  direction=1
  pos_shift = neg_shift = 1
  while(!found) {
    #print(sprintf("Looking for M=%i", m))
    if (m %in% M) {
      #print("Found it!")
      # Subject 133_S_1031 has 2 m18 examinations so I use max (mimics the behavior for m18_indices)
      p = max(rownames(which(subject_lines_nonnull['M'] == m, arr.ind=TRUE)))
      #print(sprintf("Adding line %s", p))
      m18_nonnull_indices = append(m18_nonnull_indices, p)
      found = TRUE
    } else {
      #print("Not found")
      # Add or remove 6 months
      if (direction > 0) {
        m = M0 + pos_shift*6
        pos_shift = pos_shift+1
      } else {
        m = M0 - neg_shift*6
        neg_shift = neg_shift+1
        if (m==0) {
          warning(sprintf("Reached baseline for %s", ptid))
        }
      }
      # Alternate direction
      direction = -direction
    }
  }
}

# Extract examination <= 18 months (and index by PTID)
adnimerge_m18 = adnimerge[m18_indices, ]
rownames(adnimerge_m18) <- adnimerge_m18$PTID

# Extract examination closest to 18 months with non-null MMSE (and index by PTID)
adnimerge_m18_nonnull = adnimerge[m18_nonnull_indices, ]
rownames(adnimerge_m18_nonnull) <- adnimerge_m18_nonnull$PTID

####################################
# Extract information for ADNI 510 #
####################################

# Open ADNI 510 groups
adni510_groups <- read.csv(INPUT_GROUPS)
adni510_groups$Group.ADNI <- factor(adni510_groups$Group.ADNI, levels=c("control", "MCInc", "MCIc", "AD"))
adni510_subjects = adni510_groups$PTID

# Merge & write
adni510_bl_groups = merge(adnimerge_bl, adni510_groups)
write.csv(adni510_bl_groups, OUTPUT_BL_GROUPS, row.names=FALSE)
adni510_m18_groups = merge(adnimerge_m18, adni510_groups)
write.csv(adni510_m18_groups, OUTPUT_M18_GROUPS, row.names=FALSE)
adni510_m18_nonnull_groups = merge(adnimerge_m18_nonnull, adni510_groups)
write.csv(adni510_m18_nonnull_groups, OUTPUT_M18_NONNULL_GROUPS, row.names=FALSE)

#################
# Merge with QC #
#################

# Open QC
qc <- read.csv(INPUT_QC_GRADE)
rownames(qc) = qc$PTID

# Merge & write (order is the same in those tables)
adni510_bl_groups_qc = merge(adni510_bl_groups, qc)
write.csv(adni510_bl_groups_qc, OUTPUT_BL_GROUPS_QC, row.names=FALSE)
adni510_m18_groups_qc = merge(adni510_m18_groups, qc)
write.csv(adni510_m18_groups_qc, OUTPUT_M18_GROUPS_QC, row.names=FALSE)
adni510_m18_nonnull_groups_qc = merge(adni510_m18_nonnull_groups, qc)
write.csv(adni510_m18_nonnull_groups_qc, OUTPUT_M18_NONNULL_GROUPS_QC, row.names=FALSE)
