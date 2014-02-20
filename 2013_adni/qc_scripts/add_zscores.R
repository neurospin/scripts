
# d from 01qc_tissues-volumes_spm-segmentation.R script

p = read.table("/home/mp210984/tarball/git/neurospin/scripts/2013_adni/scripts/subjects_to_be_checked.txt", header=TRUE)
for(i in 1:dim(p)[1]) { p$ZscoreGM[i] = d$ZscoreGM[which(d$subject == as.character(p[i, 2]))]}
for(i in 1:dim(p)[1]) { p$ZscoreWM[i] = d$ZscoreWM[which(d$subject == as.character(p[i, 2]))]}
for(i in 1:dim(p)[1]) { p$ZscoreCSF[i] = d$ZscoreCSF[which(d$subject == as.character(p[i, 2]))]}
for(i in 1:dim(p)[1]) { p$Zscore_vox_out_of_mask[i] = d$Zscore_vox_out_of_mask[which(d$subject == as.character(p[i, 2]))]}

write.table(p, "/home/mp210984/tarball/git/neurospin/scripts/2013_adni/scripts/subjects_to_be_checked_with_zscores.txt", sep= " ", quote=FALSE)
