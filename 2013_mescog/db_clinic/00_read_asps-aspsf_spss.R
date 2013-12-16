setwd("/home/edouard/data/2013_mescog/clinic")
library(foreign)
#D=read.spss("klin_daten_30092012.sav", to.data.frame =TRUE)
#write.table(D, "klin_daten_30092012.csv", sep="\t", row.names = FALSE)


D=read.spss("ASPFS_klinVariables_20130711.sav", to.data.frame =TRUE)
write.table(D, "ASPFS_klinVariables_20130711.csv", sep="\t", row.names = FALSE)

D=read.spss("klin_daten_30092012.sav", to.data.frame =TRUE)
write.table(D, "klin_daten_30092012.csv", sep="\t", row.names = FALSE)

D=read.spss("ASPS_klinVariables_20130806.sav", to.data.frame =TRUE)
write.table(D, "ASPS_klinVariables_20130806.csv", sep="\t", row.names = FALSE)


