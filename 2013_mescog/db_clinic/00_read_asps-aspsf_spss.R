setwd("/neurospin/mescog/clinic")
library(foreign)
#D=read.spss("klin_daten_30092012.sav", to.data.frame =TRUE)
#write.table(D, "klin_daten_30092012.csv", sep="\t", row.names = FALSE)


D=read.spss("ASPS_Family_klinVariables_20131218.sav", to.data.frame =TRUE)
write.table(D, "ASPS_Family_klinVariables_20131218.csv", sep="\t", row.names = FALSE)

D=read.spss("klin_daten_30092012.sav", to.data.frame =TRUE)
write.table(D, "klin_daten_30092012.csv", sep="\t", row.names = FALSE)

D=read.spss("ASPS_klinVariables_20131218.sav", to.data.frame =TRUE)
write.table(D, "ASPS_klinVariables_20131218.csv", sep="\t", row.names = FALSE)


