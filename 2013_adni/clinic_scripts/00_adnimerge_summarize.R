##############################################################################################################################
#
# Read ADNIMERGE and produce a simplified csv file with one line per subject containing:
# - all baseline information (suffix .bl and no suffix should be the same)
# - 2 years (suffix .m24)
# - 800 days (suffix .d800)
# - last known information (suffix .last)
# - CONV_TO_MCI: CONVERTION in days to MCI from bl
# - CONV_TO_AD: CONVERTION in days to AD from bl
#
##############################################################################################################################


# INPUT: "ADNIMERGE_0.0.1.tar.gz"
if(FALSE){
install.packages("Hmisc")
install.packages("lubridate")
install.packages("ADNIMERGE_0.0.1.tar.gz", repos = NULL, type = "source")
}

library(lubridate)
library(ADNIMERGE)

WD = "/neurospin/brainomics/2013_adni/clinic"
OUTPUT = paste(WD, "adnimerge_simplified.csv", sep="/")


setwd(WD)
# http://adni.bitbucket.org/
length(adnimerge$PTID) # 11074
length(unique(adnimerge$PTID)) # 1736

##############################################################################################################################
## RM NA DX
##############################################################################################################################
D = adnimerge
table(adnimerge$DX)
#  NL             MCI        Dementia       NL to MCI MCI to Dementia  NL to Dementia       MCI to NL Dementia to MCI  Dementia to NL 
#2290            3306            1550              76             281               3              57               6               0

D = D[!is.na(D$DX), ]
levels(D$DX.bl)
D$DX.bl = as.character(D$DX.bl)

##############################################################################################################################
## Recoding
##############################################################################################################################
#"CN"   "SMC"  "EMCI" "LMCI" "AD"
# SMC: Subjective memory complaints
# CN: cognitively normal
# EMCI early MCI
# LMCI late MCI
D$DX.bl <- gsub("EMCI", "MCI", D$DX.bl)
D$DX.bl <- gsub("LMCI", "MCI", D$DX.bl)
# CTL = CN + SMC
D$DX.bl <- gsub("SMC", "CTL", D$DX.bl)
D$DX.bl <- gsub("CN", "CTL", D$DX.bl)
unique(D$DX.bl)
D$DX.bl = as.factor(D$DX.bl)
table(D$DX.bl)
#AD   CN  MCI  SMC 
#1078 2164 4123  202

levels(D$DX)
#"NL"  "MCI" "Dementia" "NL to MCI"  "MCI to Dementia" "NL to Dementia"  "MCI to NL" "Dementia to MCI" "Dementia to NL" 
# NL: cognitively normal subjects
D$DX = as.character(D$DX)
D$DX <- gsub("NL to ", "", D$DX)
D$DX <- gsub("MCI to ", "", D$DX)
D$DX <- gsub("Dementia to ", "", D$DX)
D$DX <- gsub("NL", "CTL", D$DX)
D$DX <- gsub("Dementia", "AD", D$DX)
table(D$DX)
# AD  CTL  MCI 
# 1834 2347 3388 

cols_not_bl = colnames(D)[grep(".bl", colnames(D), invert=TRUE)]
##############################################################################################################################
## Simplify table: one line per subject: all bl + 2 years (suffix .m24) + 800 days (suffix .d800) + last known (suffix .last)
## CONVERTION in days CONV_TO_MCI and CONV_TO_AD
##############################################################################################################################
simple = NULL
for(id in unique(D$PTID)){
  #id ="072_S_5207"
  # id = unique(D$PTID)[1]
  d = D[D$PTID == id, ]
  if(nrow(d)>=1){
  #cat(id)
  idx_last = which(max(d$EXAMDATE) == d$EXAMDATE)
  idx_bl =  which(min(d$EXAMDATE) == d$EXAMDATE)
  days_since_bl = as.numeric((d$EXAMDATE - d$EXAMDATE.bl[1]))
  idx_2yrs = which(min(abs(days_since_bl - 2 * 365)) == abs(days_since_bl - 2 * 365) )
  idx_800days = which(min(abs(days_since_bl - 800)) == abs(days_since_bl - 800) )
  #idx_3yr = which(min(abs(days_since_bl - 3 * 365)) == abs(days_since_bl - 3 * 365) )
  #idx_ad = sum(d$DX == "AD")[1]
  #if(d$DX.bl != d[idx_bl, "DX"])cat(id)
  d_2yrs = d[idx_2yrs, cols_not_bl]
  colnames(d_2yrs) = paste(colnames(d_2yrs), "m24", sep=".")
  d_800days = d[idx_800days, cols_not_bl]
  colnames(d_800days) = paste(colnames(d_800days), "d800", sep=".")
  d_last = d[idx_last, cols_not_bl]
  colnames(d_last) = paste(colnames(d_last), "last", sep=".")
  tuple = cbind(d[idx_bl, ], d_2yrs, d_800days, d_last)
  #tuple = data.frame(d[idx_bl, ], DX.last=d[idx_last, "DX"], MMSE.2yrs=d[idx_2yrs, "MMSE"],  MMSE.800days=d[idx_800days, "MMSE"])
  if(is.na(tuple$DX.bl)) tuple$DX.bl=tuple$DX
  if(tuple$DX.bl != tuple$DX)cat(id,"baseline DX mismatch\n")
  # look for convertion
  CONV_TO_MCI = -1
  CONV_TO_AD = -1
  if(length(unique(d$DX)>1)){
    for(i in 1:nrow(d)){
      #days_since_bl = as.numeric((d$EXAMDATE[i] - d$EXAMDATE[1]))
      # MCI that goes back to CTL: reset
      if((CONV_TO_MCI > 0) && (d$DX[i] == "CTL")) CONV_TO_MCI = -1
      # AD that goes back to MCI: reset
      if((CONV_TO_AD > 0) && (d$DX[i] != "AD")) CONV_TO_AD = -1
      if((CONV_TO_MCI < 0) && (d$DX[i] == "MCI")) CONV_TO_MCI = days_since_bl[i]
      if((CONV_TO_AD < 0) && (d$DX[i] == "AD")) CONV_TO_AD = days_since_bl[i]
    }
  }
  tuple$CONV_TO_MCI = CONV_TO_MCI
  tuple$CONV_TO_AD = CONV_TO_AD
  simple = rbind(simple, tuple)
  }
}

# QC: remove subject where s$DX.bl != s$DX
s = simple
pb = s[s$DX.bl != s$DX, c("PTID", "DX.bl","DX")]
print(pb)
# PTID DX.bl  DX
# 1764  021_S_0332   MCI  AD
# 7027  094_S_2201   MCI CTL
# 9336  100_S_4512   MCI CTL
# 10357 003_S_4892   MCI  AD
# Remove them

simple = simple[s$DX.bl == s$DX, ]
write.csv(simple, OUTPUT, row.names = FALSE)

table(simple$DX)
#AD CTL MCI 
#339 520 865
table(simple$DX.last)
#CTL  AD MCI 
#495 605 624 
