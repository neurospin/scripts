WD="/neurospin/mescog/proj_wmh_patterns/PCA"
setwd(WD)
OUTPUT_CSV = "pc_learn_fr_x_clinic_mulm.csv"

pc = read.csv("pc_learn_fr.csv")
clin = read.csv("dataset_clinic_niglob_20140121.csv")
clin$ID = as.integer(sub("CAD_", "", clin$ID))
D = merge(clin, pc, by=c("ID", "Site"), all=TRUE)

D$TMTB_TIME_DELTA = D$TMTB_TIME.M36 - D$TMTB_TIME
D$MDRS_TOTAL_DELTA = D$MDRS_TOTAL.M36 - D$MDRS_TOTAL
D$MRS_DELTA = D$MRS.M36 - D$MRS
D$BARTHEL_DELTA = D$BARTHEL.M36 - D$BARTHEL
D$MMSE_DELTA = D$MMSE.M36 - D$MMSE
D$LLV_DELTA = D$LLV.M36 - D$LLV
D$LLcount_DELTA = D$LLcount.M36 - D$LLcount
D$MBcount_DELTA = D$MBcount.M36 - D$MBcount

Dfr = D[D$SITE == "FR", ]
DGE = D[D$SITE == "GR", ]

TARGETS = c(
"TMTB_TIME_DELTA", "TMTB_TIME.M36" ,"TMTB_TIME",
"MDRS_TOTAL_DELTA", "MDRS_TOTAL.M36", "MDRS_TOTAL",
"MRS_DELTA", "MRS.M36", "MRS",
"BARTHEL_DELTA", "BARTHEL.M36", "BARTHEL",
"MMSE_DELTA", "MMSE.M36", "MMSE",
"AGE_AT_INCLUSION", "SEX", "EDUCATION",
"LLV", "LLVn","LLcount","WMHV","WMHVn","MBcount","BPF", "BRAINVOL",
"LLV.M36","LLcount.M36","MBcount.M36",
"LLV_DELTA", "LLcount_DELTA", "MBcount_DELTA")

RESULTS = NULL
for(TARGET in TARGETS){
  res = rbind(
  data.frame(Var=TARGET, PC=1, Site="FR", as.list(summary(lm(formula(paste(TARGET,"~PC1")), data=Dfr))$coefficients[2, ]), check.names=F),
  data.frame(Var=TARGET, PC=2, Site="FR", as.list(summary(lm(formula(paste(TARGET,"~PC2")), data=Dfr))$coefficients[2, ]), check.names=F),
  data.frame(Var=TARGET, PC=3, Site="FR", as.list(summary(lm(formula(paste(TARGET,"~PC3")), data=Dfr))$coefficients[2, ]), check.names=F),
  data.frame(Var=TARGET, PC=1, Site="GE", as.list(summary(lm(formula(paste(TARGET,"~PC1")), data=DGE))$coefficients[2, ]), check.names=F),
  data.frame(Var=TARGET, PC=2, Site="GE", as.list(summary(lm(formula(paste(TARGET,"~PC2")), data=DGE))$coefficients[2, ]), check.names=F),
  data.frame(Var=TARGET, PC=3, Site="GE", as.list(summary(lm(formula(paste(TARGET,"~PC3")), data=DGE))$coefficients[2, ]), check.names=F))
  colnames(res) = c("Var", "PC", "Site", "Estimate", "Std.Error", "tval", "pval")
  res = cbind(res, r2=NA)
  m = summary(lm(formula(paste(TARGET,"~PC2+PC3")), data=Dfr))
  pval = 1 - pf(m$fstatistic[1], m$fstatistic[2], m$fstatistic[3])
  fr = data.frame(Var=TARGET, PC=23, Site="FR", Estimate=NA, Std.Error=NA, tval=NA, pval=pval, r2=m$r.squared)
  m = summary(lm(formula(paste(TARGET,"~PC2+PC3")), data=DGE))
  pval = 1 - pf(m$fstatistic[1], m$fstatistic[2], m$fstatistic[3])
  ge = data.frame(Var=TARGET, PC=23, Site="GE", Estimate=NA, Std.Error=NA, tval=NA, pval=pval, r2=m$r.squared)
  res = rbind(res, fr, ge)
  if(is.null(RESULTS)) RESULTS = res else RESULTS = rbind(RESULTS, res)
}


rownames(RESULTS) = 1:(nrow(RESULTS))
RESULTS = RESULTS[order(RESULTS$pval), ]

write.csv(RESULTS, OUTPUT_CSV, row.names=FALSE)

stat = RESULTS[RESULTS$PC != 1 & RESULTS$pval < 0.05,]
stat = stat[order(stat$pval),]
