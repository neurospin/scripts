WD="/neurospin/mescog/proj_wmh_patterns/PCA"
setwd(WD)
OUTPUT_MULM = "mulm_PCs_vs_clinic"
OUTPUT_FIG_PCs_vs_clinic = "PCs_vs_clinic"

pc = read.csv("pc_learn_fr.csv")
clin = read.csv("dataset_clinic_niglob_20140121.csv")
clin$ID = as.integer(sub("CAD_", "", clin$ID))
D = merge(clin, pc, by=c("ID", "SITE"), all=TRUE)

D$TMTB_TIME_DELTA = D$TMTB_TIME.M36 - D$TMTB_TIME
D$MDRS_TOTAL_DELTA = D$MDRS_TOTAL.M36 - D$MDRS_TOTAL
D$MRS_DELTA = D$MRS.M36 - D$MRS
D$BARTHEL_DELTA = D$BARTHEL.M36 - D$BARTHEL
D$MMSE_DELTA = D$MMSE.M36 - D$MMSE
D$LLV_DELTA = D$LLV.M36 - D$LLV
D$LLcount_DELTA = D$LLcount.M36 - D$LLcount
D$MBcount_DELTA = D$MBcount.M36 - D$MBcount

levels(D$SITE) = c("FR", "GE")

Dfr = D[D$SITE == "FR", ]
DGE = D[D$SITE == "GE", ]

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

write.csv(RESULTS, OUTPUT_MULM, row.names=FALSE)

stat = RESULTS[RESULTS$PC != 1 & RESULTS$pval < 0.05,]
stat = stat[order(stat$pval),]


##########################################################################################################################
TARGETS = c(
"TMTB_TIME",
 #"TMTB_TIME_DELTA", "TMTB_TIME.M36" ,
 "MDRS_TOTAL",
#"MDRS_TOTAL_DELTA", "MDRS_TOTAL.M36",
  "MRS",
 #"MRS_DELTA", "MRS.M36",
  #"BARTHEL_DELTA", "BARTHEL.M36", "BARTHEL",
  "MMSE",
  #"MMSE_DELTA", "MMSE.M36", 
  "AGE_AT_INCLUSION", "SEX", "EDUCATION",
 "LLcount", #"LLcount_DELTA", "LLcount.M36",
  "LLV", #"LLV.M36", "LLV_DELTA", "LLVn"
 "WMHV",#"WMHVn",
  "MBcount", #"MBcount.M36", "MBcount_DELTA"
  "BPF"
  #"BRAINVOL",
)
library(reshape)
library(ggplot2)
 
D2 = D[c("ID","SITE","PC1","PC2", "PC3", TARGETS)]
Dm = melt(D2, id=c("ID","SITE","PC1","PC2","PC3"))

svg(paste(OUTPUT_FIG_PCs_vs_clinic, "PC1.svg", sep="_"), width = 10, height = 7)
p = ggplot(Dm, aes(x=PC1, y=value)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter")  +  facet_wrap(~variable, scales="free") + scale_color_manual(values=c("blue", "red"))
p = p + stat_smooth(formula=y~x, method="lm", aes(colour=SITE))
print(p)

svg(paste(OUTPUT_FIG_PCs_vs_clinic, "PC2.svg", sep="_"), width = 10, height = 7)
p = ggplot(Dm, aes(x=PC2, y=value)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter")  +  facet_wrap(~variable, scales="free") + scale_color_manual(values=c("blue", "red"))
p = p + stat_smooth(formula=y~x, method="lm", aes(colour=SITE))
print(p)
dev.off()

svg(paste(OUTPUT_FIG_PCs_vs_clinic, "PC3.svg", sep="_"), width = 10, height = 7)
p = ggplot(Dm, aes(x=PC3, y=value)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter")  +  facet_wrap(~variable, scales="free") + scale_color_manual(values=c("blue", "red"))
p = p + stat_smooth(formula=y~x, method="lm", aes(colour=SITE))
print(p)
dev.off()

TARGETS = c(
  "MRS.M36",
"MBcount.M36"
)
D2 = D[c("ID","SITE","PC1","PC2", "PC3", TARGETS)]
Dm = melt(D2, id=c("ID","SITE","PC1","PC2","PC3"))

svg("PC2_vs_MRSM36-MBcount.36.svg", width = 7, height = 4)
p = ggplot(Dm, aes(x=PC2, y=value)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter")  +  facet_wrap(~variable, scales="free") + scale_color_manual(values=c("blue", "red"))
p = p + stat_smooth(formula=y~x, method="lm", aes(colour=SITE))
print(p)
dev.off()


TARGETS = c(
  "MRS",
  "MMSE", 
  "MDRS_TOTAL"
)
D2 = D[c("ID","SITE","PC1","PC2", "PC3", TARGETS)]
Dm = melt(D2, id=c("ID","SITE","PC1","PC2","PC3"))

svg("PC3_vs_MRS-MMSE-MDRS_TOTAL.svg", width = 10.5, height = 4)
p = ggplot(Dm, aes(x=PC3, y=value)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter")  +  facet_wrap(~variable, scales="free") + scale_color_manual(values=c("blue", "red"))
p = p + stat_smooth(formula=y~x, method="lm", aes(colour=SITE))
print(p)
dev.off()


