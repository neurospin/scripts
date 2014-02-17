#display.brewer.pal(6, "Paired")
palette = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), sep="/")
if (!file.exists(OUTPUT)) dir.create(OUTPUT)
VALIDATION = "CV"

source(paste(SRC,"utils.R",sep="/"))

db = read_db(INPUT_DATA)
D = db$DB

################################################################################################
## Find best cut-off on LLV and BPF to explain M36-M0
################################################################################################
D$TMTB_TIME_EVOL = (D$TMTB_TIME.M36 - D$TMTB_TIME)
D$MDRS_TOTAL_EVOL = (D$MDRS_TOTAL.M36 - D$MDRS_TOTAL)
D$MRS_EVOL = (D$MRS.M36 - D$MRS)
D$MMSE_EVOL = (D$MMSE.M36 - D$MMSE)

targets=c("TMTB_TIME_EVOL", "MDRS_TOTAL_EVOL", "MRS_EVOL", "MMSE_EVOL")

# CUT-OFF LLV ----------------------------------------------------------------------------------
LLV_CUTOFF=NULL
#Y = as.matrix(D[, targets] / D[, c("TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE")])
Y = as.matrix(D[, targets])
for(llv_cutoff in sort(unique(D$LLV))){
  for(TARGET in targets){
  #TARGET = "MMSE_EVOL" TARGET = "MMSE_EVOL"; TARGET = "TMTB_TIME_EVOL"
  #llv_cutoff = 1500#1225.247
  where = D$LLV > llv_cutoff
  #D$LLV_G = as.factor(where)
  try({
  ttest = t.test(D[!where , TARGET], D[where , TARGET]);
  dmeans = ttest$estimate[2] - ttest$estimate[1];
  #formula = formula(paste(TARGET, "~", "LLV_G"))
  #s = anova(aov(formula,data=D))
  LLV_CUTOFF=rbind(LLV_CUTOFF, data.frame(TARGET=TARGET, llv_cutoff=llv_cutoff, tstat=ttest$statistic, pval=ttest$p.value, dmeans=dmeans));
  })
  }
#   try({
#   s = summary(manova(Y~D$LLV_G))$stats;
#   STAT_ALL=rbind(STAT_ALL, data.frame(llv_cutoff=llv_cutoff, fstat=s[1, "approx F"], pval= s[1, "Pr(>F)"]))})
}

LLV_CUTOFF$SIGNIFICANT = as.factor(LLV_CUTOFF$pval < 0.05)

pdf("/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed/M36-M0~cutoff_LLV.pdf")

p=ggplot(LLV_CUTOFF, aes(x=llv_cutoff, y = dmeans)) + 
  geom_point(aes(colour=SIGNIFICANT)) +
  facet_wrap(~TARGET, scales="free") +
  ggtitle("mean(M36-M0; LLV>cutoff) - mean(M36-M0; LLV<=cutoff) ")
print(p)
dev.off()

# => CUTOFF LLV = 1500 

# CUT-OFF BPF ----------------------------------------------------------------------------------
BPF_CUTOFF=NULL
#Y = as.matrix(D[, targets] / D[, c("TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE")])
Y = as.matrix(D[, targets])
for(llv_cutoff in sort(unique(D$BPF))){
  for(TARGET in targets){
    #TARGET = "MMSE_EVOL" TARGET = "MMSE_EVOL"; TARGET = "TMTB_TIME_EVOL"
    #llv_cutoff = 1500#1225.247
    where = D$BPF < llv_cutoff
    #D$LLV_G = as.factor(where)
    try({
      ttest = t.test(D[!where , TARGET], D[where , TARGET]);
      dmeans = ttest$estimate[2] - ttest$estimate[1];
      #formula = formula(paste(TARGET, "~", "LLV_G"))
      #s = anova(aov(formula,data=D))
      BPF_CUTOFF=rbind(BPF_CUTOFF, data.frame(TARGET=TARGET, llv_cutoff=llv_cutoff, tstat=ttest$statistic, pval=ttest$p.value, dmeans=dmeans));
    })
  }
}

BPF_CUTOFF$SIGNIFICANT = as.factor(BPF_CUTOFF$pval < 0.05)

pdf("/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed/M36-M0~cutoff_BPF.pdf")

p=ggplot(BPF_CUTOFF, aes(x=llv_cutoff, y = dmeans)) + 
  geom_point(aes(colour=SIGNIFICANT)) +
  facet_wrap(~TARGET, scales="free") +
  ggtitle("mean(M36-M0; BPF<cutoff) - mean(M36-M0; BPF>=cutoff) ")
print(p)
dev.off()

# => CUT OFF PAS CLAIR

################################################################################################
## Partition RULE: M36 ~ BASELINE color by group
################################################################################################
rpart_mmse=rpart(MMSE_EVOL~LLV+BPF,data=D)
prune(rpart_mmse, cp=.015)


grp = rep(NA, nrow(D))
#grp[D$LLV >= 1592] = "Lacunes"
#grp[D$BPF < 0.773] = "Atrophy"
grp[D$BPF < 0.75] = "Atrophy"
grp[D$LLV >= 1500] = "Lacunes"

D$GRP = as.factor(grp)
D$MDRS_TOTAL.M36[D$MDRS_TOTAL.M36 == min(D$MDRS_TOTAL.M36, na.rm=T)]=NA
D2 = NULL
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  D2 = rbind(D2, data.frame(VAR=BASELINE, M36=D[, TARGET], M0=D[, BASELINE], SITE=D[, "SITE"], ID=D[, "ID"], GRP=D[, "GRP"]))
}

pdf("/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed/M36~M0_cutoff_LLV-BPF.pdf")

Dg = D2[!is.na(D2$GRP), ]
Dng = D2[is.na(D2$GRP), ]

p = ggplot(D2, aes(x = M0, y = M36)) + 
  geom_point(data=Dng, aes(colour=GRP), alpha=.4, position = "jitter") +
  geom_point(data=Dg, aes(colour=GRP), alpha=1)+#, position = "jitter") +
  geom_abline(linetype="dotted") + 
  #stat_smooth(formula=y~x-1, method="lm", aes(colour=GRP))+
  facet_wrap(~VAR, scales="free") +
  ggtitle("M36 ~ M0")

print(p)
dev.off()

# BUILD RULES

## MMSE
#tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
tree = c("LLV>1500", "BPF<0.75")
rpart_mmse=rpart(MMSE_EVOL~LLV+BPF,data=D)
prune(rpart_mmse, cp=.015)
1) root 198 1057.44400  0.2556633  
2) LLV>=1964.283 7   93.71429 -6.4285710 *
  3) LLV< 1964.283 191  639.51460  0.5006352  
6) BPF< 0.8187689 66  271.72180  0.1003232 *
  7) BPF>=0.8187689 125  351.63200  0.7120000 *
  
## MMSE
  