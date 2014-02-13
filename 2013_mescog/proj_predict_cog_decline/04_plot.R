require(ggplot2)
require(glmnet)
require(reshape)
library(RColorBrewer)
library(plyr)

#display.brewer.pal(6, "Paired")
palette = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
source(paste(SRC,"utils.R",sep="/"))

BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140128_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140204_nomissing_BPF-LLV_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT = "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed"
OUTPUT=INPUT


# PLOT PRED ERR AT M36 in function M0 ---------------------------
ERR_pred = read.csv(paste(INPUT, "error_pred_M36_by_M0.csv", sep="/"))

pdf(paste(OUTPUT, "error_pred_M36_by_M0.pdf", sep="/"), width=10, height=7)
err_m36_by_m0 = ggplot(ERR_pred, aes(x = M0, y = M36_err)) + geom_point(aes(colour=PREDICTORS), alpha=1) + stat_smooth(aes(colour=PREDICTORS, fill = PREDICTORS), alpha=.2) + facet_wrap(~TARGET, scale="free")
print(err_m36_by_m0)
dev.off()

#ERR_all = read.csv(paste(INPUT, "error_refitallglm_M36_by_M0.csv", sep="/"))
#ERR_all = read.csv(paste(INPUT, "error_refitallglm-nointer_M36_by_M0.csv", sep="/"))
ERR_all = read.csv(paste(INPUT, "error_refitall_rpart_M36_by_M0.csv", sep="/"))

#pdf(paste(OUTPUT, "error_refitallglm_M36_by_M0.pdf", sep="/"), width=10, height=7)
#pdf(paste(OUTPUT, "error_refitallglm-nointer_M36_by_M0.pdf", sep="/"), width=10, height=7)
pdf(paste(OUTPUT, "error_refitall_rparts_M36_by_M0.pdf", sep="/"), width=10, height=7)
err_m36_by_m0 = ggplot(ERR_all, aes(x = M0, y = M36_err)) + geom_point(aes(colour=PREDICTORS), alpha=.1, position = position_jitter(w = 0.05, h = 0.05)) + stat_smooth(aes(colour=PREDICTORS, fill = PREDICTORS), alpha=.2) + facet_wrap(~TARGET, scale="free")
print(err_m36_by_m0)
dev.off()


# PLOT PRED DIFF ERR AT M36 in function M0 ---------------------------

#ERR = ERR_pred; out_file = "error_diff_pred_M36_by_M0.pdf"
#ERR = ERR_all; out_file = "error_diff_refitallglm_M36_by_M0.pdf"
#ERR = ERR_all; out_file = "error_diff_refitallglm-nointer_M36_by_M0.pdf"
ERR = ERR_all; out_file = "error_diff_refitall_rpart_M36_by_M0.pdf"


bni_vs_b = merge(ERR[ERR$PREDICTORS == "BASELINE+NIGLOB",], ERR[ERR$PREDICTORS == "BASELINE",] , by = c("ID", "M36_true", "TARGET" ,"M0"), suffixes=c("_bni", "_b"))
bni_vs_b$diff_err = abs(bni_vs_b$M36_err_b) - abs(bni_vs_b$M36_err_bni)

bni_vs_b$COMP = "ERR_B-BNI"
DIFF = bni_vs_b[, c("ID", "TARGET", "M0", "COMP", "diff_err")]

pdf(paste(OUTPUT, out_file, sep="/"), width=10, height=7)
err_diff_m36_by_m0 = ggplot(DIFF, aes(x = M0, y = diff_err)) + geom_point(alpha=.5) + stat_smooth(alpha=.2) + facet_wrap(~TARGET, scale="free")
print(err_diff_m36_by_m0)
dev.off()


if(sum(ERR$PREDICTORS == "BASELINE+CLINIC+NIGLOB")>0){
  bcni_vs_bc = merge(ERR[ERR$PREDICTORS == "BASELINE+CLINIC+NIGLOB",], ERR[ERR$PREDICTORS == "BASELINE+CLINIC",] , by = c("ID", "M36_true", "TARGET" ,"M0"), suffixes=c("_bcni", "_bc"))
  bcni_vs_bc$diff_err = abs(bcni_vs_bc$M36_err_bc) - abs(bcni_vs_bc$M36_err_bcni)
  bcni_vs_bc$COMP = "ERR_BC-BCNI"
  DIFF = rbind(DIFF, bcni_vs_bc[, c("ID", "TARGET", "M0", "COMP", "diff")])
  pdf(paste(OUTPUT, out_file, sep="/"), width=10, height=7)
  err_diff_m36_by_m0 = ggplot(DIFF, aes(x = M0, y = diff_err)) + geom_point(aes(colour=COMP), alpha=1) + stat_smooth(aes(colour=COMP, fill = COMP), alpha=.2) + facet_wrap(~TARGET, scale="free")
  print(err_diff_m36_by_m0)
  dev.off()
}



ddply(DIFF, as.quoted(c("COMP" ,"TARGET")), summarise, prop_increase=sum(diff>0)/length(diff))

ddply(DIFF, as.quoted(c("COMP" ,"TARGET")), summarise, mean_diff=mean(diff_err))
