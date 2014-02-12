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
INPUT = "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed"
OUTPUT=INPUT
VALIDATION = "CV"


################################################################################################
## READ INPUT
################################################################################################
load(file=paste(INPUT, "/RESULTS_",VALIDATION,"_100PERMS.Rdata", sep=""))

#db = read_db(INPUT_DATA)

ALPHA = names(RESULTS[[TARGET]][["BASELINE"]])
PNZERO = names(RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]])
SEED = 11
PERM = 1

RES = NULL
vars = variables()
for(TARGET in vars$col_targets){
  #TARGET = "TMTB_TIME.M36"
b_y_pred_te = bni_y_pred_te = bc_y_pred_te = bcni_y_pred_te = y_true = id = c()

BASELINE = strsplit(TARGET, "[.]")[[1]][1]
for(i in 5){
b = RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["GLM"]]
bni = RESULTS[[TARGET]][["BASELINE+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]
bc = RESULTS[[TARGET]][["BASELINE+CLINIC"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]
bcni = RESULTS[[TARGET]][["BASELINE+CLINIC+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]

id = c(id, b$D_te$ID)
y_true = c(y_true, b$y_true_te)
b_y_pred_te= c(b_y_pred_te, b$y_pred_te)
bni_y_pred_te= c(bni_y_pred_te, bni$y_pred_te)
bc_y_pred_te= c(bc_y_pred_te, bc$y_pred_te)
bcni_y_pred_te= c(bcni_y_pred_te, bcni$y_pred_te)
}
  
RES=rbind(RES,
data.frame(TARGET=TARGET, PREDICTORS="BASELINE",
  ID = id,
  M0      = b$D_te[, BASELINE],
  M0_true = y_true,
  M36_pred    = b_y_pred_te,
  M36_err     = b_y_pred_te - y_true,
  M36_err_abs     = abs(b_y_pred_te - y_true)),

data.frame(TARGET=TARGET, PREDICTORS="BASELINE+NIGLOB",
  ID = id,
  M0      = b$D_te[, BASELINE],
  M0_true = y_true,
  M36_pred = bni_y_pred_te,
  M36_err  = bni_y_pred_te - y_true,
  M36_err_abs  = abs(bni_y_pred_te - y_true)),

data.frame(TARGET=TARGET, PREDICTORS="BASELINE+CLINIC",
   ID = id,
   M0      = b$D_te[, BASELINE],
   M0_true = y_true,
   M36_pred    = bc_y_pred_te,
   M36_err     = bc_y_pred_te - y_true,
   M36_err_abs     = abs(bc_y_pred_te - y_true)),

data.frame(TARGET=TARGET, PREDICTORS="BASELINE+CLINIC+NIGLOB",
           ID = id,
           M0   = b$D_te[, BASELINE],
           M0_true = y_true,
           M36_pred = bcni_y_pred_te,
           M36_err  = bcni_y_pred_te - y_true,
           M36_err_abs  = abs(bcni_y_pred_te - y_true)))
}

write.csv(RES, paste(OUTPUT, "error_pred_M36_by_M0.csv", sep="/"), row.names=FALSE)

# PLOT PRED ERR AT M36 in function M0
pdf(paste(OUTPUT, "error_pred_M36_by_M0.pdf", sep="/"), width=10, height=7)
err_m36_by_m0 = ggplot(RES, aes(x = M0, y = M36_err)) + geom_point(aes(colour=PREDICTORS), alpha=1) + stat_smooth(aes(colour=PREDICTORS, fill = PREDICTORS), alpha=.2) + facet_wrap(~TARGET, scale="free")
print(err_m36_by_m0)
dev.off()



