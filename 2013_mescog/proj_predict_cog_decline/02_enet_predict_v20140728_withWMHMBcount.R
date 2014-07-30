#require(ggplot2)
require(glmnet)
#require(reshape)
#library(RColorBrewer)
#library(plyr)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
#OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "enet", "M36", sep="/")
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "enet", "CHANGE", sep="/")

if (!file.exists(OUTPUT)) dir.create(OUTPUT)
VALIDATION = "CV"
#VALIDATION = "FR-GE"
RM_TEST_OUTLIERS = FALSE

source(paste(SRC,"utils.R",sep="/"))

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
db$DB

db$DB$TMTB_TIME.CHANGE = (db$DB$TMTB_TIME.M36 - db$DB$TMTB_TIME)
db$DB$MDRS_TOTAL.CHANGE = (db$DB$MDRS_TOTAL.M36 - db$DB$MDRS_TOTAL)
db$DB$MRS.CHANGE = (db$DB$MRS.M36 - db$DB$MRS)
db$DB$MMSE.CHANGE = (db$DB$MMSE.M36 - db$DB$MMSE)

################################################################################################
## SIMPLE ENET PREDICTION NO PLOT/PERM etc.
################################################################################################
# M36 ~ M0
if(FALSE){
PNZEROS = 0.5# c(.1, .25 , .5, .75)
ALPHAS = 0.25# seq(0, 1, .25)
#SEEDS = seq(1, 100)
if(VALIDATION == "CV") SEEDS = 11
if(VALIDATION == "FR-GE") SEEDS = 50
TARGETS = db$col_targets
}
if(TRUE){
# CHANGES ~ M0
PNZEROS = 0.5# c(.1, .25 , .5, .75)
ALPHAS = 0.25# seq(0, 1, .25)
SEEDS = 46#seq(1, 100)
#SEED ALPHA PNZERO diff_te_mu diff_te_sd
#46   46  0.25    0.5  0.1919573  0.1745975
#if(VALIDATION == "CV") SEEDS = 11
#if(VALIDATION == "FR-GE") SEEDS = 50
TARGETS = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE")
}

NPERM = 1#1000
NFOLD = 5
length(PNZEROS) * length(ALPHAS) * length(SEEDS) * NPERM * 4 * 4
FORGET = TRUE

SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+NIGLOBFULL"       = c(db$col_niglob, db$col_niglob_full),
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob),
                "BASELINE+CLINIC+NIGLOBFULL"= c(db$col_clinic, db$col_niglob, db$col_niglob_full))
SETTINGS2 = list("BASELINE"       = c(),
                "BASELINE+LLV"       = "LLV",
                "BASELINE+BPF"       = "BPF",
                "BASELINE+MBcount"       = "MBcount",
                "BASELINE+LLV+BPF"       = c("LLV", "BPF"),
                "BASELINE+LLV+BPF+MBcount"       = c("LLV", "BPF", "MBcount"))
RESULTS_TAB = NULL
RESULTS = list()

#seed=11
for(TARGET in TARGETS){
  RESULTS[[TARGET]] = list()
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"; TARGET =  "MRS.M36"; TARGET =          "MMSE.M36" 
  # TARGET = "MMSE.CHANGE"
  cat("** TARGET:", TARGET, "**\n" )
for(PREDICTORS_STR in names(SETTINGS)){
  RESULTS[[TARGET]][[PREDICTORS_STR]] = list()
  #PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"
  #PREDICTORS_STR = "BASELINE+NIGLOB"
  #print(PREDICTORS)
  #PREXIF = paste(OUTPUT, "/",TARGET, "~", PREDICTORS_STR , sep="")
  #cat(PREXIF,"\n")
  cat("** PREDICTORS_STR:", PREDICTORS_STR, "**\n" )
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
for(ALPHA in ALPHAS){
  #cat(" ** ALPHA:",ALPHA,"**\n")
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]] = list()
for(PNZERO in PNZEROS){
    #cat(" ** PNZERO:",PNZERO,"**\n")
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]] = list()
for(SEED in SEEDS){
  #cat("  ** SEED:", SEED, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]] = list()
  set.seed(SEED)
  #pdf(paste(OUTPUT, "/",TARGET, "_datasets_qqplot.pdf", sep=""))
  #qqplot(D_tr[, TARGET], D_te[, TARGET], main=TARGET)
  #dev.off()
  #cat("=== ", TARGET, " ===\n")
  #print(D_tr_summary)
  #print(D_te_summary)
for(PERM in 1:NPERM){
  DB = db$DB
  if(PERM != 1){
    DB[!is.na(DB[, TARGET]), TARGET] = sample(DB[!is.na(DB[, TARGET]), TARGET])
  }
  #cat("    ** PERM:", PERM, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]] = list()
  #SPLITS = split_db_site_stratified_with_same_target_distribution_rm_na(DB, TARGET)
  if(VALIDATION == "CV"){
    SPLITS = kfold_site_stratified_rm_na(DB, TARGET, k=NFOLD)
  }
  if(VALIDATION == "FR-GE"){
    SPLITS = twofold_bysite_rm_na(DB, TARGET)
  }
for(FOLD in 1:length(SPLITS)){
  #cat("   ** fold:", FOLD, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]] = list()
  D_tr = SPLITS[[FOLD]]$tr
  D_te = SPLITS[[FOLD]]$te
  if(RM_TEST_OUTLIERS){
    idx_test_keep =
      D_te[, BASELINE] <= max(D_tr[, BASELINE], na.rm=TRUE) &
      D_te[, TARGET]   <= max(D_tr[, TARGET], na.rm=TRUE)   &
      D_te[, BASELINE] >= min(D_tr[, BASELINE], na.rm=TRUE) &
      D_te[, TARGET]   >= min(D_tr[, TARGET], na.rm=TRUE)
    D_te = D_te[idx_test_keep, ]
  }
  X_tr = as.matrix(D_tr[, PREDICTORS])
  y_true_tr = D_tr[, TARGET]
  X_te = as.matrix(D_te[, PREDICTORS])
  y_true_te = D_te[, TARGET]
  #y = c(y_true_tr, y_true_te)

  # ENET -------------------
  if(dim(X_tr)[2]>1){
  cv_glmnet = cv.glmnet(X_tr, y_true_tr, alpha=ALPHA)
  lambda = cv_glmnet$lambda.min # == cv_glmnet$lambda[which.min(cv_glmnet$cvm)]
  enet_nzero = cv_glmnet$nzero[which.min(cv_glmnet$cvm)][[1]]
  enet_nzero_min = max(round(dim(X_tr)[2]*PNZERO), 2)
  if(enet_nzero < enet_nzero_min)
    lambda = cv_glmnet$lambda[which(cv_glmnet$nzero > enet_nzero_min)[1]]
  # if cannot find such lambda take the last one (least penalization)
  if(is.na(lambda)) lambda = cv_glmnet$lambda[length(cv_glmnet$lambda)]
  mod_enet = glmnet(X_tr, y_true_tr, lambda=lambda, alpha=ALPHA)
  enet_nzero = sum(mod_enet$beta!=0)
  coef_enet = as.double(mod_enet$beta); names(coef_enet) = rownames(mod_enet$beta); coef_enet = coef_enet[coef_enet!=0]
  }
  # GLM -------------------
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  mod_glm = lm(formula, data=D_tr)
  coef_glm = mod_glm$coefficients
  
  # Predict ENET
  if(dim(X_tr)[2]>1){
  y_pred_te_enet = predict(mod_enet, X_te)
  y_pred_tr_enet = predict(mod_enet, X_tr)
  loss_te_enet = loss_reg(y_true_te, y_pred_te_enet, NULL, suffix="te")
  #loss_te_enet = loss_reg(y_true_te[idx_test_keep], y_pred_te_enet[idx_test_keep], NULL, suffix="te")
  loss_tr_enet = loss_reg(y_true_tr, y_pred_tr_enet, NULL, suffix="tr")
  }
  # Predict GLM
  y_pred_te_glm = predict(mod_glm, D_te)
  y_pred_tr_glm = predict(mod_glm, D_tr)
  loss_te_glm = loss_reg(y_true_te, y_pred_te_glm, NULL, suffix="te")
  #loss_te_glm = loss_reg(y_true_te[idx_test_keep], y_pred_te_glm[idx_test_keep], NULL, suffix="te")
  loss_tr_glm = loss_reg(y_true_tr, y_pred_tr_glm, NULL, suffix="tr")

  # GLM
  result_glm = list(FOLD=FOLD, PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
  mod=mod_glm, coef_glm=coef_glm,
  y_pred_te=y_pred_te_glm,
  y_pred_tr=y_pred_tr_glm,
  loss_te=loss_te_glm,
  loss_tr=loss_tr_glm,
  # DATA          
  X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te,
  D_tr=D_tr, D_te=D_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["GLM"]] = result_glm
  if(dim(X_tr)[2]>1){
    result_enet = list(# ENET
      mod = mod_enet, coef_enet=coef_enet, 
      y_pred_te = y_pred_te_enet,
      y_pred_tr = y_pred_tr_enet,
      loss_te = loss_te_enet,
      loss_tr = loss_tr_enet,
      X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te, 
      D_tr=D_tr, D_te=D_te)           
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["ENET"]] = result_enet
  }
  #save(result, file=paste(PREXIF, ".Rdata", sep=""))

  # RESULTS_TAB
  res = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="GLM",
       dim=paste(dim(X_tr), collapse="x"),
       as.list(c(loss_te_glm, loss_tr_glm, mse_te_se=NA, r2_te_se=NA, cor_te_se=NA)),
       nzero=(length(coef_glm)-1),
       coef=paste(names(coef_glm), collapse=", "),
       coef_val=paste(coef_glm, collapse=", "))
  RESULTS_TAB = rbind(RESULTS_TAB, res)
  if(dim(X_tr)[2]>1){
  res_enet = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="ENET", 
            dim=paste(dim(X_tr), collapse="x"),
            as.list(c(loss_te_enet, loss_tr_enet, mse_te_se=NA, r2_te_se=NA, cor_te_se=NA)),
            nzero=enet_nzero,
            coef=paste(names(coef_enet), collapse=", "),
            coef_val=paste(coef_enet, collapse=", "))
  RESULTS_TAB = rbind(RESULTS_TAB, res_enet)
  }
} # FOLD
  perm_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]]
  y_true_tr_glm = c()
  y_pred_tr_glm = c()
  y_true_te_glm = c()
  y_pred_te_glm = c()
  loss_te_glm = NULL
  y_true_te_enet = c()
  y_pred_te_enet = c()
  y_true_tr_enet = c()
  y_pred_tr_enet = c()
  loss_te_enet = NULL
  for(FOLD in 1:length(perm_curr)){
    cv_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]]
    y_true_tr_glm = c(y_true_tr_glm, cv_curr[["GLM"]]$y_true_tr)
    y_pred_tr_glm = c(y_pred_tr_glm, cv_curr[["GLM"]]$y_pred_tr)
    y_true_te_glm = c(y_true_te_glm, cv_curr[["GLM"]]$y_true_te)
    y_pred_te_glm = c(y_pred_te_glm, cv_curr[["GLM"]]$y_pred_te)
    loss_te_glm = rbind(loss_te_glm, cv_curr[["GLM"]]$loss_te)
    try({
      y_true_tr_enet= c(y_true_tr_enet, cv_curr[["ENET"]]$y_true_tr);
      y_pred_tr_enet= c(y_pred_tr_enet, cv_curr[["ENET"]]$y_pred_tr);
      y_true_te_enet= c(y_true_te_enet, cv_curr[["ENET"]]$y_true_te);
      y_pred_te_enet= c(y_pred_te_enet, cv_curr[["ENET"]]$y_pred_te)
      loss_te_enet = rbind(loss_te_enet, cv_curr[["ENET"]]$loss_te)
      })
  }
  #print(dim(loss_te_glm))
  loss_te_se_glm = apply(loss_te_glm, 2, sd) / sqrt(nrow(loss_te_glm))
  names(loss_te_se_glm) <- paste(names(loss_te_se_glm), "se", sep="_")
  RESULTS_TAB = rbind(RESULTS_TAB, 
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="ALL", MODEL="GLM", 
             dim=paste(dim(X_tr), collapse="x"),
             as.list(c(loss_reg(y_true_te_glm, y_pred_te_glm, suffix="te"),
                       loss_reg(y_true_tr_glm, y_pred_tr_glm, suffix="tr"),
                       loss_te_se_glm)),
             nzero=NA, coef=NA, coef_val=NA))
  if(!is.null(loss_te_enet)){
    #print(dim(loss_te_glm))
    loss_te_se_enet = apply(loss_te_enet, 2, sd) / sqrt(nrow(loss_te_enet))
    names(loss_te_se_enet) <- paste(names(loss_te_se_enet), "se", sep="_")
  RESULTS_TAB = rbind(RESULTS_TAB, 
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="ALL", MODEL="ENET", 
                 dim=paste(dim(X_tr), collapse="x"),
                 as.list(c(loss_reg(y_true_te_enet, y_pred_te_enet, suffix="te"),
                           loss_reg(y_true_tr_enet, y_pred_tr_enet, suffix="tr"),
                           loss_te_se_enet)),
                 nzero=NA, coef=NA, coef_val=NA))
  }
  if(FORGET && PERM!=1)
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]] = list()
} # PERM
} # SEED
} # ALPHA
} # PNZERO
} # PREDICTORS_STR
} # TARGET
#write.csv(paste(OUTPUT, "RESULTS_TAB__10CV.csv", sep="/"), row.names=FALSE)
write.csv(RESULTS_TAB, paste(OUTPUT, "/RESULTS_TAB_",VALIDATION,".csv", sep=""), row.names=FALSE)
#write.csv(RESULTS_TAB, paste(OUTPUT, "/RESULTS_TAB_1000PERMS.csv",VALIDATION,".csv", sep=""), row.names=FALSE)
#write.csv(RESULTS_TAB, paste(OUTPUT, "/RESULTS_TAB_",VALIDATION, "LLV-BPF-MBcount.csv", sep=""), row.names=FALSE)

save(RESULTS, file=paste(OUTPUT, "/RESULTS_",VALIDATION,".Rdata", sep=""))
