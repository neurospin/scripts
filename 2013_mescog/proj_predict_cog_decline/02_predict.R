require(ggplot2)
require(glmnet)
require(reshape)
library(RColorBrewer)
library(plyr)

#display.brewer.pal(6, "Paired")
palette = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140128_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140204_nomissing_BPF-LLV_imputed_lm.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), sep="/")
if (!file.exists(OUTPUT)) dir.create(OUTPUT)
#VALIDATION = "CV"
#RM_TEST_OUTLIERS = FALSE
VALIDATION = "FR-GE"
RM_TEST_OUTLIERS = TRUE

source(paste(SRC,"utils.R",sep="/"))

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB)# 372  29
#ALPHA=.95 #enet

################################################################################################
## SIMPLE ENET PREDICTION NO PLOT/PERM etc.
################################################################################################
#GRID=NULL
#ALPHAS = seq(0, 1, 0.05)
#ALPHAS = seq(0, 1, 0.25)
#ALPHAS = seq(0, 1, 0.5)
PNZEROS = c(.1, .25 , .5, .75)
ALPHAS = seq(0, 1, .25)
SEEDS = seq(1, 5)
NPERM = 1
NFOLD = 5
length(PNZEROS) * length(ALPHAS) * length(SEEDS) * NPERM * 4 * 4

SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob))
RESULTS_TAB = NULL
RESULTS = list()

#seed=11
for(TARGET in db$col_targets){
  RESULTS[[TARGET]] = list()
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"; TARGET =  "MRS.M36"; TARGET =          "MMSE.M36" 
  cat("** TARGET:", TARGET, "**\n" )
for(PREDICTORS_STR in names(SETTINGS)){
  RESULTS[[TARGET]][[PREDICTORS_STR]] = list()
  #PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"
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
  cat("    ** PERM:", PERM, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]] = list()
  #SPLITS = split_db_site_stratified_with_same_target_distribution_rm_na(db$DB, TARGET)
  if(VALIDATION == "CV"){
    SPLITS = kfold_site_stratified_rm_na(db$DB, TARGET, k=NFOLD)
  }
  if(VALIDATION == "FR-GE"){
    SPLITS = twofold_bysite_rm_na(db$DB, TARGET)
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
  try({
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
  })  
} # PERM
} # SEED
} # ALPHA
} # PNZERO
} # PREDICTORS_STR
} # TARGET
#write.csv(paste(OUTPUT, "RESULTS_TAB__10CV.csv", sep="/"), row.names=FALSE)
write.csv(RESULTS_TAB, paste(OUTPUT, "/RESULTS_TAB_",VALIDATION,".csv", sep=""), row.names=FALSE)
save(RESULTS, file=paste(OUTPUT, "/RESULTS_",VALIDATION,".Rdata", sep=""))

if(FALSE){
## CHOOSE BEST PARAMETERS --------------------------------------------------------------------------------------------------
diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == "ALL",], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
#SEED ALPHA PNZERO diff_te_mu diff_te_sd
#79    4     1    0.5  0.1207913  0.1217304
diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == 1,], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
#SEED ALPHA PNZERO diff_te_mu diff_te_sd
#79    4     1    0.5   0.257412  0.1432386
diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == 2,], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
#SEED ALPHA PNZERO diff_te_mu diff_te_sd
#89    5   0.5    0.1 0.04937349 0.09421086

print(diff_summary$max)
SEED_CHOOSEN = diff_summary$max$SEED
print(diff_summary$diff_by_keys[diff_summary$diff_by_keys$SEED == SEED_CHOOSEN, ])

Rbest = RESULTS_TAB[RESULTS_TAB$SEED==SEED_CHOOSEN & RESULTS_TAB$FOLD == "ALL", ]

## PLOT CV --------------------------------------------------------------------------------------------------
pcv = ggplot(Rbest, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-r2_te_se, ymax=r2_te+r2_te_se), width=.1)+
  #  geom_point() +
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("CV")

x11(); print(pcv)

svg(paste(OUTPUT, "/RESULTS_",VALIDATION,".svg", sep=""))
print(pcv)
dev.off()
}

## PLOT TRAIN-TEST --------------------------------------------------------------------------------------------------
if(VALIDATION == "FR-GE"){
  ## CHOOSE BEST PARAMETERS --------------------------------------------------------------------------------------------------
  diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == "ALL",], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
  #SEED ALPHA PNZERO diff_te_mu diff_te_sd
  #   4     1    0.5  0.1207913  0.1217304
  diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == 1,], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
  #SEED ALPHA PNZERO diff_te_mu diff_te_sd
  #   4     1    0.5   0.257412  0.1432386
  diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == 2,], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
  print(diff_summary$max)
  #SEED ALPHA PNZERO diff_te_mu diff_te_sd
  #   5   0.5    0.1 0.04937349 0.09421086
  average_by_seed = ddply(diff_summary$diff_by_keys, as.quoted(c("ALPHA" ,"PNZERO", "TARGET")), summarise, diff_te_mu=mean(diff_te), diff_te_sd=sd(diff_te))
  
  # Plot diff ~ ALPHA facet
  pd <- position_dodge(.1) # move them .05 to the left and right
  average_by_seed$PNZERO = as.factor(average_by_seed$PNZERO)
  ggplot(average_by_seed , aes(x=ALPHA, y=diff_te_mu, colour=PNZERO)) + 
    #geom_errorbar(aes(ymin=diff_te_mu-diff_te_sd, ymax=diff_te_mu+diff_te_sd), width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd) + facet_grid(TARGET~., scale="free")
  
  ## BAR PLOT with best seed
  library(plyr)  
  R = RESULTS_TAB[RESULTS_TAB$FOLD != "ALL" & RESULTS_TAB$SEED==diff_summary$max$SEED,]
  R = rbind(R[(R$PREDICTORS == "BASELINE") & (R$MODEL =="GLM"),], R[(R$MODEL =="ENET"),])
  R = R[,c("FOLD", "MODEL","ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr")]
  ## ----------------------------
  ## PLOT R2 bar plot
  
  Rfr = R[R$FOLD==1, ]
  pd <- position_dodge(.1) # move them .05 to the left and right
  pbfr = ggplot(Rfr, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
    geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
    facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("FR>GE")
  x11();print(pbfr)
  Rge = R[R$FOLD==2, ]
  pd <- position_dodge(.1) # move them .05 to the left and right
  pbge = ggplot(Rge, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
    geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
    facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("GE>FR")
  x11();print(pbge)
  pdf(paste(OUTPUT, "/RESULTS_",VALIDATION,".pdf", sep=""))
  print(pbfr)
  print(pbge)
  dev.off()  
}
## PLOT CV --------------------------------------------------------------------------------------------------

if(FALSE){
################################################################################################
## PLOT
################################################################################################


Rm = melt(R, id.vars=c("MODEL","ALPHA","SEED","TARGET","PREDICTORS"))
# 4 TARGETs x 4 PREDICTORS x 2 score x 2 SEED x  x 11 ALPHA
nrow(Rm)
4 * 4 * 2 * length(SEEDS) * length(ALPHAS)
Rm.stat = summarySE(Rm, measurevar="value", groupvars=c("MODEL","ALPHA", "TARGET","PREDICTORS", "variable"))
nrow(Rm.stat)
4 * 4 * 3 * length(alphas)

## ----------------------------
## PLOT R2 as function of alpha
pd <- position_dodge(.1) # move them .05 to the left and right
ggplot(Rm.stat, aes(x=ALPHA, y=value, colour=PREDICTORS)) + 
  geom_errorbar(aes(ymin=value-ci, ymax=value+ci), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) + facet_grid(TARGET~variable, scale="free")



R.stat = summarySE(R, measurevar= c("r2_te",""), groupvars=c("MODEL","ALPHA", "TARGET","PREDICTORS"))
nrow(R.stat)
# 4 TARGETs x 4 PREDICTORS 

pd <- position_dodge(.1) # move them .05 to the left and right
pbar = ggplot(R.stat, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-r2_te_se, ymax=r2_te+r2_te_se), width=.1) + 
#  geom_point() +
  facet_wrap(~TARGET) + scale_fill_manual(values=pal)
print(pbar)
#scale_color_manual(values=pal)
}