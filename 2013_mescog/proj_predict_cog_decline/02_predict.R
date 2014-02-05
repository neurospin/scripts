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
VALIDATION = "CV"
RM_TEST_OUTLIERS = TRUE
#VALIDATION = "FR-GE"
#RM_TEST_OUTLIERS = TRUE

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
ALPHAS = 1
SEEDS = seq(1, 100)
NPERM = 1
NFOLD = 5
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
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
for(ALPHA in ALPHAS){
  cat(" ** ALPHA:",ALPHA,"**\n")
  RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]] = list()  
for(SEED in SEEDS){
  cat("  ** SEED:", SEED, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]] = list()
  set.seed(SEED)
  #pdf(paste(OUTPUT, "/",TARGET, "_datasets_qqplot.pdf", sep=""))
  #qqplot(D_tr[, TARGET], D_te[, TARGET], main=TARGET)
  #dev.off()
  #cat("=== ", TARGET, " ===\n")
  #print(D_tr_summary)
  #print(D_te_summary)
for(PERM in 1:NPERM){
  cat("    ** PERM:", PERM, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]] = list()
  #SPLITS = split_db_site_stratified_with_same_target_distribution_rm_na(db$DB, TARGET)
  if(VALIDATION == "CV"){
    SPLITS = kfold_site_stratified_rm_na(db$DB, TARGET, k=NFOLD)
  }
  if(VALIDATION == "FR-GE"){
    SPLITS = twofold_bysite_rm_na(db$DB, TARGET)
  }
for(FOLD in 1:length(SPLITS)){
  cat("   ** fold:", FOLD, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]][[FOLD]] = list()
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
  set.seed(SEED)
  cv_glmnet = cv.glmnet(X_tr, y_true_tr, alpha=ALPHA)
  lambda = cv_glmnet$lambda.min # == cv_glmnet$lambda[which.min(cv_glmnet$cvm)]
  enet_nzero = cv_glmnet$nzero[which.min(cv_glmnet$cvm)][[1]]
  enet_nzero_min = max(round(dim(X_tr)[2]/10), 2)
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
  RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]][[FOLD]][["GLM"]] = result_glm
  if(dim(X_tr)[2]>1){
    result_enet = list(# ENET
      mod = mod_enet, coef_enet=coef_enet, 
      y_pred_te = y_pred_te_enet,
      y_pred_tr = y_pred_tr_enet,
      loss_te = loss_te_enet,
      loss_tr = loss_tr_enet,
      X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te, 
      D_tr=D_tr, D_te=D_te)           
    RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]][[FOLD]][["ENET"]] = result_enet
  }
  #save(result, file=paste(PREXIF, ".Rdata", sep=""))

  # RESULTS_TAB
  res = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="GLM",
       dim=paste(dim(X_tr), collapse="x"),
       as.list(c(loss_te_glm, loss_tr_glm, mse_te_se=NA, r2_te_se=NA, cor_te_se=NA)),
       nzero=(length(coef_glm)-1),
       coef=paste(names(coef_glm), collapse=", "),
       coef_val=paste(coef_glm, collapse=", "))
  RESULTS_TAB = rbind(RESULTS_TAB, res)
  if(dim(X_tr)[2]>1){
  res_enet = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="ENET", 
            dim=paste(dim(X_tr), collapse="x"),
            as.list(c(loss_te_enet, loss_tr_enet, mse_te_se=NA, r2_te_se=NA, cor_te_se=NA)),
            nzero=enet_nzero,
            coef=paste(names(coef_enet), collapse=", "),
            coef_val=paste(coef_enet, collapse=", "))
  RESULTS_TAB = rbind(RESULTS_TAB, res_enet)
  }
} # FOLD
  perm_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]]
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
    cv_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[ALPHA]][[SEED]][[PERM]][[FOLD]]
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
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, SEED=SEED, PERM=PERM, FOLD="ALL", MODEL="GLM", 
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
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, SEED=SEED, PERM=PERM, FOLD="ALL", MODEL="ENET", 
                 dim=paste(dim(X_tr), collapse="x"),
                 as.list(c(loss_reg(y_true_te_enet, y_pred_te_enet, suffix="te"),
                           loss_reg(y_true_tr_enet, y_pred_tr_enet, suffix="tr"),
                           loss_te_se_enet)),
                 nzero=NA, coef=NA, coef_val=NA))
  })
} # PERM
} # SEED
} # ALPHA
} # PREDICTORS_STR
} # TARGET
#write.csv(paste(OUTPUT, "RESULTS_TAB__10CV.csv", sep="/"), row.names=FALSE)
write.csv(RESULTS_TAB, paste(OUTPUT, "/RESULTS_TAB_",VALIDATION,".csv", sep=""), row.names=FALSE)
save(RESULTS, file=paste(OUTPUT, "/RESULTS_",VALIDATION,".Rdata", sep=""))

R = RESULTS_TAB[RESULTS_TAB$FOLD == "ALL",]
R = rbind(R[(R$PREDICTORS == "BASELINE") & (R$MODEL =="GLM"),], R[(R$MODEL =="ENET"),])
R = R[,c("MODEL","ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr", "r2_te_se")]
# 4 TARGETs x 4 PREDICTORS x 2 score x 2 SEED x  x 11 ALPHA
nrow(R) == 4 * 4 * length(SEEDS) * length(ALPHAS)

#ddply(R, .(MODEL, ALPHA, SEED, PREDICTORS), summarise,mean=mean(r2_te),sd=sd(r2_te))


# Which is best seed
b    = R[R$PREDICTORS == "BASELINE",               c("SEED", "TARGET", "r2_te", "r2_te_se")]
bni  = R[R$PREDICTORS == "BASELINE+NIGLOB",        c("SEED", "TARGET", "r2_te", "r2_te_se")]
bc   = R[R$PREDICTORS == "BASELINE+CLINIC",        c("SEED", "TARGET", "r2_te", "r2_te_se")]
bcni = R[R$PREDICTORS == "BASELINE+CLINIC+NIGLOB", c("SEED", "TARGET", "r2_te", "r2_te_se")]
m1 = merge(b, bni, by=c("SEED", "TARGET"), suffixes=c("_b", "_bni"))
m2 = merge(bc, bcni, by=c("SEED", "TARGET"), suffixes=c("_bc", "_bcni"))
m = merge(m1, m2, by=c("SEED", "TARGET"))
nrow(m) == 4 * length(SEEDS)
## ICI


b_vs_bni$diff = b_vs_bni$r2_te_bni - b_vs_bni$r2_te_b

nrow(b_vs_bni) == 4 * length(SEEDS)

bc_vs_bcni = merge(bc, bcni, by=c("SEED", "TARGET"), suffixes=c("_b", "_bni"))
nrow(bc_vs_bcni) == 4 * length(SEEDS)

bc_vs_bcni$diff = bc_vs_bcni$r2_te_bni - bc_vs_bcni$r2_te_b


bc_vs_bni_byseed = ddply(b_vs_bni,~SEED,summarise,mean=mean(diff),sd=sd(diff))
bc_vs_bni_byseed[bc_vs_bni_byseed$mean == max(bc_vs_bni_byseed$mean),]
#28   28 0.04563271 0.03174524 (10CV)
#78   78 0.03927785 0.02698512 (5CV)
#53   53 0.03415659 0.02925164 (5CV) no missing BPF, LLV
# 
bc_vs_bcni_byseed = ddply(bc_vs_bcni,~SEED,summarise,mean=mean(diff),sd=sd(diff))
bc_vs_bcni_byseed[bc_vs_bcni_byseed$mean == max(bc_vs_bcni_byseed$mean),]

CHOOSEN_SEED = which.max(bc_vs_bni_byseed$mean + bc_vs_bcni_byseed$mean)
Rbest = R[R$SEED== CHOOSEN_SEED, ]
#61   61 0.03755026 0.02776236
#86   86 0.04224727 0.03323897
#89   89 0.03524849 0.03154105 (5CV) no missing BPF, LLV
# Seed 61
#Rp = R[R$SEED==1 ,c("MODEL","ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr", "r2_te_se")]
#Rp = R[R$SEED==86 ,c("MODEL","ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr", "r2_te_se")]
#Rp = R[R$SEED==40 ,c("MODEL","ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr", "r2_te_se")]

#CHOOSEN_SEED = 50 #FR-GE

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

## PLOT TRAIN-TEST --------------------------------------------------------------------------------------------------
if(VALIDATION == "FR-GE"){ 
  library(plyr)  
  R = RESULTS_TAB[RESULTS_TAB$FOLD != "ALL" & RESULTS_TAB$SEED==CHOOSEN_SEED,]
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