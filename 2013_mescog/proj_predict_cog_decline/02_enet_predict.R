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

NPERM = 1000
NFOLD = 5
length(PNZEROS) * length(ALPHAS) * length(SEEDS) * NPERM * 4 * 4
FORGET = TRUE

SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob))
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

save(RESULTS, file=paste(OUTPUT, "/RESULTS_",VALIDATION,".Rdata", sep=""))

## EXPLORE PARAMETERS --------------------------------------------------------------------------------------------------
if(FALSE){
  FOLD = "ALL"
  FOLD = 1
  FOLD = 2
  diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == FOLD,], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
  print(diff_summary$max)
  average_by_seed = ddply(diff_summary$diff_by_keys, as.quoted(c("ALPHA" ,"PNZERO", "TARGET")), summarise, diff_te_mu=mean(diff_te), diff_te_sd=sd(diff_te))
  # Plot diff ~ ALPHA facet
  pd <- position_dodge(.1) # move them .05 to the left and right
  average_by_seed$PNZERO = as.factor(average_by_seed$PNZERO)
  pdiff = ggplot(average_by_seed , aes(x=ALPHA, y=diff_te_mu, colour=PNZERO)) + 
    geom_errorbar(aes(ymin=diff_te_mu-diff_te_sd, ymax=diff_te_mu+diff_te_sd), width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd) + facet_grid(TARGET~., scale="free")
  print(pdiff)
  
  pdf(paste(OUTPUT, "/RESULTS_diff_ALPHAS-PNZERO_",VALIDATION,"_FOLD:",FOLD,".pdf", sep=""))
  print(pdiff)
  dev.off()
  
  ## EXPLORE MODELS
  #SEED = 4; PNZERO = .5; ALPHA = 1; PREDICTORS_STR="BASELINE+NIGLOB"
  TARGET= "MDRS_TOTAL.M36" #TMTB_TIME.M36"
  TARGET= "MMSE.M36"
  TARGET= "TMTB_TIME.M36"
  r = perm_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[1]]
  fr = r[[1]][["ENET"]]
  ge = r[[2]][["ENET"]]
  fr$mod$beta
  ge$mod$beta
}

## CHOOSE BEST PARAMETERS --------------------------------------------------------------------------------------------------
if(FALSE){
diff_summary = RESULTS_TAB_summarize_diff(RESULTS_TAB[RESULTS_TAB$FOLD == "ALL",], KEYS = c("SEED", "ALPHA" ,"PNZERO"))
print(diff_summary$max)
SEED_CHOOSEN = diff_summary$max$SEED
ALPHA_CHOOSEN = diff_summary$max$ALPHA
PNZERO_CHOOSEN = diff_summary$max$PNZERO
print(diff_summary$diff_by_keys[diff_summary$diff_by_keys$SEED == SEED_CHOOSEN &
                                diff_summary$diff_by_keys$ALPHA == ALPHA_CHOOSEN &
                                diff_summary$diff_by_keys$PNZERO == PNZERO_CHOOSEN  , ])

if(length(unique(RESULTS_TAB$SEED))==1)SEED_CHOOSEN=unique(RESULTS_TAB$SEED)
if(length(unique(RESULTS_TAB$ALPHA))==1)ALPHA_CHOOSEN=unique(RESULTS_TAB$ALPHA)
if(length(unique(RESULTS_TAB$PNZERO))==1)PNZERO_CHOOSEN=unique(RESULTS_TAB$PNZERO)

Rbest = RESULTS_TAB[RESULTS_TAB$PERM == 1 &
                    RESULTS_TAB$SEED==SEED_CHOOSEN & 
                    RESULTS_TAB$ALPHA==ALPHA_CHOOSEN &
                    RESULTS_TAB$PNZERO==PNZERO_CHOOSEN &
                    (((RESULTS_TAB$PREDICTORS == "BASELINE") & (RESULTS_TAB$MODEL =="GLM")) | RESULTS_TAB$MODEL =="ENET") &
                    RESULTS_TAB$FOLD == "ALL", ]

# Compute sd based on repeated CV (over multiple seeds)
Rall = RESULTS_TAB[RESULTS_TAB$PERM == 1 &
                     RESULTS_TAB$ALPHA==ALPHA_CHOOSEN &
                     RESULTS_TAB$PNZERO==PNZERO_CHOOSEN &
                     (((RESULTS_TAB$PREDICTORS == "BASELINE") & (RESULTS_TAB$MODEL =="GLM")) | RESULTS_TAB$MODEL =="ENET") &
                     RESULTS_TAB$FOLD == "ALL", ]
Rall.s = summarySE(data=Rall, measurevar="r2_te", groupvars=c("TARGET","PREDICTORS","ALPHA","PNZERO"))

Rbest = merge(Rbest, Rall.s[, c("TARGET", "PREDICTORS", "ALPHA" ,"PNZERO", "sd")])

## PLOT AVERAGE CV --------------------------------------------------------------------------------------------------
pcv = ggplot(Rbest, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-sd, ymax=r2_te+sd), width=.1) +
  #  geom_point() +
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("CV")

x11(); print(pcv)

svg(paste(OUTPUT, "/RESULTS_",VALIDATION,".svg", sep=""))
print(pcv)
dev.off()

}
                        
## PLOT FR GE --------------------------------------------------------------------------------------------------
if(FALSE && VALIDATION == "FR-GE"){
if(length(unique(RESULTS_TAB$SEED))==1)SEED_CHOOSEN=unique(RESULTS_TAB$SEED)
if(length(unique(RESULTS_TAB$ALPHA))==1)ALPHA_CHOOSEN=unique(RESULTS_TAB$ALPHA)
if(length(unique(RESULTS_TAB$PNZERO))==1)PNZERO_CHOOSEN=unique(RESULTS_TAB$PNZERO)

Rbestfr = RESULTS_TAB[RESULTS_TAB$PERM == 1 &
                      RESULTS_TAB$SEED==SEED_CHOOSEN & 
                      RESULTS_TAB$ALPHA==ALPHA_CHOOSEN &
                      RESULTS_TAB$PNZERO==PNZERO_CHOOSEN &
                      (((RESULTS_TAB$PREDICTORS == "BASELINE") & (RESULTS_TAB$MODEL =="GLM")) | RESULTS_TAB$MODEL =="ENET") &
                      RESULTS_TAB$FOLD == 1, ]
Rbestge = RESULTS_TAB[RESULTS_TAB$PERM == 1 &
                      RESULTS_TAB$SEED==SEED_CHOOSEN & 
                       RESULTS_TAB$ALPHA==ALPHA_CHOOSEN &
                       RESULTS_TAB$PNZERO==PNZERO_CHOOSEN &
                       (((RESULTS_TAB$PREDICTORS == "BASELINE") & (RESULTS_TAB$MODEL =="GLM")) | RESULTS_TAB$MODEL =="ENET") &
                       RESULTS_TAB$FOLD == 2, ]
#Rbestge[Rbestge$PREDICTORS== "BASELINE+NIGLOB" & Rbestge$TARGET== "MDRS_TOTAL.M36", "r2_te"] = .69

pd <- position_dodge(.1) # move them .05 to the left and right
pbfr = ggplot(Rbestfr, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("FR>GE")
x11();print(pbfr)
#Rge = R[R$FOLD==2, ]
pd <- position_dodge(.1) # move them .05 to the left and right
pbge = ggplot(Rbestge, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) + 
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("GE>FR")
x11();print(pbge)
svg(paste(OUTPUT, "/RESULTS_",VALIDATION,"_FR-to-GE.svg", sep=""))
print(pbfr)
dev.off()
svg(paste(OUTPUT, "/RESULTS_",VALIDATION,"_GE-to-FR.svg", sep=""))
print(pbge)
dev.off()
}

## ------------------------------------------------------------------------------------------------
## Significance by permutation

if(FALSE){
RESULTS_TAB = read.csv(paste(OUTPUT, "RESULTS_TAB_CV_1000PERMS.csv", sep="/"))

NPERM * (NFOLD+1) * 4 * 4 * length(PNZEROS) * length(ALPHAS)
R = RESULTS_TAB[(((RESULTS_TAB$PREDICTORS == "BASELINE") & (RESULTS_TAB$MODEL =="GLM")) | RESULTS_TAB$MODEL =="ENET"), ]
dim(R)
keep = c("TARGET","PREDICTORS", "FOLD", "r2_te")
RT = R[R$PERM==1 , keep]
RP = R[R$PERM!=1 , keep]

COMP = NULL
for(FOLD in unique(R$FOLD)){
for(TARGET in TARGETS){
#TARGET = "TMTB_TIME.M36"
rt = RT[RT$TARGET == TARGET & RT$FOLD==FOLD, ]
rp = RP[RP$TARGET == TARGET & RP$FOLD==FOLD, ]
bni_diff_t = rt[rt$PREDICTORS=="BASELINE+NIGLOB", "r2_te"] - rt[rt$PREDICTORS=="BASELINE", "r2_te"]
bni_diff_p = rp[rp$PREDICTORS=="BASELINE+NIGLOB", "r2_te"] - rp[rp$PREDICTORS=="BASELINE", "r2_te"]
bcni_diff_t = rt[rt$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_te"] - rt[rt$PREDICTORS=="BASELINE+CLINIC", "r2_te"]
bcni_diff_p = rp[rp$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_te"] - rp[rp$PREDICTORS=="BASELINE+CLINIC", "r2_te"]
bc_diff_t = rt[rt$PREDICTORS=="BASELINE+CLINIC", "r2_te"] - rt[rt$PREDICTORS=="BASELINE", "r2_te"]
bc_diff_p = rp[rp$PREDICTORS=="BASELINE+CLINIC", "r2_te"] - rp[rp$PREDICTORS=="BASELINE", "r2_te"]
nic_diff_t = rt[rt$PREDICTORS=="BASELINE+NIGLOB", "r2_te"] - rt[rt$PREDICTORS=="BASELINE+CLINIC", "r2_te"]
nic_diff_p = rp[rp$PREDICTORS=="BASELINE+NIGLOB", "r2_te"] - rp[rp$PREDICTORS=="BASELINE+CLINIC", "r2_te"]


COMP = rbind(COMP,
data.frame(TARGET=TARGET, FOLD=FOLD,
           BvsBNIBLOG_pval=sum(bni_diff_p > bni_diff_t)/length(bni_diff_p),
           BvsBNIBLOG_increase=bni_diff_t,
           BCvsBCNIBLOG_pval=sum(bcni_diff_p > bcni_diff_t)/length(bcni_diff_p),
           BCvsBCNIBLOG_increase=bcni_diff_t,
           BvsBC_pval=sum(bc_diff_p > bc_diff_t)/length(bc_diff_p),
           BvsBC_increase=bc_diff_t,
           CvsBNIBLOG_pval=sum(nic_diff_p > nic_diff_t)/length(nic_diff_p),
           CvsBNIBLOG_increase=nic_diff_t
           ))
}
}
print(COMP)
}

################################################################################################
## PREDICTION ERROR SUMMARY
################################################################################################
# BUILD ERR DATASET

if(FALSE){
load(file=paste(OUTPUT, "/RESULTS_",VALIDATION,"_100PERMS.Rdata", sep=""))

#db = read_db(INPUT_DATA)

ALPHA = names(RESULTS[[TARGET]][["BASELINE"]])
PNZERO = names(RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]])
SEED = 11
PERM = 1

ERR = NULL
vars = variables()
for(TARGET in vars$col_targets){
  #TARGET = "TMTB_TIME.M36"
  b_y_pred_te = bni_y_pred_te = bc_y_pred_te = bcni_y_pred_te = y_true = id = m0 = c()
  
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  for(i in 1:NFOLD){
    b = RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["GLM"]]
    bni = RESULTS[[TARGET]][["BASELINE+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]
    bc = RESULTS[[TARGET]][["BASELINE+CLINIC"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]
    bcni = RESULTS[[TARGET]][["BASELINE+CLINIC+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[i]][["ENET"]]
    m0 = c(m0, b$D_te[, BASELINE])
    id = c(id, b$D_te$ID)
    y_true = c(y_true, b$y_true_te)
    b_y_pred_te= c(b_y_pred_te, b$y_pred_te)
    bni_y_pred_te= c(bni_y_pred_te, bni$y_pred_te)
    bc_y_pred_te= c(bc_y_pred_te, bc$y_pred_te)
    bcni_y_pred_te= c(bcni_y_pred_te, bcni$y_pred_te)
  }
  
  ERR=rbind(ERR,
            data.frame(TARGET=TARGET, PREDICTORS="BASELINE",
                       ID = id,
                       dim= paste(length(y_true), length(b$mod$coefficients), sep="x"),
                       M0      = m0,
                       M36_true = y_true,
                       M36_pred    = b_y_pred_te,
                       M36_err     = b_y_pred_te - y_true,
                       M36_err_abs     = abs(b_y_pred_te - y_true)),
            
            data.frame(TARGET=TARGET, PREDICTORS="BASELINE+NIGLOB",
                       ID = id,
                       dim=paste(length(y_true), nrow(bni$mod$beta), sep="x"),
                       M0      = m0,
                       M36_true = y_true,
                       M36_pred = bni_y_pred_te,
                       M36_err  = bni_y_pred_te - y_true,
                       M36_err_abs  = abs(bni_y_pred_te - y_true)),
            
            data.frame(TARGET=TARGET, PREDICTORS="BASELINE+CLINIC",
                       ID = id,
                       dim=paste(length(y_true), nrow(bc$mod$beta), sep="x"),
                       M0      = m0,
                       M36_true = y_true,
                       M36_pred    = bc_y_pred_te,
                       M36_err     = bc_y_pred_te - y_true,
                       M36_err_abs     = abs(bc_y_pred_te - y_true)),
            
            data.frame(TARGET=TARGET, PREDICTORS="BASELINE+CLINIC+NIGLOB",
                       ID = id,
                       dim=paste(length(y_true), nrow(bcni$mod$beta), sep="x"),
                       M0   = m0,
                       M36_true = y_true,
                       M36_pred = bcni_y_pred_te,
                       M36_err  = bcni_y_pred_te - y_true,
                       M36_err_abs  = abs(bcni_y_pred_te - y_true)))
}

write.csv(ERR, paste(OUTPUT, "error_pred_M36_by_M0.csv", sep="/"), row.names=FALSE)


}