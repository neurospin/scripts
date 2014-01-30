#require(ggplot2)
require(glmnet)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140128_imputed.csv", sep="/")

# OUTPUT ---
OUTPUT = paste(BASE_DIR, "20140128_pool-FR-GE", sep="/")
if (!file.exists(OUTPUT)) dir.create(OUTPUT)

source(paste(SRC,"utils.R",sep="/"))

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB)# 372  29
ALPHA=.95 #enet

################################################################################################
## SIMPLE ENET PREDICTION NO PLOT/PERM etc.
################################################################################################
RM_1_OUTLIERS = FALSE
TMP=NULL
for(ALPHA in seq(0, 1, 0.05)){
for(seed in 1:100){
#seed=11
SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob))
RESULTS = NULL
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"; TARGET =  "MRS.M36"; TARGET =          "MMSE.M36" 
  set.seed(seed)
  split = split_db_site_stratified_with_same_target_distribution_rm_na(db$DB, TARGET)
  #pdf(paste(OUTPUT, "/",TARGET, "_datasets_qqplot.pdf", sep=""))
  #qqplot(split$D1[, TARGET], split$D2[, TARGET], main=TARGET)
  #dev.off()
  #cat("=== ", TARGET, " ===\n")
  #print(split$D1_summary)
  #print(split$D2_summary)
  
for(PREDICTORS_STR in names(SETTINGS)){
  #PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"
  #print(PREDICTORS)
  PREXIF = paste(OUTPUT, "/",TARGET, "~", PREDICTORS_STR , sep="")
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
#  split$D1 = db$DB_FR[!is.na(db$DB_FR[, TARGET]), ]
#  split$D2 = db$DB_GE[!is.na(db$DB_GE[, TARGET]), ]
  X1 = as.matrix(split$D1[, PREDICTORS])
  y1 = split$D1[, TARGET]
  X2 = as.matrix(split$D2[, PREDICTORS])
  y2 = split$D2[, TARGET]

  # ENET -------------------
  if(dim(X1)[2]>1){
  set.seed(seed)
  cv_glmnet_1 = cv.glmnet(X1, y1, alpha=ALPHA)
  set.seed(seed)
  cv_glmnet_2 = cv.glmnet(X2, y2, alpha=ALPHA)
  lambda_1 = cv_glmnet_1$lambda.min # == cv_glmnet_1$lambda[which.min(cv_glmnet_1$cvm)]
  lambda_2 = cv_glmnet_2$lambda.min # == cv_glmnet_2$lambda[which.min(cv_glmnet_2$cvm)]
  enet_nzero_1 = cv_glmnet_1$nzero[which.min(cv_glmnet_1$cvm)][[1]]
  enet_nzero_2 = cv_glmnet_2$nzero[which.min(cv_glmnet_2$cvm)][[1]]
  #
  enet_nzero_min = max(round(dim(X1)[2]/3), 2)
  if(enet_nzero_1 < enet_nzero_min)
    lambda_1 = cv_glmnet_1$lambda[which(cv_glmnet_1$nzero > enet_nzero_min)[1]]
  if(enet_nzero_2 < enet_nzero_min)
    lambda_2 = cv_glmnet_2$lambda[which(cv_glmnet_2$nzero > enet_nzero_min)[1]]
  #
  mod_enet_1 = glmnet(X1, y1, lambda=lambda_1, alpha=ALPHA)
  mod_enet_2 = glmnet(X2, y2, lambda=lambda_2, alpha=ALPHA)
  enet_nzero_1 = sum(mod_enet_1$beta!=0)
  enet_nzero_2 = sum(mod_enet_2$beta!=0)
  coef_enet_1 = as.double(mod_enet_1$beta); names(coef_enet_1) = rownames(mod_enet_1$beta); coef_enet_1 = coef_enet_1[coef_enet_1!=0]
  coef_enet_2 = as.double(mod_enet_2$beta); names(coef_enet_2) = rownames(mod_enet_2$beta); coef_enet_2 = coef_enet_2[coef_enet_2!=0]
  }
  # GLM -------------------
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  mod_glm_1 = lm(formula, data=split$D1)
  mod_glm_2 = lm(formula, data=split$D2)
  coef_glm_1 = mod_glm_1$coefficients
  coef_glm_2 = mod_glm_2$coefficients
  
  if(RM_1_OUTLIERS){
    y_keep_1 =
      split$D1[, BASELINE] <= max(split$D2[, BASELINE], na.rm=TRUE) &
      split$D1[, TARGET]   <= max(split$D2[, TARGET], na.rm=TRUE)   &
      split$D1[, BASELINE] >= min(split$D2[, BASELINE], na.rm=TRUE) &
      split$D1[, TARGET]   >= min(split$D2[, TARGET], na.rm=TRUE)
  } else{y_keep_1 = rep(TRUE, nrow(X1))}
  
  # Predict ENET
  if(dim(X1)[2]>1){
  y_enet_pred_12 = predict(mod_enet_1, X2)
  y_enet_pred_11 = predict(mod_enet_1, X1)
  y_enet_pred_22 = predict(mod_enet_2, X2)
  y_enet_pred_21 = predict(mod_enet_2, X1[y_keep_1, ])
  
  loss_enet_12 = round(loss_reg(y2, y_enet_pred_12, NULL, suffix="1.2"), 2)
  loss_enet_21 = round(loss_reg(y1[y_keep_1], y_enet_pred_21, NULL, suffix="2.1"), 2)
  loss_enet_11 = round(loss_reg(y1, y_enet_pred_11, NULL, suffix="1.1"), 2)
  loss_enet_22 = round(loss_reg(y2, y_enet_pred_22, NULL, suffix="2.2"), 2)
  }
  # Predict GLM
  y_glm_pred_12 = predict(mod_glm_1, split$D2)
  y_glm_pred_11 = predict(mod_glm_1, split$D1)
  y_glm_pred_22 = predict(mod_glm_2, split$D2)
  y_glm_pred_21 = predict(mod_glm_2, split$D1[y_keep_1, ])
  
  loss_glm_12 = round(loss_reg(y2, y_glm_pred_12, NULL, suffix="1.2"), 2)
  loss_glm_21 = round(loss_reg(y1[y_keep_1], y_glm_pred_21, NULL, suffix="2.1"), 2)
  loss_glm_11 = round(loss_reg(y1, y_glm_pred_11, NULL, suffix="1.1"), 2)
  loss_glm_22 = round(loss_reg(y2, y_glm_pred_22, NULL, suffix="2.2"), 2)
  
  # result to save
  # GLM
  result = list(PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
  mod_glm_1=mod_glm_1, mod_glm_2=mod_glm_2, coef_glm_1=coef_glm_1, coef_glm_2=coef_glm_2,
  y_glm_pred_12=y_glm_pred_12,
  y_glm_pred_21=y_glm_pred_21,
  y_glm_pred_11=y_glm_pred_11,
  y_glm_pred_22=y_glm_pred_22,
  loss_glm_12=loss_glm_12,
  loss_glm_21=loss_glm_21,
  loss_glm_11=loss_glm_11,
  loss_glm_22=loss_glm_22,
  # DATA          
  X1=X1, X2=X2, y1=y1, y2=y2, y_keep_1=y_keep_1, D1=split$D1, D2=split$D2)
                
  if(dim(X1)[2]>1){
    result_enet = list(# ENET
      mod_enet_1=mod_enet_1, mod_enet_2=mod_enet_2, coef_enet_1=coef_enet_1, coef_enet_2=coef_enet_2,
      y_enet_pred_12=y_enet_pred_12,
      y_enet_pred_21=y_enet_pred_21,
      y_enet_pred_11=y_enet_pred_11,
      y_enet_pred_22=y_enet_pred_22,
      loss_enet_12=loss_enet_12,
      loss_enet_21=loss_enet_21,
      loss_enet_11=loss_enet_11,
      loss_enet_22=loss_enet_22)
    result = c(result, result_enet)
  }
  #save(result, file=paste(PREXIF, ".Rdata", sep=""))

  # results
  res = data.frame(MODEL="GLM", TARGET=TARGET, PREDICTORS=PREDICTORS_STR, dim_1=paste(dim(X1), collapse="x"), dim_2=paste(dim(X2), collapse="x"),
                       as.list(c(loss_glm_12, loss_glm_21)),
                       nzero_1=(length(coef_glm_1)-1), nzero_2=(length(coef_glm_2)-1),
                       coef_1=paste(names(coef_glm_1), collapse=", "),
                       coef_2=paste(names(coef_glm_2), collapse=", "),
                       coef_1_val=paste(coef_glm_1, collapse=", "),
                       coef_2_val=paste(coef_glm_2, collapse=", "))
  if(dim(X1)[2]>1){
  res_enet = data.frame(MODEL="ENET", TARGET=TARGET, PREDICTORS=PREDICTORS_STR, dim_1=paste(dim(X1), collapse="x"), dim_2=paste(dim(X2), collapse="x"),
                   as.list(c(loss_enet_12, loss_enet_21)),
                    nzero_1=enet_nzero_1, nzero_2=enet_nzero_2,
                    coef_1=paste(names(coef_enet_1), collapse=", "),
                    coef_2=paste(names(coef_enet_2), collapse=", "),
                    coef_1_val=paste(coef_enet_1, collapse=", "),
                    coef_2_val=paste(coef_enet_2, collapse=", "))
  res = rbind(res, res_enet)
  }
  RESULTS = rbind(RESULTS, res)
  #if(is.null(RESULTS)) RESULTS = data.1ame(as.list(res)) else RESULTS = rbind(RESULTS, data.1ame(as.list(res)))
}
}


r = RESULTS[RESULTS$MODEL == "ENET", c("TARGET","PREDICTORS", "r2_1.2", "r2_2.1")]
#diff_tot=sum(r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", c("r2_12","r2_2.1")] - r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC", c("r2_12", "r2_2.1")])
diff_1=sum(r[r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_1.2"] - r[r$PREDICTORS=="BASELINE+CLINIC", "r2_1.2"])    
diff_2=sum(r[r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_2.1"] - r[r$PREDICTORS=="BASELINE+CLINIC", "r2_2.1"])    
tmp = data.frame(alpha=ALPHA, seed=seed, diff_1, diff_2, tot=diff_1+diff_2)
print(tmp)
TMP = rbind(TMP, tmp)
}}

#write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)
###
###

