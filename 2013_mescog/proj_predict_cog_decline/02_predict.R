#install.packages("glmnet")
require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#setwd(WD)
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140121.csv", sep="/")
OUTPUT = paste(BASE_DIR, "20140128_pool-FR-GE", sep="/")

source(paste(SRC,"utils.R",sep="/"))
if (!file.exists(OUTPUT)) dir.create(OUTPUT)
OUTPUT_SUMMARY = paste(OUTPUT, "results_summary.csv", sep="/")

TO_REMOVE = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
              "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
              "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
              "LLVn", "WMHVn", "BRAINVOL", "LLcount")



# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA, TO_REMOVE)
dim(db$DB_FR)# 239  42 # 244  46
dim(db$DB_GE)# 126  42 # 128  46

################################################################################################
# REMOVE FR outliers
OUTPUT = paste(BASE_DIR, "20140120_remove-predictors", sep="/")
TO_REMOVE = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
              "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
              "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
              "LLVn", "WMHVn", "BRAINVOL", "LLcount")

#FORCE.ALPHA=1 #lasso
FORCE.ALPHA=.95 #enet

################################################################################################
## SIMPLE ENET PREDICTION NO PLOT/PERM etc.
################################################################################################
#RM_FR_OUTLIERS = FALSE
RM_FR_OUTLIERS = TRUE
#for(seed in 1:100){
#seed=11
RESULTS = NULL

SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob))

for(PREDICTORS_STR in names(SETTINGS)){
  #PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"
  
  #print(PREDICTORS)
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREXIF = paste(OUTPUT, "/",TARGET, "~", PREDICTORS_STR , sep="")
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
  DBfr = db$DB_FR[!is.na(db$DB_FR[, TARGET]), ]
  DBgr = db$DB_GE[!is.na(db$DB_GE[, TARGET]), ]
  Xfr = as.matrix(DBfr[, PREDICTORS])
  yfr = DBfr[, TARGET]
  Xgr = as.matrix(DBgr[, PREDICTORS])
  ygr = DBgr[, TARGET]

  # ENET -------------------
  if(dim(Xfr)[2]>1){
  alpha = FORCE.ALPHA
  set.seed(1)
  cv_glmnet_fr = cv.glmnet(Xfr, yfr, alpha=alpha)
  set.seed(20)
  cv_glmnet_gr = cv.glmnet(Xgr, ygr, alpha=alpha)
  lambda_fr = cv_glmnet_fr$lambda.min # == cv_glmnet_fr$lambda[which.min(cv_glmnet_fr$cvm)]
  lambda_gr = cv_glmnet_gr$lambda.min # == cv_glmnet_gr$lambda[which.min(cv_glmnet_gr$cvm)]
  enet_nzero_fr = cv_glmnet_fr$nzero[which.min(cv_glmnet_fr$cvm)][[1]]
  enet_nzero_gr = cv_glmnet_gr$nzero[which.min(cv_glmnet_gr$cvm)][[1]]
  #
  enet_nzero_min = max(round(dim(Xfr)[2]/4), 2)
  if(enet_nzero_fr < enet_nzero_min)
    lambda_fr = cv_glmnet_fr$lambda[which(cv_glmnet_fr$nzero > enet_nzero_min)[1]]
  if(enet_nzero_gr < enet_nzero_min)
    lambda_gr = cv_glmnet_gr$lambda[which(cv_glmnet_gr$nzero > enet_nzero_min)[1]]
  #
  mod_enet_fr = glmnet(Xfr, yfr, lambda=lambda_fr, alpha=alpha)
  mod_enet_gr = glmnet(Xgr, ygr, lambda=lambda_gr, alpha=alpha)
  enet_nzero_fr = sum(mod_enet_fr$beta!=0)
  enet_nzero_gr = sum(mod_enet_gr$beta!=0)
  enet_coef_fr = as.double(mod_enet_fr$beta); names(enet_coef_fr) = rownames(mod_enet_fr$beta); enet_coef_fr = enet_coef_fr[enet_coef_fr!=0]
  enet_coef_gr = as.double(mod_enet_gr$beta); names(enet_coef_gr) = rownames(mod_enet_gr$beta); enet_coef_gr = enet_coef_gr[enet_coef_gr!=0]
  }
  # GLM -------------------
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  mod_glm_fr = lm(formula, data=DBfr)
  mod_glm_gr = lm(formula, data=DBgr)
  glm_coef_fr = mod_glm_fr$coefficients
  glm_coef_gr = mod_glm_gr$coefficients
  
  if(RM_FR_OUTLIERS){
    fr_keep =
      DBfr[, BASELINE] <= max(DBgr[, BASELINE], na.rm=TRUE) &
      DBfr[, TARGET]   <= max(DBgr[, TARGET], na.rm=TRUE)   &
      DBfr[, BASELINE] >= min(DBgr[, BASELINE], na.rm=TRUE) &
      DBfr[, TARGET]   >= min(DBgr[, TARGET], na.rm=TRUE)
  } else{fr_keep = rep(TRUE, nrow(Xfr))}
  
  # Predict ENET
  if(dim(Xfr)[2]>1){
  y_enet_pred_frgr = predict(mod_enet_fr, Xgr)
  y_enet_pred_frfr = predict(mod_enet_fr, Xfr)
  y_enet_pred_grgr = predict(mod_enet_gr, Xgr)
  y_enet_pred_grfr = predict(mod_enet_gr, Xfr[fr_keep, ])
  
  loss_enet_frgr = round(loss.reg(ygr, y_enet_pred_frgr, NULL, suffix="FR.GR"), 2)
  loss_enet_grfr = round(loss.reg(yfr[fr_keep], y_enet_pred_grfr, NULL, suffix="GR.FR"), 2)
  loss_enet_frfr = round(loss.reg(yfr, y_enet_pred_frfr, NULL, suffix="FR.FR"), 2)
  loss_enet_grgr = round(loss.reg(ygr, y_enet_pred_grgr, NULL, suffix="GR.GR"), 2)
  }
  # Predict GLM
  y_glm_pred_frgr = predict(mod_glm_fr, DBgr)
  y_glm_pred_frfr = predict(mod_glm_fr, DBfr)
  y_glm_pred_grgr = predict(mod_glm_gr, DBgr)
  y_glm_pred_grfr = predict(mod_glm_gr, DBfr[fr_keep, ])
  
  loss_glm_frgr = round(loss.reg(ygr, y_glm_pred_frgr, NULL, suffix="FR.GR"), 2)
  loss_glm_grfr = round(loss.reg(yfr[fr_keep], y_glm_pred_grfr, NULL, suffix="GR.FR"), 2)
  loss_glm_frfr = round(loss.reg(yfr, y_glm_pred_frfr, NULL, suffix="FR.FR"), 2)
  loss_glm_grgr = round(loss.reg(ygr, y_glm_pred_grgr, NULL, suffix="GR.GR"), 2)
  
  # result to save
  # GLM
  result = list(PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
  mod_glm_fr=mod_glm_fr, mod_glm_gr=mod_glm_gr, glm_coef_fr=glm_coef_fr, glm_coef_gr=glm_coef_gr,
  y_glm_pred_frgr=y_glm_pred_frgr,
  y_glm_pred_grfr=y_glm_pred_grfr,
  y_glm_pred_frfr=y_glm_pred_frfr,
  loss_glm_frgr=loss_glm_frgr,
  loss_glm_grfr=loss_glm_grfr,
  loss_glm_frfr=loss_glm_frfr,
  loss_glm_grgr=loss_glm_grgr,
  # DATA          
  Xfr=Xfr, Xgr=Xgr, yfr=yfr, ygr=ygr, fr_keep=fr_keep, DBfr=DBfr, DBgr=DBgr)
                
  if(dim(Xfr)[2]>1){
    result_enet = list(# ENET
      mod_enet_fr=mod_enet_fr, mod_enet_gr=mod_enet_gr, enet_coef_fr=enet_coef_fr, enet_coef_gr=enet_coef_gr,
      y_enet_pred_frgr=y_enet_pred_frgr,
      y_enet_pred_grfr=y_enet_pred_grfr,
      y_enet_pred_frfr=y_enet_pred_frfr,
      loss_enet_frgr=loss_enet_frgr,
      loss_enet_grfr=loss_enet_grfr,
      loss_enet_frfr=loss_enet_frfr,
      loss_enet_grgr=loss_enet_grgr)
    result = c(result, result_enet)
  }
  save(result, file=paste(PREXIF, ".Rdata", sep=""))

  # results
  res = data.frame(MODEL="GLM", TARGET=TARGET, PREDICTORS=PREDICTORS_STR, dim_fr=paste(dim(Xfr), collapse="x"), dim_gr=paste(dim(Xgr), collapse="x"),
                       as.list(c(loss_glm_frgr, loss_glm_grfr)),
                       nzero_fr=(length(glm_coef_fr)-1), nzero_gr=(length(glm_coef_gr)-1),
                       coef_fr=paste(names(glm_coef_fr), collapse=", "),
                       coef_gr=paste(names(glm_coef_gr), collapse=", "),
                       coef_fr_val=paste(glm_coef_fr, collapse=", "),
                       coef_gr_val=paste(glm_coef_gr, collapse=", "))
  if(dim(Xfr)[2]>1){
  res_enet = data.frame(MODEL="ENET", TARGET=TARGET, PREDICTORS=PREDICTORS_STR, dim_fr=paste(dim(Xfr), collapse="x"), dim_gr=paste(dim(Xgr), collapse="x"),
                   as.list(c(loss_enet_frgr, loss_enet_grfr)),
                    nzero_fr=enet_nzero_fr, nzero_gr=enet_nzero_gr,
                    coef_fr=paste(names(enet_coef_fr), collapse=", "),
                    coef_gr=paste(names(enet_coef_gr), collapse=", "),
                    coef_fr_val=paste(enet_coef_fr, collapse=", "),
                    coef_gr_val=paste(enet_coef_gr, collapse=", "))
  res = rbind(res, res_enet)
  }
  if(is.null(RESULTS)) RESULTS = res else RESULTS = rbind(RESULTS, res)
  #if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}
}
r = RESULTS
#diff_tot=sum(r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", c("r2_FR.GR","r2_GR.FR")] - r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC", c("r2_FR.GR", "r2_GR.FR")])
diff_fr=sum(r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_GR.FR"] - r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC", "r2_GR.FR"])    
diff_gr=sum(r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC+NIGLOB", "r2_FR.GR"] - r[r$TARGET!="BARTHEL.M36" & r$PREDICTORS=="BASELINE+CLINIC", "r2_FR.GR"])    

#cat(seed, diff_fr, diff_gr, "\n")
#}

write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)
###
###

