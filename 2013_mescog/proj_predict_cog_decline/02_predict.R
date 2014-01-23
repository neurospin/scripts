#install.packages("glmnet")
require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#setwd(WD)
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140121.csv", sep="/")

source(paste(SRC,"utils.R",sep="/"))
#TO_REMOVE = c("LLVn", "WMHVn")

################################################################################################
##OUTPUT = paste(BASE_DIR, "20140115_all-predictors", sep="/")
TO_REMOVE = c("BRAINVOL")
  
################################################################################################
OUTPUT = paste(BASE_DIR, "20140120_remove-predictors", sep="/")
TO_REMOVE = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
              "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
              "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
              "LLVn", "WMHVn", "BRAINVOL", "LLcount")

################################################################################################
# OUTPUT = paste(BASE_DIR, "20140120_remove-predictors_butnotLLcount", sep="/")
# TO_REMOVE = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
#               "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
#               "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
#               "LLVn", "WMHVn", "BRAINVOL")


if (!file.exists(OUTPUT)) dir.create(OUTPUT)
OUTPUT_SUMMARY = paste(OUTPUT, "results_summary.csv", sep="/")

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA, TO_REMOVE)
dim(db$DB_FR)# 239  42 # 244  46
dim(db$DB_GR)# 126  42 # 128  46

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
  DBgr = db$DB_GR[!is.na(db$DB_GR[, TARGET]), ]
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

################################################################################################
## FULL COMPLEX PIPELINE
################################################################################################
MOD.SEL.CV = "manual.cv.lambda.min"
#MOD.SEL.CV = "manual.cv.lambda.min.glm"
#MOD.SEL.CV = "manual.cv.lambda.1sd"
#MOD.SEL.CV = "manual.cv.lambda.1sd.glm"
#MOD.SEL.CV = "auto.max.r2"
#MOD.SEL.CV = "auto.max.cor"
#MOD.SEL.CV = "auto.min.mse"

BOOTSTRAP.NB=100; PERMUTATION.NB=100
DATA_STR = "FR"
#DATA_STR = "GR"

################################################################################################
## GLM: M36~BASELINE
################################################################################################
RESULTS = NULL


if(DATA_STR == "FR"){
  DBLEARN = db$DB_FR
  DBTEST = db$DB_GR
}
if(DATA_STR == "GR"){
  DBLEARN = db$DB_GR
  DBTEST = db$DB_FR
}

PREDICTORS_STR = "BASELINE_NOINTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  INTERCEPT=FALSE;bootstrap.nb=5;permutation.nb=5
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, FALSE, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                                   bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}
PREDICTORS_STR = "BASELINE"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                               bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}

################################################################################################
## GLM: M36~NIGLOB
################################################################################################
PREDICTORS_STR = "NIGLOB"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  PREDICTORS = db$col_niglob
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                               bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}

################################################################################################
## ENET: M36~NIGLOB
################################################################################################
PREDICTORS_STR = "NIGLOB"

for(TARGET in db$col_targets){
  #TARGET =  "MMSE.M36"
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MMSE.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = db$col_niglob
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  #source(paste(SRC,"utils.R",sep="/"))
  #bootstrap.nb=5; permutation.nb=5
  res = do.a.lot.of.things.glmnet(DBtr, DBte, TARGET, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE, FORCE.ALPHA, MOD.SEL.CV,
                                  bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(res) else RESULTS = rbind(RESULTS, data.frame(res))
}

################################################################################################
## GLM: M36~BASELINE+NIGLOB
################################################################################################
PREDICTORS_STR = "BASELINE+NIGLOB_NOINTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(BASELINE, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                               bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}
PREDICTORS_STR = "BASELINE+NIGLOB"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(BASELINE, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                               bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}

################################################################################################
## ENET: M36~BASELINE+CLINIC
################################################################################################
PREDICTORS_STR = "BASELINE+CLINIC"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = db$col_clinic
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  #source(paste(SRC,"utils.R",sep="/"))
  #bootstrap.nb=5; permutation.nb=5
  res = do.a.lot.of.things.glmnet(DBtr, DBte, TARGET, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE, FORCE.ALPHA, MOD.SEL.CV,
                                  bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(res) else RESULTS = rbind(RESULTS, data.frame(res))
}


################################################################################################
## ENET: M36~BASELINE+NIGLOB
################################################################################################
PREDICTORS_STR = "BASELINE+NIGLOB"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(BASELINE, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  #source(paste(SRC,"utils.R",sep="/"))
  #bootstrap.nb=5; permutation.nb=5
  res = do.a.lot.of.things.glmnet(DBtr, DBte, TARGET, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE, FORCE.ALPHA, MOD.SEL.CV,
                                bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(res) else RESULTS = rbind(RESULTS, data.frame(res))
}


################################################################################################
## ENET: M36~CLINIC+NIGLOB
################################################################################################
PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(db$col_clinic, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), ]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), ]
  res = do.a.lot.of.things.glmnet(DBtr, DBte, TARGET, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE, FORCE.ALPHA, MOD.SEL.CV,
                                  bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(res) else RESULTS = rbind(RESULTS, data.frame(res))
}

#d1 = read.csv(OUTPUT_SUMMARY)
#RESULTS = rbind(RESULTS, d1)
write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)

# rsync -azvun --delete /neurospin/mescog/2014_mescog_predict_cog_decline ~/data/



# 
# for(TARGET in db$col_targets){
#     #TARGET = "TMTB_TIME.M36"
#     #TARGET = "MDRS_TOTAL.M36"
#     PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
#     PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
#     if (!file.exists(PREFIX)) dir.create(PREFIX)
#     setwd(PREFIX)
#     DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
#     DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
#     ## FR CV
#     y = DBtr[,TARGET]
#     cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
#     dump("cv",file="cv.schema.R")
#     
#     y.pred.cv = c();  y.true.cv = c()
#     for(test in cv){
#         DBtr_train = DBtr[-test,]
#         DBtr_test = DBtr[test,]
#         formula = formula(paste(TARGET,"~",PREDICTORS))
#         modlm = lm(formula, data = DBtr_train)
#         y.pred.cv = c(y.pred.cv, predict(modlm, DBtr_test))
#         y.true.cv = c(y.true.cv, DBtr_test[,TARGET])
#     }
#     loss.cv = round(loss.reg(y.true.cv, y.pred.cv, df2=2),digit=2)
#     ## CV predictions
#     ## ==============
# 
#     cat("\nCV predictions\n",file=LOG_FILE,append=TRUE)
#     cat("--------------\n",file=LOG_FILE,append=TRUE)
# 
#     sink(LOG_FILE, append = TRUE)
#     print(loss.cv)
#     sink()
# 
#     ## Fit and predict all (same train) data, (look for overfit)
#     ## =========================================================
# 
#     cat("\nFit and predict all (same train) data, (look for overfit)\n",file=LOG_FILE,append=TRUE)
#     cat("---------------------------------------------------------\n",file=LOG_FILE,append=TRUE)
#     #formula = formula(paste(TARGET,"~", PREDICTORS))
#     modlm.fr = lm(formula, data = DBtr)
#     y.pred.all = predict(modlm.fr, DBtr)
#     loss.all = round(loss.reg(y, y.pred.all, df=2),digit=2)
#     sink(LOG_FILE, append = TRUE)
#     print(loss.all)
#     sink()
#     
#     ## Plot true vs predicted
#     ## ======================
#     
#     pdf("cv_glm_true-vs-pred.pdf")
#     p = qplot(y.true.cv, y.pred.cv, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",loss.cv[["R2"]],"]",sep=""))
#     print(p); dev.off()
#   
#     # - true vs. pred (no CV)
#     
#     #svg("all.bestcv.glmnet.true-vs-pred.svg")
#     pdf("all_glm_true-vs-pred.pdf")
#     p = qplot(y, y.pred.all, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - no CV - [R2=", loss.all[["R2"]],"]",sep=""))
#     print(p); dev.off()
# 
#     cat("\nGeneralize on Test dataset:\n",file=LOG_FILE,append=TRUE)
#     cat("-------------------------------\n",file=LOG_FILE,append=TRUE)
#     y.preDBtr = predict(modlm.fr, DBtr)
#     y.preDBte  = predict(modlm.fr, DBte)
# 
#     loss.fr = round(loss.reg(DBtr[,TARGET], y.preDBtr, df2=2), digit=2)
#     loss.d  = round(loss.reg(DBte[,TARGET],  y.preDBte,  df2=2), digit=2)
#     
#     pdf("all_glm_true-vs-preDBte.pdf")
#     y.preDBte = as.vector(y.preDBte)
#     p = qplot(DBte[,TARGET], y.preDBte, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - TEST - [R2=", loss.d[["R2"]] ,"]",sep=""))
#     print(p);dev.off()    
#     
#     sink(LOG_FILE, append = TRUE)
#     print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
#     sink()
#     
#     res = data.frame(data=DATA_STR, target=TARGET, predictors="BASELINE", dim=paste(dim(DBtr)[1], dim(DBtr)[2]-1, sep="x"),
#       r2_cv=loss.cv["R2"], cor_cv=loss.cv["cor"], fstat_cv=loss.cv["fstat"],
#       r2_all=loss.all["R2"], cor_all=loss.all["R2"], r2_test=loss.d["R2"], cor_test=loss.d["cor"], fstat_test=loss.d["fstat"])
#     if(is.null(RESULTS)) RESULTS = res else RESULTS = rbind(RESULTS, res)
# }