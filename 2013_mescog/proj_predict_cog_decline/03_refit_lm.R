SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
source(paste(SRC,"utils.R",sep="/"))

BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140128_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140204_nomissing_BPF-LLV_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
OUTPUT = "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed"

VALIDATION = "CV"

db = read_db(INPUT_DATA)

## -------------------------------------------------------------------------------------------
ERR = NULL
SUMMARY = NULL


for(TARGET in db$col_targets){
  # TARGET = "TMTB_TIME.M36"
  cat("== ", TARGET, " =============================================================================== \n" )
  d = db$DB[!is.na(db$DB[, TARGET]),]
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  # NO NI
  PREDICTORS_STR = "BASELINE"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = BASELINE
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod = lm(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod, d)
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  ERR=rbind(ERR, data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs))
  print(summary(mod))
  
  # With NI
  PREDICTORS_STR = "BASELINE+NIGLOB"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = unique(c(BASELINE,db$col_niglob))
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod_ni = lm(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod_ni, d)
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  ERR=rbind(ERR, data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs))
  print(summary(mod_ni))
  
  cat("-- COMPARISON (ANOVA) ----------------------------------------------------------------------- \n" )
  print(anova(mod, mod_ni))
}

write.csv(ERR, paste(OUTPUT, "error_refitallglm-nointer_M36_by_M0.csv", sep="/"), row.names=FALSE)
#write.csv(ERR, paste(OUTPUT, "error_refitallglm_M36_by_M0.csv", sep="/"), row.names=FALSE)