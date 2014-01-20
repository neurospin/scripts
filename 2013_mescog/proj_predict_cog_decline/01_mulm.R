#install.packages("glmnet")
#require(glmnet)
#require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/2014_mescog_predict_cog_decline"
#setwd(WD)
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
source(paste(SRC,"utils.R",sep="/"))
OUTPUT = paste(BASE_DIR, "results_201401", sep="/")
LOG_FILE = "log.txt"

OUTPUT_SUMMARY = paste(OUTPUT, "results_mulm.csv")

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB_FR)# 239  42
dim(db$DB_GR)# 126  42

DBFR = db$DB_FR
DBGR = db$DB_GR

################################################################################################
## M36~each variable
################################################################################################
RESULTS = NULL

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  dbfr = db$DB_FR[!is.na(db$DB_FR[, TARGET]), ]
  dbgr = db$DB_GR[!is.na(db$DB_GR[, TARGET]), ]
  RES = NULL
  for(PRED in c(db$col_clinic, db$col_niglob)){
    #PRED="TMTB_TIME"
    #PRED= "MDRS_TOTAL"
    formula = formula(paste(TARGET,"~",PRED))
    modfr = lm(formula, data = dbfr)
    modgr = lm(formula, data = dbgr)
    
    loss.frfr=round(loss.reg(dbfr[, TARGET], predict(modfr, dbfr), df=2),digit=2)[c("R2", "cor")]
    loss.frgr=round(loss.reg(dbgr[, TARGET], predict(modfr, dbgr), df=2),digit=2)[c("R2", "cor")]
    loss.grgr=round(loss.reg(dbgr[, TARGET], predict(modgr, dbgr), df=2),digit=2)[c("R2", "cor")]
    loss.grfr=round(loss.reg(dbfr[, TARGET], predict(modgr, dbfr), df=2),digit=2)[c("R2", "cor")]
    
    names(loss.frfr) = c("r2_ff","cor_ff")
    names(loss.frgr) = c("r2_fg","cor_fg")
    names(loss.grgr) = c("r2_gg","cor_gg")
    names(loss.grfr) = c("r2_gf","cor_gf")
    res = data.frame(target=TARGET, pred=PRED, as.list(c(loss.frfr, loss.frgr, loss.grgr, loss.grfr)))
    if(is.null(RES)) RES = res else RES = rbind(RES, res)
  }
  RES = RES[order(RES$r2_fg, decreasing=TRUE),]
  if(is.null(RESULTS)) RESULTS = RES else RESULTS = rbind(RESULTS, RES)
}

write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)

