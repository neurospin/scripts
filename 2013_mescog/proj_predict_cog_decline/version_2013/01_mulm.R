#install.packages("glmnet")
#require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#setwd(WD)
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140121.csv", sep="/")
OUTPUT = sub(".csv", "", INPUT_DATA)
OUTPUT_MULM = paste(OUTPUT, "_mulm_targets-x-predictors.csv")
OUTPUT_EVOL = paste(OUTPUT, "_targets_evolution.csv")
OUTPUT_EVOL_PLOT = paste(OUTPUT, "_targets_evolution_plot.svg")


source(paste(SRC,"utils.R",sep="/"))
OUTPUT = paste(BASE_DIR, "results_20140121", sep="/")
if (!file.exists(OUTPUT)) dir.create(OUTPUT)
LOG_FILE = "log.txt"

OUTPUT_SUMMARY = paste(OUTPUT, "results_mulm.csv")

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA, c())
dim(db$DB_FR)# 239  42
dim(db$DB_GE)# 126  42

DBFR = db$DB_FR
DBGR = db$DB_GE

################################################################################################
## M36~each variable
################################################################################################
RESULTS = NULL

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  dbfr = db$DB_FR[!is.na(db$DB_FR[, TARGET]), ]
  dbgr = db$DB_GE[!is.na(db$DB_GE[, TARGET]), ]
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

write.csv(RESULTS, OUTPUT_MULM, row.names=FALSE)


######################################################################
## visualize evoltion
DB = NULL
STAT = NULL
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  dbfr = data.frame(VAR=BASELINE, db$DB_FR[!is.na(db$DB_FR[, TARGET]), c("ID", "SITE", BASELINE, TARGET)])
  dbgr = data.frame(VAR=BASELINE, db$DB_GE[!is.na(db$DB_GE[, TARGET]), c("ID", "SITE", BASELINE, TARGET)])
  colnames(dbfr) = c("VAR", "ID", "SITE", "BASELINE", "M36")
  colnames(dbgr) = c("VAR", "ID", "SITE", "BASELINE", "M36")
  stat = data.frame(
    VAR=BASELINE,
    SITE=c("FR", "GR"),
    COEF=c(lm(M36~BASELINE-1, dbfr)$coefficients[[1]], lm(M36~BASELINE-1, dbgr)$coefficients[[1]]))
  if(is.null(stat)) STAT = stat else STAT = rbind(STAT, stat)
  if(is.null(DB)) DB = rbind(dbfr, dbgr) else DB = rbind(DB, rbind(dbfr, dbgr))  
}

write.csv(STAT, OUTPUT_EVOL, row.names=FALSE)

# VAR SITE      COEF
# 1   TMTB_TIME   FR 0.9895366
# 2   TMTB_TIME   GR 0.9039351
# 3  MDRS_TOTAL   FR 0.9839453
# 4  MDRS_TOTAL   GR 0.9971815
# 5         MRS   FR 1.0371991
# 6         MRS   GR 0.8804348
# 7     BARTHEL   FR 0.9647207
# 8     BARTHEL   GR 0.9986662
# 9        MMSE   FR 1.0077605
# 10       MMSE   GR 1.0016716

svg(OUTPUT_EVOL_PLOT, width = 10, height = 7)
p = ggplot(DB, aes(x = BASELINE, y = M36)) + geom_point(aes(colour=SITE), alpha=.5, position = "jitter") + geom_abline(linetype="dotted") + 
  stat_smooth(formula=y~x-1, method="lm", aes(colour=SITE)) +  facet_wrap(~VAR, scales="free") + scale_color_manual(values=c("blue", "red"))
print(p)
dev.off()
