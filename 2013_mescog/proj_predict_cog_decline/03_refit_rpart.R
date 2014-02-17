library(rpart)
library(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
source(paste(SRC,"utils.R",sep="/"))

BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT = "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed"
OUTPUT=INPUT

db = read_db(INPUT_DATA)

## REFIT ALL -------------------------------------------------------------------------------------------
ERR = NULL
SUMMARY = NULL

pdf(paste(OUTPUT, "refitall_rpart_tree_M36_by_M0.pdf", sep="/"))

for(TARGET in db$col_targets){
  # TARGET = "TMTB_TIME.M36"
  # TARGET =  "MMSE.M36"
  cat("== ", TARGET, " =============================================================================== \n" )
  d = db$DB[!is.na(db$DB[, TARGET]),]
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  # NO NI
  PREDICTORS_STR = "BASELINE"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = BASELINE
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod = rpart(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod, d)
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  plot(mod, uniform=TRUE, main=paste(TARGET, "~", PREDICTORS_STR, "R2=", round(loss_reg(y_true=M36_true, y_pred=M36_pred)["r2"][[1]],2)))
  text(mod, use.n=TRUE, all=TRUE, cex=.8)
  ERR=rbind(ERR, data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs))
  print(mod)
  
  # With NI
  PREDICTORS_STR = "BASELINE+NIGLOB"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = unique(c(BASELINE,db$col_niglob))
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod_ni = rpart(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod_ni, d)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  plot(mod_ni, uniform=TRUE, main=paste(TARGET, "~", PREDICTORS_STR, "R2=", round(loss_reg(y_true=M36_true, y_pred=M36_pred)["r2"][[1]],2)))
  text(mod_ni, use.n=TRUE, all=TRUE, cex=.8)
  
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  r = data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs)
  ERR=rbind(ERR, r)
  print(mod_ni)
  ## PLOT
  r$GROUP = as.factor(mod_ni$where)
  p = ggplot(r, aes(x = M0, y = M36_true)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
    geom_abline(linetype="dotted") + ggtitle(paste(TARGET, "~", PREDICTORS_STR))
  #x11()
  print(p)
  ##
}

write.csv(ERR, paste(OUTPUT, "error_refitall_rpart_M36_by_M0.csv", sep="/"), row.names=FALSE)

dev.off()
#write.csv(ERR, paste(OUTPUT, "error_refitallrpart_inter_M36_by_M0.csv", sep="/"), row.names=FALSE)

#########################################################################################################
## MMSE

if(FALSE){ 
  ## REFIT ALL "MMSE.M36"
TARGET =  "MMSE.M36"
d = db$DB[!is.na(db$DB[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]

#tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
tree = c("LLV>=1592", "BPF<0.773")
mod = rpart_inter.learn(data=d, TARGET="MMSE.M36", BASELINE="MMSE", tree=tree)
# "MMSE>=25.5" have inddeed the same intercept than the one of group 5 ("BPF>=0.773")
#mod$intercepts[3] = mod$intercepts[5]
y_pred_rtree = rpart_inter.predict(mod, data=d, limits=c(0, 30))
d$MMSE.M36_pred = y_pred_rtree 
loss_rtree = loss_reg(d$MMSE.M36, y_pred_rtree)

pdf(paste(OUTPUT, "refitall_rpart_intercept_MMSE_M36_by_M0.pdf", sep="/"))
#pdf(paste(OUTPUT, "refitall_partmlm_MMSE_M36_by_M0.pdf", sep="/"))
d$GROUP = as.factor(attr(y_pred_rtree, "group"))
p_true = ggplot(d, aes(x = MMSE, y = MMSE.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE"))+
  geom_smooth(method="lm")
print(p_true)


p_pred = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE, R2=",round(loss_rtree["r2"][[1]],2)))
print(p_true)
print(p_pred)
dev.off()


## CV  "MMSE.M36" ---------------------------------------------------------------------------
TARGET =  "MMSE.M36"
#d = db$DB[!is.na(db$DB[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]
tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
DB = db$DB
SPLITS = kfold_site_stratified_rm_na(DB, TARGET, k=5)
y_true_tr = y_true_te = y_pred_te_partinter = y_pred_tr_partinter  = y_pred_tr_glm = y_pred_te_glm = group = m0 = id = c()
for(FOLD in 1:length(SPLITS)){
  #FOLD = 1
  D_tr = SPLITS[[FOLD]]$tr
  D_te = SPLITS[[FOLD]]$te  
  y_true_tr_fold = D_tr[, TARGET]
  y_true_te_fold = D_te[, TARGET]
  m0 = c(m0, SPLITS[[FOLD]]$te[, BASELINE])
  id = c(id, SPLITS[[FOLD]]$te[, "ID"])
  y_true_tr = c(y_true_tr, y_true_tr_fold)
  y_true_te = c(y_true_te, y_true_te_fold)
  
  # rpart_inter -------------------
  mod_rpart_inter = rpart_inter.learn(data=D_tr, TARGET="MMSE.M36", BASELINE="MMSE", tree=tree)
  # "MMSE>=25.5" have inddeed the same intercept than the one of group 5 ("BPF>=0.773")
  mod_rpart_inter$intercepts[3] = mod_rpart_inter$intercepts[5]
  y_pred_te_partinter_fold = rpart_inter.predict(mod_rpart_inter, D_te, limits=c(0, 30))
  group = c(group, attr(y_pred_te_partinter_fold,"group"))
  y_pred_tr_partinter_fold = rpart_inter.predict(mod_rpart_inter, D_tr, limits=c(0, 30))
  #loss_tr_rpart_inter = loss_reg(y_true_tr_fold,  y_pred_tr_partinter_fold, NULL, suffix="tr")
  print(loss_te_rpart_inter)
  y_pred_tr_partinter = c(y_pred_tr_partinter, y_pred_tr_partinter_fold)
  y_pred_te_partinter = c(y_pred_te_partinter, y_pred_te_partinter_fold)
  
  # LM
  formula = formula(paste(TARGET,"~", BASELINE))
  mod_glm = lm(formula, data=D_tr)
  y_pred_te_glm_fold = predict(mod_glm, D_te)
  y_pred_tr_glm_fold = predict(mod_glm, D_tr)
  #loss_tr_rpart_inter = loss_reg(y_true_tr_fold,  y_pred_tr_glm_fold, NULL, suffix="tr")
  y_pred_tr_glm = c(y_pred_tr_glm, y_pred_tr_glm_fold)
  y_pred_te_glm = c(y_pred_te_glm, y_pred_te_glm_fold)
  
}
loss_partinter = loss_reg(y_true_te, y_pred_te_partinter, NULL, suffix="te")
loss_glm = loss_reg(y_true_te, y_pred_te_glm, NULL, suffix="te")

CV = data.frame(ID=id, M0=m0, M36_true=y_true_te, 
                M36_pred_te_partinter = y_pred_te_partinter, GROUP=as.factor(group),
                M36_pred_te_glm = y_pred_te_glm)

pdf(paste(OUTPUT, "cv_rpart_intercept_MMSE_M36_by_M0.pdf", sep="/"))
p_true = ggplot(CV, aes(x = M0, y = M36_true)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE"))
p_pred = ggplot(CV, aes(x = M0, y = M36_pred_te_partinter)) +
  geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_point(alpha=.7, aes(y=M36_pred_te_glm), position = "jitter") +
  geom_abline(linetype="dotted") + ggtitle(paste("CV rpart_inter MMSE.M36", "~", "MMSE, R2=",round(loss_partinter["r2_te"][[1]],2)))
print(p_true)
print(p_pred)
dev.off()

}
