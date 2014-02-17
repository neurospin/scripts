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


TARGET =  "MMSE.M36"
d = db$DB[!is.na(db$DB[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]

#tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
#partitions = c("LLV>=1592", "BPF<0.773")
partitions = c("LLV>=1500", "BPF<0.75", "MMSE<26")

mlm = partmlm.learn(data=d, TARGET, BASELINE, partitions)
# "MMSE>=25.5" have inddeed the same intercept than the one of group 5 ("BPF>=0.773")
#mod$intercepts[3] = mod$intercepts[5]
y_pred_rtree = partmlm.predict(mlm, data=d, limits=c(0, 30))
d$MMSE.M36_pred = y_pred_rtree 
loss_partmlm = loss_reg(d$MMSE.M36, y_pred_rtree)

pdf(paste(OUTPUT, "refitall_partmlm_MMSE_M36_by_M0.pdf", sep="/"))
d$GROUP = as.factor(attr(y_pred_rtree, "group"))
p_true = ggplot(d, aes(x = MMSE, y = MMSE.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MMSE.M36", "~", "MMSE"))+
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted")# +
#print(p_true)
p_pred = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE, R2=",round(loss_partmlm["r2"][[1]],2)))
print(p_true)
print(p_pred)
dev.off()

svg(paste(OUTPUT, "refitall_partmlm_MMSE_M36_by_M0.svg", sep="/"))
print(p_true)
dev.off()

## CV  "MMSE.M36" ---------------------------------------------------------------------------
TARGET =  "MMSE.M36"
#d = db$DB[!is.na(db$DB[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]
#tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
DB = db$DB
SPLITS = kfold_site_stratified_rm_na(DB, TARGET, k=5)
y_true_tr = y_true_te = y_pred_te_partmlm = y_pred_tr_partmlm  = y_pred_tr_glm = y_pred_te_glm = group = m0 = id = c()
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
  
  # partmlm -------------------
  mod_partmlm = partmlm.learn(data=D_tr, TARGET="MMSE.M36", BASELINE="MMSE", partitions=partitions)
  y_pred_te_partmlm_fold = partmlm.predict(mod_partmlm, D_te, limits=c(0, 30))
  group = c(group, attr(y_pred_te_partmlm_fold,"group"))
  y_pred_tr_partmlm_fold = partmlm.predict(mod_partmlm, D_tr, limits=c(0, 30))
  #loss_tr_partmlm = loss_reg(y_true_tr_fold,  y_pred_tr_partmlm_fold, NULL, suffix="tr")
  #print(loss_te_partmlm)
  y_pred_tr_partmlm = c(y_pred_tr_partmlm, y_pred_tr_partmlm_fold)
  y_pred_te_partmlm = c(y_pred_te_partmlm, y_pred_te_partmlm_fold)
  
  # LM
  formula = formula(paste(TARGET,"~", BASELINE))
  mod_glm = lm(formula, data=D_tr)
  y_pred_te_glm_fold = predict(mod_glm, D_te)
  y_pred_tr_glm_fold = predict(mod_glm, D_tr)
  #loss_tr_partmlm = loss_reg(y_true_tr_fold,  y_pred_tr_glm_fold, NULL, suffix="tr")
  y_pred_tr_glm = c(y_pred_tr_glm, y_pred_tr_glm_fold)
  y_pred_te_glm = c(y_pred_te_glm, y_pred_te_glm_fold)
}


loss_partmlm = loss_reg(y_true_te, y_pred_te_partmlm, NULL, suffix="te")
loss_glm = loss_reg(y_true_te, y_pred_te_glm, NULL, suffix="te")

CV = data.frame(ID=id, M0=m0, M36_true=y_true_te, 
                M36_pred_te_partmlm = y_pred_te_partmlm, GROUP=as.factor(group),
                M36_pred_te_glm = y_pred_te_glm)

pdf(paste(OUTPUT, "cv_partmlm_MMSE_M36_by_M0.pdf", sep="/"))
p_true = ggplot(CV, aes(x = M0, y = M36_true, group=GROUP, colour=GROUP)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_smooth(method="lm", se=F) +
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE"))

p_pred = ggplot(CV, aes(x = M0, y = M36_pred_te_partmlm, group=GROUP)) +
  geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_point(alpha=.7, aes(y=M36_pred_te_glm), position = "jitter") +
  geom_abline(linetype="dotted") + ggtitle(paste("CV partmlm MMSE.M36", "~", "MMSE, R2=",round(loss_partmlm["r2_te"][[1]],2)))
print(p_true)
print(p_pred)
dev.off()

