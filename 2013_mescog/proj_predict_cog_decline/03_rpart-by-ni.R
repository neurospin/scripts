SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "rpart", sep="/")
# "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed/rpart"
if (!file.exists(OUTPUT)) dir.create(OUTPUT)

source(paste(SRC,"utils.R",sep="/"))

db = read_db(INPUT_DATA)
D = db$DB

################################################################################################
## Find best cut-off on LLV and BPF to explain M36-M0 using rpart
################################################################################################
library(rpart)
library(ggplot2)

D$TMTB_TIME.CHANGE = (D$TMTB_TIME.M36 - D$TMTB_TIME)
D$MDRS_TOTAL.CHANGE = (D$MDRS_TOTAL.M36 - D$MDRS_TOTAL)
D$MRS.CHANGE = (D$MRS.M36 - D$MRS)
D$MMSE.CHANGE = (D$MMSE.M36 - D$MMSE)

## ---------------------------------------------------------------------------------------------
## -- MMSE.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "MMSE.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod = rpart(MMSE.CHANGE~MMSE+LLV+BPF, data=d)
rpart_mod = prune(rpart_mod, cp=.02)

plot(rpart_mod, uniform=TRUE, main=paste(M36, "~LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)

#rpart_mod = prune(rpart_mod, cp=.015)
#n=198 (121 observations deleted due to missingness)
# node), split, n, deviance, yval
# * denotes terminal node
# 
# 1) root 198 1057.44400  0.2556633  
# 2) LLV>=1964.283 7   93.71429 -6.4285710 *
#   3) LLV< 1964.283 191  639.51460  0.5006352  
# 6) BPF< 0.8187689 66  271.72180  0.1003232 *
#   7) BPF>=0.8187689 125  351.63200  0.7120000 *

group = rpart.groups(rpart_mod, data=d)
mlm_mod = subgrouplm.learn(group, d, M36, BASELINE)
subgrouplm.predict(mlm_mod, group, d, limits=c(0, 30))
#subgrouplm.rpart_mod, group, data, limits=c(-Inf, +Inf)
y_pred_rtree = subgrouplm.predict(mlm_mod, group, data=d, limits=c(0, 30))
d$MMSE.M36_pred = y_pred_rtree 
loss_partmlm_mod = loss_reg(d$MMSE.M36, y_pred_rtree)
print(loss_partmlm_mod)
#mse        r2       cor 
#2.4549013 0.8741672 0.9349691

label = unlist(attr(attr(y_pred_rtree, "group"), "name"))

d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

p_true_m36_m0 = ggplot(d, aes(x = MMSE, y = MMSE.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MMSE.M36", "~", "MMSE"))+
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted")# +
#print(p_true)
p_pred_m36_m0 = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE, R2=",round(loss_partmlm_mod["r2"][[1]],2)))

ds = summarySE(data=d, "MMSE.CHANGE", "GROUP")
p_true_boxplot = 
  ggplot(d, aes(x = GROUP, y = MMSE.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MMSE.CHANGE", "~", "GROUP"))

pdf(paste(OUTPUT, "rpart_MMSE.CHANGE.pdf", sep="/"))
plot(rpart_mod, uniform=TRUE, main=paste(M36, "~MMSE+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_true_boxplot)
dev.off()


## ---------------------------------------------------------------------------------------------
## -- MRS.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "MRS.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod = rpart(MRS.CHANGE~MRS+LLV+BPF, data=d)
rpart_mod = prune(rpart_mod, cp=.02)

plot(rpart_mod, uniform=TRUE, main=paste(M36, "~LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)

# 1) root 207 136.850200  0.27053140  
# 2) MRS>=1.5 43  33.860470 -0.16279070  
# 4) LLV< 1251.394 36  23.638890 -0.30555560 *
#   5) LLV>=1251.394 7   5.714286  0.57142860 *
#   3) MRS< 1.5 164  92.798780  0.38414630  
# 6) BPF>=0.7858275 142  60.732390  0.28169010  
# 12) BPF>=0.8572125 66  16.621210  0.07575758 *
#   13) BPF< 0.8572125 76  38.881580  0.46052630  
# 26) LLV< 626.7295 65  24.861540  0.35384620 *
#   27) LLV>=626.7295 11   8.909091  1.09090900 *
#   7) BPF< 0.7858275 22  20.954550  1.04545500 *

group = rpart.groups(rpart_mod, data=d)
mlm_mod = subgrouplm.learn(group, d, M36, BASELINE)
subgrouplm.predict(mlm_mod, group, d, limits=c(0, 30))
#subgrouplm.rpart_mod, group, data, limits=c(-Inf, +Inf)
y_pred_rtree = subgrouplm.predict(mlm_mod, group, data=d, limits=c(0, 30))
d$MRS.M36_pred = y_pred_rtree 
loss_partmlm_mod = loss_reg(d$MRS.M36, y_pred_rtree)
print(loss_partmlm_mod)
# mse        r2       cor 
# 0.5958352 0.6757811 0.8220590 

label = unlist(attr(attr(y_pred_rtree, "group"), "name"))

d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

p_true_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MRS.M36", "~", "MRS"))+
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted")# +
#print(p_true)
p_pred_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MRS.M36", "~", "MRS, R2=",round(loss_partmlm_mod["r2"][[1]],2)))

ds = summarySE(data=d, "MRS.CHANGE", "GROUP")
p_true_boxplot = 
  ggplot(d, aes(x = GROUP, y = MRS.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MRS.CHANGE", "~", "GROUP"))

pdf(paste(OUTPUT, "rpart_MRS.CHANGE.pdf", sep="/"))
plot(rpart_mod, uniform=TRUE, main=paste(M36, "~MRS+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_true_boxplot)
dev.off()

# Simplfy rule 
d$GROUP2=d$GROUP
levels(d$GROUP2)[levels(d$GROUP2) == "/MRS< 1.5/BPF>=0.7858/BPF>=0.8572"] = "/MRS< 1.5/BPF>=0.8572"

#decrease 

m1 = 


groups.l = list()
groups.l["decrease"] = list(labels = "/MRS>=1.5/LLV< 1251")
#=> -0.3

groups.l["stable"] = list(labels = "/MRS< 1.5/BPF>=0.8572")
# => 0

groups.l["moderate increase"] = list(labels = c("/MRS>=1.5/LLV>=1251", "/MRS< 1.5/BPF>=0.7858/BPF< 0.8572/LLV< 626.7"))
#=>  +0.5

groups.l["large increase"] = list(labels = c("/MRS< 1.5/BPF>=0.7858/BPF< 0.8572/LLV>=626.7", "/MRS< 1.5/BPF< 0.7858"))
#=>  +1

for(n in names(groups.l)){
  mask = rep(T, length(d$GROUP2))
  for(l in groups.l[[n]]$labels)  mask = mask & (d$GROUP2 == mask) 
  groups.l[[n]]$mask = mask
}
