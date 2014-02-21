SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "rpart", sep="/")
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

