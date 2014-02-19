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
D$TMTB_TIME_CHANGE = (D$TMTB_TIME.M36 - D$TMTB_TIME)
D$MDRS_TOTAL_CHANGE = (D$MDRS_TOTAL.M36 - D$MDRS_TOTAL)
D$MRS_CHANGE = (D$MRS.M36 - D$MRS)
D$MMSE_CHANGE = (D$MMSE.M36 - D$MMSE)

## ---------------------------------------------------------------------------------------------
## -- MMSE_CHANGE
## ---------------------------------------------------------------------------------------------

TARGET =  "MMSE.M36"
d = D[!is.na(D[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]

mod = rpart(MMSE_CHANGE~LLV+BPF, data=d)
prune(mod, cp=.015)
#n=198 (121 observations deleted due to missingness)
# node), split, n, deviance, yval
# * denotes terminal node
# 
# 1) root 198 1057.44400  0.2556633  
# 2) LLV>=1964.283 7   93.71429 -6.4285710 *
#   3) LLV< 1964.283 191  639.51460  0.5006352  
# 6) BPF< 0.8187689 66  271.72180  0.1003232 *
#   7) BPF>=0.8187689 125  351.63200  0.7120000 *

partitions = c("LLV>=1964.283", "BPF<0.8187689", "MMSE<26") # found with rpart
partitions = c("LLV>=1964.283", "BPF<0.75", "MMSE<26") #

mlm = partmlm.learn(data=d, TARGET, BASELINE, partitions)
# "MMSE>=25.5" have inddeed the same intercept than the one of group 5 ("BPF>=0.773")
#mod$intercepts[3] = mod$intercepts[5]
y_pred_rtree = partmlm.predict(mlm, data=d, limits=c(0, 30))
d$MMSE.M36_pred = y_pred_rtree 
loss_partmlm = loss_reg(d$MMSE.M36, y_pred_rtree)
print(loss_partmlm)
#mse        r2       cor 
#2.6652135 0.8633871 0.9291862 

pdf(paste(OUTPUT, "rpart_MMSE_CHANGE.pdf", sep="/"))
d$GROUP = factor(attr(y_pred_rtree, "group"), levels=c(partitions, "left"))

p_true_m36_m0 = ggplot(d, aes(x = MMSE, y = MMSE.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MMSE.M36", "~", "MMSE"))+
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted")# +
#print(p_true)
p_pred_m36_m0 = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE, R2=",round(loss_partmlm["r2"][[1]],2)))

ds = summarySE(data=d, "MMSE_CHANGE", "GROUP")
p_true_boxplot = 
  ggplot(d, aes(x = GROUP, y = MMSE_CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MMSE_CHANGE", "~", "GROUP"))

print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_true_boxplot)
dev.off()


## ---------------------------------------------------------------------------------------------
## -- MRS_CHANGE
## ---------------------------------------------------------------------------------------------

TARGET =  "MRS.M36"
d = D[!is.na(D[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]

mod = rpart(MRS_CHANGE~LLV+BPF, data=d)
mod = prune(mod, cp=.025)
# n= 207 
# 
# node), split, n, deviance, yval
# * denotes terminal node
# 
# 1) root 207 136.850200  0.27053140  
# 2) LLV< 2083.204 199 121.899500  0.23618090  
# 4) BPF>=0.8751428 36   7.888889 -0.05555556 *
#   5) BPF< 0.8751428 163 110.269900  0.30061350 *
#   3) LLV>=2083.204 8   8.875000  1.12500000 *

plot(mod, uniform=TRUE, main=paste(TARGET, "~LLV+BPF"))
text(mod, use.n=TRUE, all=TRUE, cex=.8)

partitions = c("LLV>2083.204", "BPF<0.8751428") # found with rpart
partitions = c("LLV>2083.204", "BPF<.75") # found with rpart

#partitions = c("LLV>=1964.283", "BPF<0.75", "MRS<26") #

mlm = partmlm.learn(data=d, TARGET, BASELINE, partitions)
# "MRS>=25.5" have inddeed the same intercept than the one of group 5 ("BPF>=0.773")
#mod$intercepts[3] = mod$intercepts[5]
y_pred_rtree = partmlm.predict(mlm, data=d, limits=c(0, 30))
d$MRS.M36_pred = y_pred_rtree 
loss_partmlm = loss_reg(d$MRS.M36, y_pred_rtree)
print(loss_partmlm)
#mse        r2       cor 
#2.6652135 0.8633871 0.9291862 

pdf(paste(OUTPUT, "rpart_MRS_CHANGE.pdf", sep="/"))
d$GROUP = factor(attr(y_pred_rtree, "group"), levels=c(partitions, "left"))

p_true_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MRS.M36", "~", "MRS"))+
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted")# +
#print(p_true)
p_pred_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MRS.M36", "~", "MRS, R2=",round(loss_partmlm["r2"][[1]],2)))

ds = summarySE(data=d, "MRS_CHANGE", "GROUP")
p_true_boxplot = 
  ggplot(d, aes(x = GROUP, y = MRS_CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MRS_CHANGE", "~", "GROUP"))

print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_true_boxplot)
dev.off()
