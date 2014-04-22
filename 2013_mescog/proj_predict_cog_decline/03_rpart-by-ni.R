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

library(rpart)
library(ggplot2)

D$TMTB_TIME.CHANGE = (D$TMTB_TIME.M36 - D$TMTB_TIME)
D$MDRS_TOTAL.CHANGE = (D$MDRS_TOTAL.M36 - D$MDRS_TOTAL)
D$MRS.CHANGE = (D$MRS.M36 - D$MRS)
D$MMSE.CHANGE = (D$MMSE.M36 - D$MMSE)

################################################################################################
## Find best cut-off on LLV and BPF to explain M36-M0 using rpart
################################################################################################

## ---------------------------------------------------------------------------------------------
## -- MMSE.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "MMSE.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod = rpart(MMSE.CHANGE~MMSE+LLV+BPF, data=d)
rpart_mod = prune(rpart_mod, cp=.02)
loss_reg(d$MMSE.CHANGE, predict(rpart_mod, d))
#mse        r2       cor 
#2.6199247 0.5094349 0.7137471

rpart_mod1 = rpart(MMSE.CHANGE~MMSE+LLV+BPF, data=d)
rpart_mod2 = rpart(MMSE.CHANGE~LLV+BPF, data=d)

loss_reg(d$MMSE.CHANGE, predict(rpart_mod1, d))
#mse        r2       cor 
#2.4612052 0.5391542 0.7342712 
loss_reg(d$MMSE.CHANGE, predict(rpart_mod2, d))
#mse        r2       cor 
#3.1233513 0.4151714 0.6443379 

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
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted") +
  theme(legend.position="bottom", legend.direction="vertical")
#print(p_true)
p_pred_m36_m0 = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE, R2=",round(loss_partmlm_mod["r2"][[1]],2))) +
  theme(legend.position="bottom", legend.direction="vertical")

ds = summarySE(data=d, "MMSE.CHANGE", "GROUP")
p_change_boxplot = 
  ggplot(d, aes(x = GROUP, y = MMSE.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MMSE.CHANGE", "~", "GROUP")) +
  theme(legend.position="bottom", legend.direction="vertical")

pdf(paste(OUTPUT, "rpart_MMSE.CHANGE.pdf", sep="/"))
plot(rpart_mod1, uniform=TRUE, main=paste("MMSE.CHANGE~MMSE+LLV+BPF"))
text(rpart_mod1, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod2, uniform=TRUE, main=paste("MMSE.CHANGE~LLV+BPF"))
text(rpart_mod2, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod, uniform=TRUE, main=paste("MMSE.CHANGE~MMSE+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_change_boxplot)
dev.off()

svg(paste(OUTPUT, "rpart_MMSE.CHANGE.svg", sep="/"))
print(p_change_boxplot)
dev.off()


## ---------------------------------------------------------------------------------------------
## -- MRS.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "MRS.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod = rpart(MRS.CHANGE~MRS+LLV+BPF, data=d)
rpart_mod = prune(rpart_mod, cp=.02)
loss_reg(d$MRS.CHANGE, predict(rpart_mod, d))
#mse        r2       cor 
#0.4864713 0.2641623 0.5139673

rpart_mod1 = rpart(MRS.CHANGE~MRS+LLV+BPF, data=d)
rpart_mod2 = rpart(MRS.CHANGE~LLV+BPF, data=d)

loss_reg(d$MRS.CHANGE, predict(rpart_mod1, d))
#mse        r2       cor 
#0.4864713 0.2641623 0.5139673 

loss_reg(d$MRS.CHANGE, predict(rpart_mod2, d))
#mse        r2       cor 
#0.5212024 0.2116280 0.4600304

plot(rpart_mod, uniform=TRUE, main=paste(BASELINE, "~LLV+BPF"))
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
#0.4440868 0.7583538 0.8708351

label = unlist(attr(attr(y_pred_rtree, "group"), "name"))

d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

p_true_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MRS.M36", "~", "MRS")) +
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted") +
  theme(legend.position="bottom", legend.direction="vertical")

p_pred_m36_m0 = ggplot(d, aes(x = MRS, y = MRS.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MRS.M36", "~", "MRS, R2=",round(loss_partmlm_mod["r2"][[1]],2))) +
  theme(legend.position="bottom", legend.direction="vertical")

ds = summarySE(data=d, "MRS.CHANGE", "GROUP")
p_change_boxplot = 
  ggplot(d, aes(x = GROUP, y = MRS.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MRS.CHANGE", "~", "GROUP")) +
  theme(legend.position="bottom", legend.direction="vertical")

# Simplfy rule
# ------------
GROUP=d$GROUP
levels(GROUP)[levels(GROUP) == "/MRS< 1.5/BPF>=0.7858/BPF>=0.8572"] = "/MRS< 1.5/BPF>=0.8572"

groups.l = list()
groups.l[["decrease"]] = list(labels = "/MRS>=1.5/LLV< 1251", val=0)
#=> -0.3
groups.l[["stable"]] = list(labels = "/MRS< 1.5/BPF>=0.8572", val=1)
# => 0
#groups.l[["increase"]] = list(labels = 
#                                c("/MRS>=1.5/LLV>=1251", "/MRS< 1.5/BPF>=0.7858/BPF< 0.8572/LLV< 626.7",
#                                  "/MRS< 1.5/BPF>=0.7858/BPF< 0.8572/LLV>=626.7", "/MRS< 1.5/BPF< 0.7858"), val=2)
#=>  +0.5 & +1

groups = rep(NA, length(GROUP))
for(n in names(groups.l)){
  mask = rep(FALSE, length(GROUP))
  for(l in groups.l[[n]]$labels)  mask = mask | (GROUP == l)
  groups.l[[n]]$mask = mask
  name = paste(groups.l[[n]]$labels, collapse=" OR ")
  groups.l[[n]]$name = paste(name, ", N=",sum(mask),sep="")
  groups[mask] = groups.l[[n]]$val
}
groups[is.na(groups)] = 2

#groups.l[["increase"]] = list(labels = "others", val=2)

groups = factor(groups, labels=c(sapply(groups.l, function(x)x$name), paste("others, N=",sum(groups==2),sep=""))) 

groups
d$GROUP2 = groups

ds = summarySE(data=d, "MRS.CHANGE", "GROUP2")
p_change_boxplot_simple = 
  ggplot(d, aes(x = GROUP2, y = MRS.CHANGE, fill=GROUP2))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MRS.CHANGE", "~", "GROUP simplified")) + theme(legend.position="bottom", legend.direction="vertical")

write.csv(ds, paste(OUTPUT, "rpart_MRS.CHANGE.csv", sep="/"))

# plot
# ------------
pdf(paste(OUTPUT, "rpart_MRS.CHANGE.pdf", sep="/"))
plot(rpart_mod1, uniform=TRUE, main=paste("MRS.CHANGE~MRS+LLV+BPF"))
text(rpart_mod1, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod2, uniform=TRUE, main=paste("MRS.CHANGE~LLV+BPF"))
text(rpart_mod2, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod, uniform=TRUE, main=paste("MRS.CHANGE~MRS+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_change_boxplot)
print(p_change_boxplot_simple)
dev.off()


svg(paste(OUTPUT, "rpart_MRS.CHANGE.svg", sep="/"))
print(p_change_boxplot_simple)
dev.off()



## ---------------------------------------------------------------------------------------------
## -- TMTB_TIME.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "TMTB_TIME.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod1 = rpart(TMTB_TIME.CHANGE~TMTB_TIME+LLV+BPF, data=d)

rpart_mod2 = rpart(TMTB_TIME.CHANGE~LLV+BPF, data=d)

loss_reg(d$TMTB_TIME.CHANGE, predict(rpart_mod1, d))
#mse           r2          cor 
#1902.5583810    0.4125577    0.6423065 
loss_reg(d$TMTB_TIME.CHANGE, predict(rpart_mod2, d))
#mse           r2          cor 
#2788.4943661    0.1390121    0.3728433 

rpart_mod = rpart_mod1

plot(rpart_mod, uniform=TRUE, main=paste(BASELINE, ".CHANGE~LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)

# 1) root 175 566775.100   3.313396  
# 2) TMTB_TIME>=173 19  81703.790 -58.105260 *
#   3) TMTB_TIME< 173 156 404669.200  10.793870  
# 6) LLV< 394.8525 126 140877.100   1.824161  
# 12) LLV>=241.7795 10   4810.400 -23.400000 *
#   13) LLV< 241.7795 116 129155.600   3.998658  
# 26) LLV< 185.043 109 106992.600   1.851782  
# 52) BPF>=0.8420647 63  28937.250  -4.915215 *
#   53) BPF< 0.8420647 46  71219.390  11.119630  
# 106) LLV>=142.475 7   7567.492 -19.809570 *
#   107) LLV< 142.475 39  55753.690  16.671020  
# 214) TMTB_TIME< 101.5 29  13585.610   7.592336 *
#   215) TMTB_TIME>=101.5 10  32846.090  42.999210 *
#   27) LLV>=185.043 7  13837.710  37.428570 *
#   7) LLV>=394.8525 30 211077.500  48.466670  
# 14) TMTB_TIME< 126.5 23  66236.870  25.304350  
# 28) BPF< 0.8480292 16  22425.940  12.562500 *
#   29) BPF>=0.8480292 7  35275.710  54.428570 *
#   15) TMTB_TIME>=126.5 7  91957.710 124.571400 *

group = rpart.groups(rpart_mod, data=d)
mlm_mod = subgrouplm.learn(group, d, M36, BASELINE)
subgrouplm.predict(mlm_mod, group, d, limits=c(0, 30))
#subgrouplm.rpart_mod, group, data, limits=c(-Inf, +Inf)
y_pred_rtree = subgrouplm.predict(mlm_mod, group, data=d, limits=c(0, 30))
d$TMTB_TIME.M36_pred = y_pred_rtree 
loss_partmlm_mod = loss_reg(d$TMTB_TIME.M36, y_pred_rtree)
print(loss_partmlm_mod)
#mse           r2          cor 
#11990.285714    -1.031316           NA 

label = unlist(attr(attr(y_pred_rtree, "group"), "name"))

d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

p_true_m36_m0 = ggplot(d, aes(x = TMTB_TIME, y = TMTB_TIME.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("TMTB_TIME.M36", "~", "TMTB_TIME")) +
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted") +
  theme(legend.position="bottom", legend.direction="vertical")

p_change_llv = ggplot(d, aes(x = LLV, y = TMTB_TIME.CHANGE, colour=LLV)) + geom_point(alpha=1) + ggtitle(paste("TMTB_TIME.CHANGE", "~", "LLV"))

p_change_bpf = ggplot(d, aes(x = BPF, y = TMTB_TIME.CHANGE, colour=BPF)) + geom_point(alpha=1) + ggtitle(paste("TMTB_TIME.CHANGE", "~", "BPF"))

p_pred_m36_m0 = ggplot(d, aes(x = TMTB_TIME, y = TMTB_TIME.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("TMTB_TIME.M36", "~", "TMTB_TIME, R2=",round(loss_partmlm_mod["r2"][[1]],2))) +
  theme(legend.position="bottom", legend.direction="vertical")

ds = summarySE(data=d, "TMTB_TIME.CHANGE", "GROUP")
p_change_boxplot = 
  ggplot(d, aes(x = GROUP, y = TMTB_TIME.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("TMTB_TIME.CHANGE", "~", "GROUP")) +
  theme(legend.position="bottom", legend.direction="vertical")

# Simplfy rule
# ------------
groups = rep(NA, length(d$GROUP))

m0 = d$TMTB_TIME>=173
m1 = d$TMTB_TIME<173 & d$LLV<394.9
m2 = d$TMTB_TIME<173 & d$LLV>=394.9


groups[m0] = 0
groups[m1] = 1 
groups[m2] = 2

groups = factor(groups, labels=c("TMTB_TIME>=173", "TMTB_TIME<173/LLV<394.9", "TMTB_TIME<173/LLV>=394.9"))

d$GROUP2 = groups

ds = summarySE(data=d, "TMTB_TIME.CHANGE", "GROUP2")
p_change_boxplot_simple = 
  ggplot(d, aes(x = GROUP2, y = TMTB_TIME.CHANGE, fill=GROUP2))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("TMTB_TIME.CHANGE", "~", "GROUP simplified")) + theme(legend.position="bottom", legend.direction="vertical")

write.csv(ds, paste(OUTPUT, "rpart_TMTB_TIME.CHANGE.csv", sep="/"))

# plot
# ------------
pdf(paste(OUTPUT, "rpart_TMTB_TIME.CHANGE.pdf", sep="/"))
plot(rpart_mod1, uniform=TRUE, main=paste("TMTB_TIME.CHANGE~TMTB_TIME+LLV+BPF"))
text(rpart_mod1, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod2, uniform=TRUE, main=paste("TMTB_TIME.CHANGE~LLV+BPF (R2=0.14)"))
text(rpart_mod2, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod, uniform=TRUE, main=paste("TMTB_TIME.CHANGE~TMTB_TIME+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_change_llv)
print(p_change_bpf)
print(p_change_boxplot)
print(p_change_boxplot_simple)
dev.off()


svg(paste(OUTPUT, "rpart_TMTB_TIME.CHANGE.svg", sep="/"))
print(p_change_boxplot_simple)
dev.off()


## ---------------------------------------------------------------------------------------------
## -- MDRS_TOTAL.CHANGE
## ---------------------------------------------------------------------------------------------

M36 =  "MDRS_TOTAL.M36"
d = D[!is.na(D[, M36]),]
BASELINE = strsplit(M36, "[.]")[[1]][1]


rpart_mod1 = rpart(MDRS_TOTAL.CHANGE~MDRS_TOTAL+LLV+BPF, data=d)

rpart_mod2 = rpart(MDRS_TOTAL.CHANGE~LLV+BPF, data=d)

loss_reg(d$MDRS_TOTAL.CHANGE, predict(rpart_mod1, d))
#mse         r2        cor 
#32.5973123  0.4251535  0.6520379 
loss_reg(d$MDRS_TOTAL.CHANGE, predict(rpart_mod2, d))
#mse         r2        cor 
#35.1516447  0.3801084  0.6165293 
rpart_mod = rpart_mod1

plot(rpart_mod, uniform=TRUE, main=paste(BASELINE, ".CHANGE~LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)



group = rpart.groups(rpart_mod, data=d)
mlm_mod = subgrouplm.learn(group, d, M36, BASELINE)
subgrouplm.predict(mlm_mod, group, d, limits=c(0, 30))
#subgrouplm.rpart_mod, group, data, limits=c(-Inf, +Inf)
y_pred_rtree = subgrouplm.predict(mlm_mod, group, data=d, limits=c(0, 30))
d$MDRS_TOTAL.M36_pred = y_pred_rtree 
loss_partmlm_mod = loss_reg(d$MDRS_TOTAL.M36, y_pred_rtree)
print(loss_partmlm_mod)
#mse            r2           cor 
#11642.5972169   -49.7042757     0.6331371 

label = unlist(attr(attr(y_pred_rtree, "group"), "name"))

d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

p_true_m36_m0 = ggplot(d, aes(x = MDRS_TOTAL, y = MDRS_TOTAL.M36, colour=GROUP, group=GROUP)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  ggtitle(paste("MDRS_TOTAL.M36", "~", "MDRS_TOTAL")) +
  geom_smooth(method="lm", se=F) + geom_abline(linetype="dotted") +
  theme(legend.position="bottom", legend.direction="vertical")

p_change_llv = ggplot(d, aes(x = LLV, y = MDRS_TOTAL.CHANGE, colour=LLV)) + geom_point(alpha=1) + ggtitle(paste("MDRS_TOTAL.CHANGE", "~", "LLV"))

p_change_bpf = ggplot(d, aes(x = BPF, y = MDRS_TOTAL.CHANGE, colour=BPF)) + geom_point(alpha=1) + ggtitle(paste("MDRS_TOTAL.CHANGE", "~", "BPF"))

p_pred_m36_m0 = ggplot(d, aes(x = MDRS_TOTAL, y = MDRS_TOTAL.M36_pred)) + geom_point(alpha=1, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MDRS_TOTAL.M36", "~", "MDRS_TOTAL, R2=",round(loss_partmlm_mod["r2"][[1]],2))) +
  theme(legend.position="bottom", legend.direction="vertical")

ds = summarySE(data=d, "MDRS_TOTAL.CHANGE", "GROUP")
p_change_boxplot = 
  ggplot(d, aes(x = GROUP, y = MDRS_TOTAL.CHANGE, fill=GROUP))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MDRS_TOTAL.CHANGE", "~", "GROUP")) +
  theme(legend.position="bottom", legend.direction="vertical")

# Simplfy rule
# ------------
groups = rep(NA, length(d$GROUP))

m0 = d$LLV>=1632 | (d$LLV<1632 & d$BPF<0.749)
m1 = !m0


groups[m0] = 0
groups[m1] = 1 

groups = factor(groups, labels=c("LLV>=1632 OR (LLV<1632 AND BPF<0.749)", "others"))

d$GROUP2 = groups

ds = summarySE(data=d, "MDRS_TOTAL.CHANGE", "GROUP2")
p_change_boxplot_simple = 
  ggplot(d, aes(x = GROUP2, y = MDRS_TOTAL.CHANGE, fill=GROUP2))+#, colour=GROUP, group=GROUP)) +
  geom_boxplot(alpha=.5)+#alpha=1, aes(colour=GROUP)) +
  geom_point(data=ds, alpha=1, size=3, colour="black") +
  ggtitle(paste("MDRS_TOTAL.CHANGE", "~", "GROUP simplified")) + theme(legend.position="bottom", legend.direction="vertical")

write.csv(ds, paste(OUTPUT, "rpart_MDRS_TOTAL.CHANGE.csv", sep="/"))

# plot
# ------------
pdf(paste(OUTPUT, "rpart_MDRS_TOTAL.CHANGE.pdf", sep="/"))
plot(rpart_mod1, uniform=TRUE, main=paste("MDRS_TOTAL.CHANGE~MDRS_TOTAL+LLV+BPF"))
text(rpart_mod1, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod2, uniform=TRUE, main=paste("MDRS_TOTAL.CHANGE~LLV+BPF"))
text(rpart_mod2, use.n=TRUE, all=TRUE, cex=.8)
plot(rpart_mod, uniform=TRUE, main=paste("MDRS_TOTAL.CHANGE~MDRS_TOTAL+LLV+BPF"))
text(rpart_mod, use.n=TRUE, all=TRUE, cex=.8)
print(p_true_m36_m0)
print(p_pred_m36_m0)
print(p_change_llv)
print(p_change_bpf)
print(p_change_boxplot)
print(p_change_boxplot_simple)
dev.off()


svg(paste(OUTPUT, "rpart_MDRS_TOTAL.CHANGE.svg", sep="/"))
print(p_change_boxplot_simple)
dev.off()

