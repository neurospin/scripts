library(ggplot2)
library(RColorBrewer)
###########################################################################################################
## UTILS
retrieve_penalty_setting<-function(d, ratios){
  pena_setting = c()
  for(i in 1:nrow(d)){
    t = d[i, ]
    tmp = c(1-t["tv"], 1-t["tv"], t["tv"]) * ratios 
    case = NULL
    for(j in 1:nrow(tmp)){
      #cat(max(abs(tmp[j, ] - t[c("l1","l2","tv")])),"\n")
      if(max(abs(tmp[j, ] - t[c("l1","l2","tv")])) < 1e-8) case = paste(as.character(ratios[j, ]), collapse="_")
    }
    pena_setting = c(pena_setting, case)
  }
  d$pena_setting = pena_setting
  return(d)
}

###########################################################################################################

WD = "/home/ed203246/Dropbox/results/2013_adni"
WD = "/home/ed203246/tmp"
setwd(WD)
 
#CONDITION = "_simple"
CONDITION = "_cs"
#CONDITION = "_gtvenet"
#CONDITION = "_cs_gtvenet"

experiments_simple =list(
  "MCIc-MCInc"="MCIc-MCInc/MCIc-MCInc.csv",
  "MCIc-CTL"="MCIc-CTL/MCIc-CTL.csv",
  "AD-CTL"= "AD-CTL/AD-CTL.csv")

experiments_cs =list(
  "MCIc-MCInc_cs"="MCIc-MCInc_cs/MCIc-MCInc_cs.csv",
  "MCIc-CTL_cs"="MCIc-CTL_cs/MCIc-CTL_cs.csv",
  "AD-CTL_cs"= "AD-CTL_cs/AD-CTL_cs.csv")

experiments_gtvenet =list(
  "MCIc-MCInc_gtvenet"="MCIc-MCInc/MCIc-MCInc_gtvenet.csv",
  "MCIc-CTL_gtvenet"="MCIc-CTL/MCIc-CTL_gtvenet.csv",
  "AD-CTL_gtvenet"= "AD-CTL/AD-CTL_gtvenet.csv")

experiments_cs_gtvenet =list(
  "MCIc-MCInc_cs_gtvenet"="MCIc-MCInc/MCIc-MCInc_cs_gtvenet.csv",
  "MCIc-CTL_cs_gtvenet"="MCIc-CTL/MCIc-CTL_cs_gtvenet.csv",
  "AD-CTL__cs_gtvenet"= "AD-CTL/AD-CTL_cs_gtvenet.csv")

if(CONDITION == "_simple") experiments = experiments_simple
if(CONDITION == "_cs") experiments = experiments_cs
if(CONDITION == "_gtvenet") experiments = experiments_gtvenet
if(CONDITION == "_cs_gtvenet") experiments = experiments_cs_gtvenet

print(experiments)

ratios = read.csv("ratios_l1l2.csv")

dat = NULL
for(exp in names(experiments)){
  d = read.csv(experiments[[exp]])[, c("recall_mean","recall_mean_std", "recall_0","recall_1", "support_0", "support_1",
"beta_cor_mean", "auc", "a", "l1", "l2", "tv", "k")]
  d = d[d$k == -1, ]
  d = retrieve_penalty_setting(d, ratios)
  d2 = d
  d2$recall_mean = d$recall_mean - mean(d$recall_mean)
  cat("===", exp ,"===\n")
  print(summary(lm(recall_mean ~ l1 + a*l2 + a*tv - 1, data=d2)))
  d$exp = exp
  dat = rbind(dat, d)
}
#unique(dat$exp)


dat$a = as.factor(dat$a)
levels(dat$a)

dat$pena_setting = factor(dat$pena_setting,
levels=c("1_0_1", "0.9_0.1_1",  "0.5_0.5_1", "0.1_0.9_1", "0.01_0.99_1", "0.001_0.999_1", "0_1_1"))
#display.brewer.pal(9,"Purples")
#display.brewer.pal(9,"Blues")
#display.brewer.pal(9,"Yellows")
#display.brewer.pal(9,"Greens")
#display.brewer.pal(9,"Oranges")
#display.brewer.pal(9,"Reds")

pal=c(brewer.pal(9,"Blues")[c(8,6)],brewer.pal(9,"Greens")[5],brewer.pal(9,"Reds")[5:8])
pdf(paste("models_comparison_y-recall_x-tv", CONDITION, "_full.pdf", sep=""))
plt = qplot(tv, recall_mean, color=pena_setting, data=dat, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

pal=c(brewer.pal(9,"Blues")[c(8,6)],brewer.pal(9,"Greens")[5],brewer.pal(9,"Reds")[5:8])
pdf(paste("models_comparison_y-auc_x-tv", CONDITION, "_full.pdf", sep=""))
plt = qplot(tv, auc, color=pena_setting, data=dat, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

pal=c(brewer.pal(9,"Blues")[c(8,6)],brewer.pal(9,"Greens")[5],brewer.pal(9,"Reds")[5:8])
pdf(paste("models_comparison_y-beta_cor_mean_x-tv", CONDITION, "_full.pdf", sep=""))
plt = qplot(tv, beta_cor_mean, color=pena_setting, data=dat, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

## Select a subset of results
unique(dat$pena_setting)
dats = dat[(dat$a %in% c(0.01, 0.05, 0.1, 1.)) & 
           (dat$pena_setting %in% c("0.9_0.1_1","0.1_0.9_1","0.5_0.5_1", "0_1_1", "1_0_1")) &
           (!(dat$tv %in% c(0.001, 0.005, 0.01, 0.05))),]

sort(unique(dats$tv))

pal=c(brewer.pal(9,"Blues")[c(8,6)],brewer.pal(9,"Greens")[5],brewer.pal(9,"Reds")[c(6,8)])
#pal=c(brewer.pal(2,"Blues"),brewer.pal(1,"Greens"),brewer.pal(9,"YlOrRd")[c(3,9)])

pdf(paste("models_comparison_y-recall_x-tv", CONDITION, ".pdf", sep=""))
plt = qplot(tv, recall_mean, color=pena_setting, data=dats, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

pdf(paste("models_comparison_y-auc_x-tv", CONDITION, ".pdf", sep=""))
plt = qplot(tv, recall_mean, color=pena_setting, data=dats, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

pdf(paste("models_comparison_y-beta_cor_mean_x-tv", CONDITION, ".pdf", sep=""))
plt = qplot(tv, beta_cor_mean, color=pena_setting, data=dats, geom=c("line")) + facet_grid(a~exp, scale="free")
plt = plt + scale_colour_manual(values=pal)
print(plt)
dev.off()

#############################################################################################################
#
res = dat[dat$a == 0.1, ]
res = rbind(
data.frame(method= "l2", res[res$l1 == 0    & res$l2 == 1  & res$tv == 0, ]),
data.frame(method= "l2+tv", res[res$l1 == 0 & res$l2 == .5 & res$tv == .5, ]),
data.frame(method= "l1", res[res$l1 == 1    & res$l2 == 0  & res$tv == 0, ]),
data.frame(method= "l1+tv", res[res$l1 == .5 & res$l2 == 0 & res$tv == .5, ]),
data.frame(method= "tv", res[res$l1 == 0 & res$l2 == 0 & res$tv == 1, ]),
data.frame(method= "l1+l2", res[res$l1 == .5 & res$l2 == .5 & res$tv == 0, ]),
data.frame(method= "l1+l2+tv", res[res$l1 == .35 & res$l2 == .35 & res$tv == .3, ]))

recall_mean_pvals = c()
recall_0_pvals = c()
recall_1_pvals = c()

for(i in 1:nrow(res)){
  n = res[i, "support_0"] + res[i, "support_1"]
  x = round(res[i, "recall_mean"] * n)
  recall_mean_pvals = c(recall_mean_pvals, 
                        binom.test(x=x, n=n, p=.5, alternative = "greater")$p.value)
  n = res[i, "support_0"]
  x = round(res[i, "recall_0"] * n)
  recall_0_pvals = c(recall_0_pvals, 
                      binom.test(x=x, n=n, p=.5, alternative = "greater")$p.value)
  n = res[i, "support_1"]
  x = round(res[i, "recall_1"] * n)
  recall_1_pvals = c(recall_1_pvals, 
                     binom.test(x=x, n=n, p=.5, alternative = "greater")$p.value)
}
res$recall_mean_pvals = recall_mean_pvals
res$recall_0_pvals = recall_0_pvals
res$recall_1_pvals = recall_1_pvals

  
res = res[order(res$exp), ]
write.csv(res, "summary.csv", row.names=FALSE)
