###
require(ggplot2)
require(glmnet)
require(reshape)
library(RColorBrewer)
library(plyr)

#display.brewer.pal(6, "Paired")
#palette = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]
palette = brewer.pal(6, "Paired")[c(1, 5, 3, 2, 6, 4)]

## M36 ##################################################################################################################################
d = read.csv("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/M36/RESULTS_TAB_CV.csv")
d = d[(((d$PREDICTORS == "BASELINE") & (d$MODEL =="GLM")) | d$MODEL =="ENET") & d$FOLD == "ALL", ]
#M36$PREDICTORS = gsub("", "", gsub("", "", gsub("", "", gsub("", "", M36$PREDICTORS))))
d$PREDICTORS = factor(d$PREDICTORS, levels=c("BASELINE", "BASELINE+NIGLOB", "BASELINE+NIGLOBFULL", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB", "BASELINE+CLINIC+NIGLOBFULL"))



d$TARGET = gsub("TMTB_TIME.M36", "TMTB", gsub("MRS.M36", "mRS", gsub("MDRS_TOTAL.M36", "MDRS", gsub("MMSE.M36", "MMSE", d$TARGET))))
d$TARGET = factor(d$TARGET, levels=c("MMSE", "MDRS", "TMTB", "mRS"))
d$r2_te_se = d$r2_te_se/5

m36 = d
#M36$PREDICTORS2 == M36$PREDICTORS
write.csv(m36, "/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/M36/RESULTS_TAB_CV_SUMMARY.csv")
#d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

m36p = ggplot(m36, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-r2_te_se, ymax=r2_te+r2_te_se), width=.1) +
  scale_y_continuous(expand=c(.1, 0))+
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("CV") + theme(legend.position="none")

#x11(); print(m36p)

svg("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/M36/RESULTS_TAB_CV_SUMMARY.svg")
print(m36p)
dev.off()

svg("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/M36/RESULTS_TAB_CV_SUMMARY_withlegend.svg")
print(m36p+theme(legend.position="bottom", legend.direction="vertical"))
dev.off()


## CHANGE ##################################################################################################################################
d = read.csv("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/CHANGE/RESULTS_TAB_CV.csv")
d = d[(((d$PREDICTORS == "BASELINE") & (d$MODEL =="GLM")) | d$MODEL =="ENET") & d$FOLD == "ALL", ]
#M36$PREDICTORS = gsub("", "", gsub("", "", gsub("", "", gsub("", "", M36$PREDICTORS))))
d$PREDICTORS = factor(d$PREDICTORS, levels=c("BASELINE", "BASELINE+NIGLOB", "BASELINE+NIGLOBFULL", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB", "BASELINE+CLINIC+NIGLOBFULL"))



d$TARGET = gsub("TMTB_TIME.CHANGE", "TMTB", gsub("MRS.CHANGE", "mRS", gsub("MDRS_TOTAL.CHANGE", "MDRS", gsub("MMSE.CHANGE", "MMSE", d$TARGET))))
d$TARGET = factor(d$TARGET, levels=c("MMSE", "MDRS", "TMTB", "mRS"))
d$r2_te_se = d$r2_te_se/5

change = d
#M36$PREDICTORS2 == M36$PREDICTORS
write.csv(change, "/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/CHANGE/RESULTS_TAB_CV_SUMMARY.csv")
#d$GROUP = factor(attr(y_pred_rtree, "group"), levels=as.integer(names(label)), labels=label)

changep = ggplot(change, aes(x = PREDICTORS, y = r2_te, fill=PREDICTORS)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-r2_te_se, ymax=r2_te+r2_te_se), width=.1) +
  scale_y_continuous(expand=c(.1, 0))+
  facet_wrap(~TARGET) + scale_fill_manual(values=palette) + ggtitle("CV") + theme(legend.position="none")
#  theme(legend.position="bottom", legend.direction="vertical")
x11(); print(changep)

svg("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/CHANGE/RESULTS_TAB_CV_SUMMARY.svg")
print(changep)
dev.off()

#svg("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/M36/RESULTS_TAB_CV_SUMMARY_withlegend.svg")
#print(m36p+theme(legend.position="bottom", legend.direction="vertical"))
#dev.off()

# d$PREDICTORS = 
#   gsub("BASELINE",   "Clinical score (at baseline)",
#        gsub("BASELINE+NIGLOB", "Clinical score + BPF & lacunes (at baseline)",
#             gsub("BASELINE+NIGLOBFULL", "Clinical score + BPF & lacunes + WMH & MB (at baseline)",
#                  gsub("BASELINE+CLINIC", "All clinical scores + cardio. risks + epidemio (at baseline)",
#                       gsub("BASELINE+CLINIC+NIGLOB",   "All clinical scores + cardio. risks + epidemio + BPF & lacunes (at baseline)",
#                            gsub("BASELINE+CLINIC+NIGLOBFULL",  "All clinical scores + cardio. risks + epidemio + BPF & lacunes + WMH & MB (at baseline)",
#                            as.character(d$PREDICTORS))))))