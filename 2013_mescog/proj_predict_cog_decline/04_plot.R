require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#BASE_DIR = "/home/edouard/data/proj_predict_cog_decline"

INPUT_FILE = "20140115_all-predictors/results_summary_plot.csv"
INPUT_FILE = "20140120_remove-predictors/results_summary_plot.csv"

setwd(BASE_DIR)
DB = read.csv(INPUT_FILE)
DB = DB[DB$target !=  'BARTHEL.M36',]
DB$predictors = factor(DB$predictors, levels= c("BASELINE","BASELINE+NIGLOB", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB"))
library(RColorBrewer)
display.brewer.pal(6, "Paired")
pal = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]

DBfr = DB[DB$data=="FR", ]
DBgr = DB[DB$data=="GR", ]

svg(sub(".csv", "_fr.svg", INPUT_FILE))
pfr = ggplot(DBfr, aes(x = predictors, y = r2_test, fill=predictors)) + geom_bar(stat = "identity", position="dodge") + facet_wrap(~target) + scale_fill_manual(values=pal) 
print(pfr)
dev.off()

svg(sub(".csv", "_gr.svg", INPUT_FILE))
pfr = ggplot(DBgr, aes(x = predictors, y = r2_test, fill=predictors)) + geom_bar(stat = "identity", position="dodge") + facet_wrap(~target) + scale_fill_manual(values=pal) 
print(pfr)
dev.off()
