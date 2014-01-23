require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"
#BASE_DIR = "/home/edouard/data/proj_predict_cog_decline"
INPUT = paste(BASE_DIR, "20140120_remove-predictors", sep="/")
INPUT_FILE = paste(OUTPUT, "results_summary_plot.csv", sep="/")


#setwd(BASE_DIR)
DB = read.csv(INPUT_FILE)
DB = DB[DB$TARGET !=  'BARTHEL.M36',]
DB$PREDICTORS = factor(DB$PREDICTORS, levels= c("BASELINE","BASELINE+NIGLOB", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB"))
library(RColorBrewer)
display.brewer.pal(6, "Paired")
pal = brewer.pal(6, "Paired")[c(1, 2, 5, 6)][c(1, 3, 2, 4)]

DBfr = DB[DB$SITE=="FR", ]
DBgr = DB[DB$SITE=="GE", ]

svg(sub(".csv", "_fr.svg", INPUT_FILE))
pfr = ggplot(DBfr, aes(x = PREDICTORS, y = r2, fill=PREDICTORS)) + geom_bar(stat = "identity", position="dodge") + facet_wrap(~TARGET) + scale_fill_manual(values=pal) 
print(pfr)
dev.off()

svg(sub(".csv", "_gr.svg", INPUT_FILE))
pfr = ggplot(DBgr, aes(x = PREDICTORS, y = r2, fill=PREDICTORS)) + geom_bar(stat = "identity", position="dodge") + facet_wrap(~TARGET) + scale_fill_manual(values=pal) 
print(pfr)
dev.off()
