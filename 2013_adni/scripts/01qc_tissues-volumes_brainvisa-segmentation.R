options <- commandArgs(trailingOnly = TRUE)

input_file <- options[1]
output_file <- options[2]

d = read.table(input_file, header=TRUE)
# colnames(d)

d$propGM = d$nvoxGM / (d$nvoxGM + d$nvoxWM)
d$propWM = d$nvoxWM / (d$nvoxGM + d$nvoxWM)

library(ggplot2) #type install.packages("ggplot2") if missing

#x11()
pdf(output_file, width=21)
d.flat = rbind(
data.frame(feat="propGM", center=d$center, val=d$propGM),
data.frame(feat="propWM", center=d$center, val=d$propWM))

p <- ggplot(d.flat, aes(factor(center), val)) + geom_boxplot() + geom_jitter() 
p + facet_grid(. ~ feat) 

d$ratioGM_WM = d$nvoxGM / d$nvoxWM
p <- ggplot(d, aes(factor(center), ratioGM_WM)) 
p + geom_boxplot() + geom_jitter() 
dev.off()