options <- commandArgs(trailingOnly = TRUE)

input_file <- options[1]
output_file <- options[2]

d = read.table(input_file, header=TRUE)
# colnames(d)

d$propGM = d$nvoxGM / (d$nvoxGM + d$nvoxWM + d$nvoxCSF)
d$propWM = d$nvoxWM / (d$nvoxGM + d$nvoxWM + d$nvoxCSF)
d$propCSF = d$nvoxCSF / (d$nvoxGM + d$nvoxWM + d$nvoxCSF)

library(ggplot2) #type install.packages("ggplot2") if missing

#x11()
pdf(output_file, width=21)
d.flat = rbind(
data.frame(feat="propGM", center=d$center, val=d$propGM),
data.frame(feat="propWM", center=d$center, val=d$propWM),
data.frame(feat="propCSF", center=d$center, val=d$propCSF))

p <- ggplot(d.flat, aes(factor(center), val)) + geom_boxplot() + geom_jitter() 
p + facet_grid(. ~ feat) 

d$prop_vox_out_of_mask = ((d$nvoxGM_nomask+d$nvoxWM_nomask+d$nvoxCSF_nomask) - (d$nvoxGM + d$nvoxWM + d$nvoxCSF))/(d$nvoxGM_nomask+d$nvoxWM_nomask+d$nvoxCSF_nomask)
p <- ggplot(d, aes(factor(center), prop_vox_out_of_mask)) 
p + geom_boxplot() + geom_jitter() 
dev.off()

