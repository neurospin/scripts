# From bioconductor install lumiHumanAll.db 
#
library(lumiHumanAll.db)
library(lumiHumanIDMapping)

x <- lumiHumanAllSYMBOL
# Get the probe identifiers that are mapped to a gene symbol
mapped_probes <- mappedkeys(x)
# Convert to a list
xx <- as.list(x[mapped_probes])


measure_exp = read.table('/neurospin/brainomics/2013_brainomics_genomics/data/pheno/residuals.pheno')
probid = read.table('/neurospin/brainomics/2013_brainomics_genomics/data/pheno/residuals.gids.list')
tmp1 = lapply(as.vector(probid), IlluminaID2nuID)
tmp2 = lapply(tmp1[[1]][,"nuID"], function(i) xx[[i]])
# eliminate the null (record suppressed by DB nuclotide for example..)
to_suppress <- which(unlist(lapply(bbb, is.null)))
length(to_suppress)

measure_exp = measure_exp[,-(to_supress+2)]
colnames(measure_exp) <- c('FID','IID',tmp2[-(to_suppress+2)])
write.table(measure_exp, file = '/neurospin/brainomics/2013_brainomics_genomics/data/pheno/filter_residuals.pheno', sep='\t',quote=F,col.names=TRUE, row.names=FALSE )
