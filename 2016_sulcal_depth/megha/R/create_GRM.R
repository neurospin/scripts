# R script to convert GRM from bin to txt
source("ParseGRMbin.R")
directory = "/neurospin/brainomics/imagen_central/kinship/"
radical = "pruned_m0.01_wsi50_wsk5_vif10.0"
GRMbin_path = paste(directory,radical, ".grm.bin", collapse = NULL, sep= "")
GRMid_path = paste(directory,radical, ".grm.id", collapse = NULL, sep= "")

ID <- read.table(GRMid_path, header=FALSE, comment.char="")   # read the file containing subject IDs
SubjID <- ID[[1]]   # extract subject IDs
Nsubj <- length(SubjID)   # calculate the number of subjects

BinFile <- file(GRMbin_path, "rb")   # read the GRM file
GRM <- readBin(BinFile, what=numeric(), n=Nsubj*(Nsubj+1)/2, size=4)   # read the binary GRM file
close(BinFile)   # close the GRM file
filename = paste(radical,".txt", collapse= NULL, sep="")
Indx <- 0
K <- matrix(0, Nsubj, Nsubj)   # allocate space
# construct GRM
for (i in 1:Nsubj){
	for (j in 1:i){
		Indx <- Indx+1
		K[i,j] <- GRM[Indx]
		K[j,i] <- GRM[Indx]   # GRM is symmetric
	}
}
list(Nsubj=Nsubj, SubjID=SubjID, K=K))
write.table(GRM, file=filename, row.names=FALSE, col.names=FALSE)
#lapply(mylist, write, "test.txt", append=TRUE, ncolumns=1000)