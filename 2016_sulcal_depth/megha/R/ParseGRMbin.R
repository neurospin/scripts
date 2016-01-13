ParseGRMbin <- function(GRMbinFile, GRMid){
# This function constructs the genetic relationship matrix (GRM) from GCTA output
# (a binary file containing the lower triangle elements of the GRM and a plain text file containing subject IDs).

	ID <- read.table(GRMid, header=FALSE, comment.char="")   # read the file containing subject IDs
	SubjID <- ID[[1]]   # extract subject IDs
	Nsubj <- length(SubjID)   # calculate the number of subjects
	
	BinFile <- file(GRMbinFile, "rb")   # read the GRM file
	GRM <- readBin(BinFile, what=numeric(), n=Nsubj*(Nsubj+1)/2, size=4)   # read the binary GRM file
	close(BinFile)   # close the GRM file
	
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
    return(list(Nsubj=Nsubj, SubjID=SubjID, K=K))
}