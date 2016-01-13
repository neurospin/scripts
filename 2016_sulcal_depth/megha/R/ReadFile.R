ReadFile <- function(PhenoFile, CovFile, GRMbinFile, GRMid, header=FALSE, delimiter="\t"){
# This function parses phenotypic data and covariates, and constructs genetic relationship matrix (GRM).
# Subjects in these files do not need to be exactly the same.
# This function will find the subjects in common in these files and sort the order of the subjects.
	
	PhenoL <- ParseFile(PhenoFile, delimiter, header)   # parse the phenotypic data
	
	if (nchar(CovFile)!=0){
		CovL <- ParseFile(CovFile, delimiter, header=FALSE)   # parse the covariate file
	}

	GRML <- ParseGRMbin(GRMbinFile, GRMid)   # parse the GRM file
	
	if (nchar(CovFile)==0){
		SubjID <- intersect(PhenoL$SubjID, GRML$SubjID)   # find subjects in common
	} else{
		SubjID <- intersect(intersect(PhenoL$SubjID, CovL$SubjID), GRML$SubjID)   # find subjects in common
	}	
	Nsubj <- length(SubjID)   # calculate the total number of common subjects
	
	# sort the order of the subjects in the phenotypic data
	IdxP <- match(SubjID, PhenoL$SubjID)
	Pheno <- PhenoL$Data[IdxP,]
	
	# sort the order of the subjects in the covariate file
	if (nchar(CovFile)==0){   # no covariates included in the model
		Cov <- matrix(1,Nsubj,1)   # intercept
		Ncov <- 1   # number of covariates (intercept only)
	} else{
		IdxC <- match(SubjID, CovL$SubjID)
		Cov <- cbind(matrix(1,Nsubj,1), CovL$Data[IdxC,])   # add intercept
		Ncov <- CovL$Ncol+1   # number of covariates (with intercept)
	}
	
	# sort the order of the subjects in the GRM
	IdxG <- match(SubjID, GRML$SubjID)
	K <- GRML$K[IdxG,IdxG]
	
	return(list(Pheno=Pheno, Cov=Cov, K=K, Nsubj=Nsubj, Npheno=PhenoL$Ncol, Ncov=Ncov, PhenoNames=PhenoL$PhenoNames, SubjID=SubjID))
}
