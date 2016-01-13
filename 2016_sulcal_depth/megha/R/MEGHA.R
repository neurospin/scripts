MEGHA <- function(PhenoFile, CovFile, GRMbinFile, GRMid, header=FALSE, delimiter="\t", Nperm=0, WriteStat=FALSE, OutDir="NA"){
# This function implements massively expedited genome-wide heritability analysis (MEGHA).
#
# Input arguments -
# PhenoFile: a plain text file containing phenotypic data
# Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
# CovFile: a plain text file containing covariates (intercept NOT included)
# If no covariate needs to be included in the model, set CovFile=""
# Col 1: family ID; Col 2: subject ID; From Col 3: numerical values
# GRMbinFile: a binary file containing the lower trianglar elements of the GRM (output from GCTA)
# GRMid: a plain text file of subject IDs corresponding to GRMbinFile
# Col 1: family ID; Col 2: subject ID
#
# [Note: Subjects in these files do not need to be exactly the same.
# This function will find the subjects in common in these files and sort the order of the subjects.]
#
# header: TRUE - PhenoFile contains a headerline; FALSE - PhenoFile does not contain a headerline; default header=FALSE
# delimiter: delimiter used in the phenotypic file and covariates file; default delimiter="\t" (tab-delimited)
# Nperm: number of permutations; set Nperm=0 if permutation inference is not needed; default Nperm=0
# WriteStat: TRUE - write point estimates and significance measures of heritability to the output directory "OutDir"
# FALSE - do not write statistics; default WriteStat=FALSE; default output directory is the working directory
#
# Output arguments (as a list) -
# Pval: MEGHA p-values
# h2: MEGHA estimates of heritability magnitude
# SE: estimated standard error of heritability magnitude
# PermPval: MEGHA permutation p-values
# PermFWEcPval: MEGHA family-wise error corrected permutation p-values
# Nsubj: total number of subjects in common
# Npheno: number of phenotypes
# Ncov: number of covariates (including intercept)
#
# Output files (if WriteStat=TRUE) -
# MEGHAstat.txt: point estimates and significance measures of heritability for each phenotype written to the output directory "OutDir"

	cat("----- Parse Data File -----\n")
	flush.console()
	
	L <- ReadFile(PhenoFile, CovFile, GRMbinFile, GRMid, header, delimiter)   # read data file
	Pheno <- L$Pheno; Cov <- L$Cov; K <- L$K; Nsubj <- L$Nsubj; Npheno <- L$Npheno; Ncov <- L$Ncov; PhenoNames <- L$PhenoNames;   # extract data
	
	cat("----- ", as.character(Nsubj), " Subjects, ", as.character(Npheno), " Phenotypes, ", as.character(Ncov), " Covariates -----\n")
	flush.console()
	
	cat("----- Compute MEGHA p-values and Heritability Estimates -----\n")
	flush.console()
	
	P0 <- diag(Nsubj) - Cov%*%solve(crossprod(Cov),t(Cov))   # compute the projection matrix
	
	# Satterthwaith approximation
	delta <- sum(diag(P0%*%K))/2
	I11 <- sum(diag(P0%*%K%*%P0%*%K))/2; I12 <- sum(diag(P0%*%K))/2; I22 <- sum(diag(P0))/2;
	I <- I11-I12^2/I22   # efficient information
	kappa <- I/(2*delta); nv <- 2*delta^2/I;   # compute the scaling parameter and the degrees of freedom
	
	Err <- crossprod(P0, Pheno)   # compute the residuals
	Sigma0 <- t(colSums(Err^2))/(Nsubj-Ncov)   # estimate of residual variance
	
	Stat <- matrix(0,Npheno,1)   # allocate space
	for (i in 1:Npheno){
		Stat[i] <- crossprod(Err[,i], crossprod(K, Err[,i]))/(2*Sigma0[i])   # compute score test statistics
	}
	
	Pval <- 1-pchisq(Stat/kappa,nv)   # compute MEGHA p-values
	
	U <- svd(P0)$u[,1:(Nsubj-Ncov)]   # singular value decomposition
	TK <- crossprod(U, crossprod(K,U))   # transformed GRM
	TNsubj <- Nsubj-Ncov   # transformed number of subjects
	stdK <- sd(TK[upper.tri(TK)])   # standard deviation of the off-diagonal elements of the transformed GRM
	
	# compute estimates of heritability magnitude
	P <- 1-2*Pval; P[P<0] <- 0;
	WaldStat = qchisq(P, 1)   # Wald statistic
	SE <- sqrt(2)/TNsubj/stdK   # empirical estimate of the standard error of heritability magnitude
	h2 <- SE*sqrt(WaldStat); h2[h2>1] <- 1;   # heritability estimates
		
	if (Nperm==0){   # check if permutation inference is needed
		PermPval <- "NA"; PermFWEcPval <- "NA";
	} else{
		cat("----- MEGHA Permutation Inference -----\n")
		flush.console()
		
		TPheno <- crossprod(U, Pheno)   # transformed phenotypes
		TP0 <- diag(TNsubj); TNcov <- 0   # transformed projection matrix and number of covariates
		
		TErr <- TPheno   # compute the transformed residuals
		TSigma0 <- t(colSums(TErr^2))/(TNsubj-TNcov)   # estimate of the transformed residual variance
		
		PermStat <- matrix(0,Npheno,Nperm)   # allocate space
		
		Nreps <- 50; Nstep <- ceiling(Nperm/Nreps);
		cat("|--------------------------------------------------|\n ")   # make the background output
		flush.console()
		
		for (s in 1:Nperm){
			if (s%%Nstep==0 || s==Nperm){
				cat("*")   # monitor progress
				flush.console()
			}
			
			# permute the transformed GRM
			if (s==1){
				TKperm <- TK   # permutation sample should always include the observation
			} else{
				TPermSubj <- sample(TNsubj)   # shuffle subject IDs
				TKperm <- TK[TPermSubj,TPermSubj]   # permute transformed GRM
			}
			
			for (i in 1:Npheno){
				PermStat[i,s] <- diag(crossprod(TErr[,i], crossprod(TKperm, TErr[,i])))/(2*TSigma0[i])   # compute permuted score test statistics
			}
		}
		cat("\n")
		
		# compute MEGHA permutation p-values
		PermPval <- matrix(1,Npheno,1)   # allocate space
		for (i in 1:Npheno){
			PermPval[i] <- sum(PermStat[i,]>=PermStat[i,1])/Nperm
		}
		
		# compute MEGHA family-wise error corrected permutation p-values
		PermFWEcPval <- matrix(1,Npheno,1)   # allocate space
		MaxPermStat <- apply(PermStat,2,max)   # compute maximum statistics for each permutation
		for (i in 1:Npheno){
			PermFWEcPval[i] <- sum(MaxPermStat>=PermStat[i,1])/Nperm
		}
	}
	
	# write MEGHA statistics
	if (WriteStat){
		cat("----- Write MEGHA Statistics -----\n")
		if (OutDir=="NA"){
			OutFile <- "MEGHAstat.txt"   # write to the current working directory
		} else{
			OutFile <- paste(OutDir, "MEGHAstat.txt", sep="")   # set output directory
		}
		
		if (header){
			if (Nperm>0){
				OutFrame <- data.frame(cbind(h2,Pval,PermPval,PermFWEcPval), row.names=PhenoNames)   # construct output data frame
				colnames(OutFrame) <- c("h2", "Pval", "PermPval", "PermFWEcPval")   # set column names
				write.table(OutFrame, OutFile, sep="\t", col.names=NA)   # write statistics
			} else{   # no permutation statistics
				OutFrame <- data.frame(cbind(h2,Pval), row.names=PhenoNames)   # construct output data frame
				colnames(OutFrame) <- c("h2", "Pval")   # set column names
				write.table(OutFrame, OutFile, sep="\t", col.names=NA)   # write statistics
			}
		} else{   # no headerline
			if (Nperm>0){
				OutFrame <- data.frame(cbind(h2,Pval,PermPval,PermFWEcPval))   # construct output data frame
				colnames(OutFrame) <- c("h2", "Pval", "PermPval", "PermFWEcPval")   # set column names
				write.table(OutFrame, OutFile, sep="\t", row.names=FALSE)   # write statistics
			} else{   # no permutation statistics
				OutFrame <- data.frame(cbind(h2,Pval))   # construct output data frame
				colnames(OutFrame) <- c("h2", "Pval")   # set column names
				write.table(OutFrame, OutFile, sep="\t", row.names=FALSE)   # write statistics
			}
		}
	}
	
	return(list(Pval=Pval, h2=h2, SE=SE, PermPval=PermPval, PermFWEcPval=PermFWEcPval, Nsubj=Nsubj, Npheno=Npheno, Ncov=Ncov, PhenoNames=PhenoNames))

}