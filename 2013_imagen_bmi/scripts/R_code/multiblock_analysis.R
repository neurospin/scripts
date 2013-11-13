#########################################
## Multiblock analysis Body Mass Index ##
#########################################

rm(list=ls()) ; gc()

# set working directory
setwd("/neurospin/brainomics/2013_imagen_bmi/data")
source("../scripts/functions.R")

# install newest version of RGCCA ("RGCCA_0.2.tar.gz")
# install.packages("scripts/RGCCA_2.0.tar.gz",repos=NULL,type="source")
#... and load package
require(RGCCA)

# load data
load("multiblock_dataset_residualized_smoothed_images_BUG.RData")
#load("multiblock_dataset_residualized_images.RData")

# A$IMGs <- matrix(rnorm(prod(dim(A$IMGs))),nrow(A$IMGs),ncol(A$IMGs))
# check dimensions : sapply(A,dim)

# To test RGCCA with random datasets:
# n <- 100
# img <- matrix(rnorm(n*40),n,40)
# snp <- matrix(rnorm(n*15),n,15)
# y <- matrix(rnorm(n*3),n,3)
# A <- list(SNPS=snp,IMGs=img,Y=y)
#
## To much memory required for the computation of the optimal tau
## system.time(res <- rgcca(A,tau="optimal",verbose=TRUE))

## Trying with one possible value for the regularization parameters
n <- nrow(A[[1]])
ind <- sample(n,0.64*n)
Asc <- lapply(A,scale2)
# system.time(res <- sgcca(lapply(Asc, function(mm)mm[ ind,]), c1 = c(0.6,0.6,1),verbose=TRUE,scale=F))
# system.time(res2 <- sgcca(lapply(Asc, function(mm)mm[ ind,]), c1 = c(0.1,0.1,1),verbose=TRUE,scale=F))
# cortest1 <- sgcca.predict(Asc, Asc, res)
# cortest2 <- sgcca.predict(Asc, Asc, res2)

A.train <- lapply(A, function(mm)mm[ ind,,drop=FALSE])
A.train.scaled <- lapply(A.train, scale2)
A.test  <- lapply(A, function(mm) mm[-ind,,drop=FALSE])

CC <- 1-diag(3)

model.train1 <- sgcca(A=A.train.scaled,C=CC,c1=c(0.1,0.1,1),scale=F)
model.train2 <- sgcca(A=A.train.scaled,C=CC,c1=c(0.9,0.9,1),scale=F)
## Building the classifier on the first components
## ... and Compute the test correlation
mscortest1 <- sgcca.predict(A.train.scaled, A.test, model.train1)
mscortest2 <- sgcca.predict(A.train.scaled, A.test, model.train2)

sapply(model.train1$a,function(u) sum(u==0))
sapply(model.train2$a,function(u) sum(u==0))






## Cross validation
cl <- tryCatch(clusters.parallel(), error = function(e) NULL)
set.seed(456)
trainmat <- GenerateLearningsets(n,method="CV",fold=5)@learnmatrix

params <- expand.grid(c11 = 1:4/5, c12 = 1:4/5, c13 = 1)

# 1 --> complete design ; 2 --> hierarchical design
C1 <- 1 - diag(3)
C2 <- cbind(c(0,0,1),c(0,0,1),c(1,1,0))
paropt1 <- paropt2 <- matrix(NA,nrow(trainmat),3)
model.train1 <- model.train2 <- NULL
cortest1 <- cortest2 <- rep(NA,nrow(trainmat))


# ### Cross validation loop
for (i in 1:nrow(trainmat)){
  cat("itération : ", i, "\n")
  ## Get train and test datasets for this iteration
  ind <- trainmat[i,]

  A.train <- lapply(A, function(mm)mm[ ind,,drop=FALSE])
  A.train.scaled <- lapply(A.train, scale2)
  A.test  <- lapply(A, function(mm) mm[-ind,,drop=FALSE])

  ## Estimating the optimal parameters on the training set
  paropt1[i,] <- sgcca.cca.cv(A.train,C=C1,scale=T,params=params,nfold=5, cl=cl)
  paropt2[i,] <- sgcca.cca.cv(A.train,C=C2,scale=T,params=params,nfold=5, cl=cl)

  model.train1[[i]] <- sgcca(A=A.train.scaled,C=C1,c1=paropt1[i,],scale=F)
  model.train2[[i]] <- sgcca(A=A.train.scaled,C=C2,c1=paropt2[i,],scale=F)
  ## Building the classifier on the first components
  ## ... and Compute the test correlation
  cortest1[i] <- sgcca.predict(A.train.scaled, A.test, model.train1[[i]])
  cortest2[i] <- sgcca.predict(A.train.scaled, A.test, model.train2[[i]])
}


### Finally a boxplot to represent
par(mar=c(9,4,4,2)+0.1)
boxplot(data.frame(cortest1,cortest2),
  names=c("Complete design","Hierarchical design"),las=2,
  main=paste("R squared of the linear model on the test data") ,ylim=0:1)

#Stop cluster
stopCluster(cl)

snp_sel1 <- sapply(model.train1,function(m) (m$astar$SNPs !=0)+0 )
snp_sel2 <- sapply(model.train2,function(m) (m$astar$SNPs !=0)+0 )
img_sel1 <- sapply(model.train1,function(m) (m$astar$IMGs !=0)+0 )
img_sel2 <- sapply(model.train2,function(m) (m$astar$IMGs !=0)+0 )

require(irr)
kappam.fleiss(snp_sel1)
kappam.fleiss(snp_sel2)
kappam.fleiss(img_sel1)
kappam.fleiss(img_sel2)

snp_signature1 <- colnames(A$SNPs)[ rowSums(snp_sel1) >= 4 ]
snp_signature2 <- colnames(A$SNPs)[ rowSums(snp_sel2) >= 4 ]
img_signature1 <- which( rowSums(img_sel1) >= 4 )
img_signature2 <- which( rowSums(img_sel2) >= 4 )

save.image("cross_validation_residualized_smoothed_images_2_designs.Rdata")

require(oro.nifti)
mask <- readNIfTI("/neurospin/brainomics/2013_imagen_bmi/data/rmask.nii")@.Data

tablo1 <- cbind(which(mask!=0,arr.ind=T)[img_signature1,],img_signature1)
tablo2 <- cbind(which(mask!=0,arr.ind=T)[img_signature2,],img_signature2)
write.table(tablo1,file="selected_features_complete_design_smoothed_images.csv",row.names=F,sep=",")
write.table(tablo2,file="selected_features_hierarchical_design_smoothed_images.csv",row.names=F,sep=",")

pdf("images_selected_features_complete_design_smoothed_images.pdf")
for (k in 1:dim(mask)[3]) {
#for (k in unique(tablo[,3])) {
  zz <- mask[,,k]
  zz[rbind(tablo1[tablo1[,3]==k,1:2])] <- 2
  image(zz,main=sprintf("k=%i",k))
}
dev.off()

pdf("images_selected_features_hierarchical_design_smoothed_images.pdf")
for (k in 1:dim(mask)[3]) {
#for (k in unique(tablo[,3])) {
  zz <- mask[,,k]
  zz[rbind(tablo2[tablo2[,3]==k,1:2])] <- 2
  image(zz,main=sprintf("k=%i",k))
}
dev.off()

#pdf("mask_moche.pdf")
#for (k in 1:dim(mask)[3]) {
#  image(mask[,,k],main=sprintf("k=%i",k))
#}
#dev.off()

library(biomaRt)
snpmart <- useMart("snp", dataset="hsapiens_snp")
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

snp_signature1 <- colnames(A$SNPs)[ rowSums(snp_sel1) >= 4 ]
snp_signature2 <- colnames(A$SNPs)[ rowSums(snp_sel2) >= 4 ]

a1 <- getBM( attributes = c("refsnp_id","ensembl_gene_stable_id","consequence_type_tv","chrom_start"), 
            filter = "snp_filter",snp_signature1,mart=snpmart)
a2 <- getBM( attributes = c("refsnp_id","ensembl_gene_stable_id","consequence_type_tv","chrom_start"), 
            filter = "snp_filter",snp_signature2,mart=snpmart)
b1 <- getBM(attributes = c( "ensembl_gene_id","hgnc_symbol","chromosome_name", "band"), filters = "ensembl_gene_id", 
            values =a1[,"ensembl_gene_stable_id"], mart = ensembl)
b2 <- getBM(attributes = c( "ensembl_gene_id","hgnc_symbol","chromosome_name", "band"), filters = "ensembl_gene_id", 
            values =a2[,"ensembl_gene_stable_id"], mart = ensembl)
ab1 <- merge(a1, b1, by.x="ensembl_gene_stable_id", by.y="ensembl_gene_id", all=TRUE)
ab2 <- merge(a2, b2, by.x="ensembl_gene_stable_id", by.y="ensembl_gene_id", all=TRUE)

write.table(ab1, file="annot_snp_complete_design_smoothed_images.csv",row=F)
write.table(ab2, file="annot_snp_hierarchical_design_smoothed_images.csv",row=F)



