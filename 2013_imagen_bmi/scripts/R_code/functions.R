require(snow)
require(MASS)



### Parallelized version of the code
clusters.parallel <- function(parr=TRUE, nodeNum=32) {
  #if (!parr) return(NULL)
  #cat("MPI master = ",Sys.getenv("HOSTNAME"),"\n")
  # Detect DSV cluster envrir
  #if (strsplit(Sys.getenv("HOSTNAME"),"-")[[1]][1]=='compute') {
  #   cl <- makeCluster(nodeNum, type = "MPI")
  #} else {
  Optionslocalhost <-list(host = "localhost")
  cl <- makeCluster(c(rep(list(Optionslocalhost), 5)), type = "SOCK")
  #}
  
  clusterEvalQ(cl, {library(RGCCA)})
  return(cl)
}


## Building the classifier on the first components
sgcca.predict <- function (train.scaled, test, model){
  # Test dataset: scaling here is not properly performed
  #   xtest1 <- scale(X1[-ind,]) %*% model.train$astar[[1]]
  #   xtest2 <- scale(X2[-ind,]) %*% model.train$astar[[2]]
  scl <- function(data, scaled, a){
    scale(data, center=attr(scaled, which="scaled:center"),
          scale=attr(scaled, which="scaled:scale"))  %*% a
  }
  xxtest <- mapply(scl, test, train.scaled, model$astar, SIMPLIFY=F)
  # run prediction
  # First trial: FAIL! 
  corchapo <- summary(lm(xxtest[[3]] ~ xxtest[[2]] + xxtest[[1]]))$r.squared
#   C <- model$C
#   mscor <- mean( ( cor(Reduce("cbind", xxtest)) * C * upper.tri(C) )**2 ) 
  return(1 - corchapo)   
}



ancillary.sgcca.cca.cv <- function(Xlist, C, ncomp, scale, method,
                                   trainmat_ind, penalty, scheme="horst")
{
   require(MASS)
   cat("Processing: ", head(trainmat_ind), "... for ", as.numeric(penalty), "\n")
   num_block <- length(Xlist)
   if (method=="LOOCV") {
     xtest <- lapply(Xlist, function(mm) t(as.matrix(mm[-trainmat_ind,,drop=FALSE])))
   } else {
#      print(xtest)
     xtest <- lapply(Xlist, function(mm) { as.matrix(mm[-trainmat_ind,,drop=FALSE])})
   }
   if (scale) {
       xxlist.train <- lapply(Xlist, function(mm) scale2(mm[trainmat_ind,,drop=FALSE]))
       scl_fun <- function(data, scaled) {
                 scale(data, center = attr(scaled, "scaled:center"),
                             scale  = attr(scaled, "scaled:scale")) }
       xxlist.test <- mapply(scl_fun, xtest, xxlist.train, SIMPLIFY=FALSE)
   } else {
       xxlist.train <- lapply(Xlist, function(mm) mm[trainmat_ind,,drop=FALSE])
       xxlist.test <- lapply(Xlist, function(mm) mm[-trainmat_ind,,drop=FALSE])
   }
   sgcca.res <- sgcca(A = xxlist.train, C = C, c1 = penalty, ncomp = ncomp,scheme=scheme)
   ## CCA solution : TODO
   #DataTrain <- Reduce("cbind", sgcca.res$Y[1:(num_block-1)])
   #colnames(DataTrain) <- paste("Y", 1:ncol(DataTrain), sep = "")
   #Ytrain <- xxlist.train[[3]]
   #rescca <- rgcca(list(DataTrain, Ytrain),tau="optimal")
   xxtest <- mapply(function(data, a){ data%*%a}, xxlist.test, sgcca.res$astar, SIMPLIFY=F)
   r2 <- summary(lm(xxtest[[3]] ~ xxtest[[2]] + xxtest[[1]]))$r.squared
#    mscor <- mean( ( cor(Reduce("cbind", xxtest)) * C * upper.tri(C) )**2 ) 
   return( list(err = 1 - r2 ) )
}

sgcca.cca.cv <- function (Xlist, C,  ncomp=rep(1, length(Xlist)),
                          nfold = nrow(Xlist[[1]]), scale = FALSE,
                          params=c(1,1,1),
                          scheme="horst",
                          cl=NULL)
{
   num_samples <- nrow(Xlist[[1]])

   if (nfold == num_samples) {
      trainmat <- GenerateLearningsets(num_samples, method = "LOOCV")@learnmatrix
      attr(trainmat, "method") <- "LOOCV"
   } else {
      trainmat <- GenerateLearningsets(num_samples, method = "CV", fold = nfold)@learnmatrix
      attr(trainmat, "method") <- "CV"
   }
   # flatten the loop on folds and the internloop on grided parameters
   # explicitely create the list of all configurations of CV and GParam
   tmp = expand.grid(fold=1:nfold, grid_param=1:nrow(params))
   flattened_folds <- trainmat[tmp[,'fold'],]
   flattened_folds <- split(flattened_folds,1:nrow(flattened_folds) )
   flattened_grid_params <- params[tmp[,'grid_param'], ]
   flattened_grid_params <- split(flattened_grid_params, 1:nrow(flattened_grid_params))
   if (is.null(cl)) {
      pb <- txtProgressBar(style = 3)
      obj = mapply(ancillary.sgcca.cca.cv,  # apply this function
                     trainmat_ind=flattened_folds,#for this fold and par
                     penalty=flattened_grid_params,
                     MoreArgs=list(Xlist, C, ncomp, scale, #on this
                                   method=attr(trainmat,"method"), scheme=scheme)
                     )
      rescv <- unlist(obj)
      close(pb)
   } else {
      tm = snow.time(
      obj <- clusterMap(cl,ancillary.sgcca.cca.cv,  # apply this function
                     trainmat_ind=flattened_folds,#for this fold and par
                     penalty=flattened_grid_params,
                     MoreArgs=list(Xlist, C, ncomp, scale, #on this
                                   method=attr(trainmat,"method"), scheme=scheme)
                     )
                     )
      print(tm)
      rescv <- unlist(obj) # clusterMap does not SIMPLIFY unlike mapply
   }
   # reshape : the lines are the folds and the col are the diff gridded params
   rescv = matrix(rescv, nrow=nfold)
   return(opt = unlist(params[which.min(colSums(rescv, na.rm=T)), ]))
}

require(methods)
setClass(Class="learningsets", 
         representation(learnmatrix="matrix", method="character", ntrain="numeric",
                        iter="numeric")
         )

roundvector <- function(x, maxint){
  fx <- floor(x)
  aftercomma <- x-fx
  roundorder <- order(aftercomma, decreasing=TRUE)
  i <- 1
  while(sum(fx) < maxint){ 
    fx[roundorder[i]] <- ceiling(x[roundorder[i]])
    i <- i+1
  }
  return(fx)
}

rowswaps <- function(blocklist){
  
  cols <- length(blocklist)
  fold <- nrow(blocklist[[1]])
  learnmatrix <- blocklist[[1]]
  for(i in 2:cols) learnmatrix <- cbind(learnmatrix, blocklist[[i]])
  rs <- rowSums( learnmatrix == 0)
  allowedzeros <- ceiling(sum(rs)/fold)
  indmatrix <-  matrix(rep(1:fold, each=cols), nrow=fold, byrow=TRUE) 
  while(any(rs > allowedzeros)){
    indmatrix <- replicate(cols, sample(1:fold))
    temp2list <- blocklist
    for(i in 1:cols) temp2list[[i]] <- blocklist[[i]][indmatrix[,i], ]
    learnmatrix <- temp2list[[1]]
    for(i in 2:cols) learnmatrix <- cbind(learnmatrix, temp2list[[i]])
    rs <- rowSums( learnmatrix == 0)
  }
  return(indmatrix)
}


#' Generate a learningsets manager a helper for the CV, LOOCV, etc processings
#' 
#' Create an object that interact with various CV schemes. It deals with
#' multiblocks objects.
#'
#' @param n numeric, number of folds chosen
#' @param y factoc vector to create balanced os stratified CV subsets
#' @param method character in c("LOOCV", "CV", "MCCV", "bootstrap")
#' @param fold Integer
#' @param niter Integer
#' @param ntrain Integer
#' @param strat boolean for stratification
#  @return a learningsets-class object
#' @references Martin Slawski, Anne-Laure Boulesteix and Christoph Bernau. (2009). CMA: Synthesis of microarray-based classification. R package version 1.14.0.
#' @export GenerateLearningsets
GenerateLearningsets <- function (n, y, method = c("LOOCV", "CV", "MCCV", "bootstrap"), 
          fold = NULL, niter = NULL, ntrain = NULL, strat = FALSE) 
{
  if (!missing(n)) {
    if (length(n) != 1 || n < 0) 
      stop("'n' must be a positive integer ! \n")
    n <- as.integer(n)
    if (!is.null(fold) && n <= fold) 
      stop("'n' is too small \n")
    if (!is.null(ntrain) && n <= ntrain) 
      stop("'n' is too small \n")
  }
  if (missing(n) & missing(y)) 
    stop("At least one of 'n' or 'y' mus be given \n")
  if (!missing(y)) 
    n <- length(y)
  method <- match.arg(method, c("LOOCV", "CV", "MCCV", "bootstrap"))
  if (!is.element(method, eval(formals(GenerateLearningsets)$method))) 
    stop("method must be one of 'LOOCV', 'CV', 'MCCV', 'bootstrap' \n")
  if (strat & missing(y)) 
    stop("If 'strat=TRUE', 'y' (class memberships) must be given \n")
  if (method == "MCCV") {
    if (is.null(niter) | is.null(ntrain)) 
      stop("With the MCCV method, arguments niter and ntrain should be given.")
    if (strat) {
      taby <- table(y)
      prop <- taby/sum(taby)
      classize <- roundvector(prop * ntrain, ntrain)
      if (any(classize < 1)) 
        stop("Generation of learningsets failed, one or several classes are too small. \n")
      indlist <- sapply(names(taby), function(z) which(y == 
        z), simplify = FALSE)
      learnmatrix <- matrix(nrow = niter, ncol = ntrain)
      lower <- cumsum(c(1, classize[-length(classize)]))
      upper <- cumsum(classize)
      for (i in 1:length(indlist)) learnmatrix[, lower[i]:upper[i]] <- t(replicate(niter, 
                                                                                   sample(indlist[[i]], classize[i], replace = FALSE)))
    }
    else learnmatrix <- t(replicate(niter, sample(n, ntrain, 
                                                  replace = FALSE)))
  }
  if (method == "CV") {
    if (is.null(niter)) 
      niter <- 1
    if (is.null(fold)) 
      stop("With the CV method, argument 'fold' must be given.")
    if (!strat) {
      if (fold == n) 
        method <- "LOOCV"
      else {
        size <- n/fold
        learnmatrix <- matrix(0, niter * fold, n - floor(size))
        size.int <- floor(size)
        size.vector <- rep(size.int, fold)
        if (size.int != size) 
          size.vector[1:((size - size.int) * fold)] <- size.vector[1:((size - 
            size.int) * fold)] + 1
        group.index <- c()
        for (j in 1:fold) group.index <- c(group.index, 
                                           rep(j, size.vector[j]))
        for (i in 1:niter) {
          group.index <- group.index[sample(n, n, replace = FALSE)]
          for (j in 1:fold) {
            whichj <- which(group.index == j)
            learnmatrix[j + (i - 1) * fold, 1:length(whichj)] <- whichj
          }
        }
        learnmatrix <- learnmatrix[, 1:max(size.vector), 
                                   drop = FALSE]
        if (size.int != size) 
          learnmatrix <- t(apply(learnmatrix, 1, function(z) setdiff(0:n, 
                                                                     z)))
        if (size.int == size) 
          learnmatrix <- t(apply(learnmatrix, 1, function(z) setdiff(1:n, 
                                                                     z)))
      }
    }
    else {
      taby <- table(y)
      prop <- taby/sum(taby)
      siz <- n - floor(n/fold)
      classize <- roundvector(prop * siz, siz)
      if (any(taby < fold)) 
        stop("Generation of learningsets failed, one or several classes are smaller than the number of folds. \n")
      indlist <- sapply(names(taby), function(z) which(y == 
        z), simplify = FALSE)
      templist <- vector(mode = "list", length = length(indlist))
      for (i in 1:length(indlist)) {
        outp <- do.call(GenerateLearningsets, args = list(n = taby[i], 
                                                          method = "CV", niter = niter, fold = fold))@learnmatrix
        templist[[i]] <- t(apply(outp, 1, function(z) ifelse(z == 
          0, 0, indlist[[i]][z])))
      }
      topass <- lapply(templist, function(z) z[1:fold, 
                                               , drop = FALSE])
      swaporder <- rowswaps(topass)
      nrep <- 1
      while (nrep < niter) {
        swaporder <- rbind(swaporder, swaporder[1:fold, 
                                                , drop = FALSE] + fold * nrep)
        nrep <- nrep + 1
      }
      for (i in 1:length(templist)) templist[[i]] <- templist[[i]][swaporder[, 
                                                                             i], ]
      learnmatrix <- templist[[1]]
      for (i in 2:length(indlist)) learnmatrix <- cbind(learnmatrix, 
                                                        templist[[i]])
    }
  }
  if (method == "LOOCV") 
    learnmatrix <- matrix(rep(1:n, each = n - 1), nrow = n)
  if (method == "bootstrap") {
    if (is.null(niter)) 
      stop("If 'method=bootstrap', the argument 'niter' must be given. \n")
    if (!strat) 
      learnmatrix <- t(replicate(niter, sample(n, replace = TRUE)))
    else {
      taby <- table(y)
      if (any(taby) < 1) 
        stop("Generation of learningsets failed, one or several classes are too small. \n")
      indlist <- sapply(names(taby), function(z) which(y == 
        z), simplify = FALSE)
      learnmatrix <- matrix(nrow = niter, ncol = n)
      lower <- cumsum(c(1, taby[-length(taby)]))
      upper <- cumsum(taby)
      for (i in 1:length(indlist)) {
        learnmatrix[, lower[i]:upper[i]] <- t(replicate(niter, 
                                                        sample(indlist[[i]], taby[i], replace = TRUE)))
      }
    }
  }
  if (strat & is.element(method, c("CV", "MCCV", "bootstrap"))) 
    method <- paste("stratified", method)
  new("learningsets", learnmatrix=learnmatrix, method=method,
      ntrain=ncol(learnmatrix), iter=nrow(learnmatrix))
  #return(list(learnmatrix = learnmatrix, method = method,
  #            ntrain = ncol(learnmatrix), iter = nrow(learnmatrix)))
}

  