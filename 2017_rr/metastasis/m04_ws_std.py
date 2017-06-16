#!/usr/bin/env python
import argparse
import sys
import os.path
import traceback
import subprocess
import nibabel as ni
import shutil
import tempfile
from glob import glob
from nilearn import plotting
from numpy import unique

WS_code="""
# libraries
###########################################################################
library(WhiteStripe)
library(oro.nifti)
library(fslr)
library(pbapply)

# customized whitestripe function
# use essentially img and 
#################################
whitestripe <- function(
  img,
  type = c("T1", "T2", "last", "largest"),
  breaks = 700, whitestripe.width = 0.05,
  whitestripe.width.l = whitestripe.width,
  whitestripe.width.u = whitestripe.width,
  arr.ind = FALSE, verbose = TRUE,
  stripped = FALSE, slices = NULL, ...)  {
  if (verbose) {
    message(paste0("Making Image VOI\n"))
  }
  if (stripped) {
    img.voi <- img[img > 0]
  } else {
    is_voi = inherits(img, "img_voi")
    if (is.null(slices) & !is_voi) {
      message(
        paste0("Using all slices of the image as slices not defined!", 
               " Use stripped = TRUE if using skull-stripped images.")
      )        
      d3 = dim(as(img, "array"))[3]
      slices = seq(d3)        
    }
    img.voi <- make_img_voi(img, slices = slices, ...)
  }
  if (verbose) {
    message(paste0(("Version locale")))
    message(paste0("Making ", type, " Histogram\n"))
  }
  img.hist = hist(img.voi, breaks = breaks, plot = TRUE)
  y.in = img.hist$counts
  x.in = img.hist$mids
  x.in = x.in[!is.na(y.in)]
  y.in = y.in[!is.na(y.in)]
  type = match.arg(type)
  stopifnot(length(type) == 1)
  if (verbose) {
    cat(paste0("Getting ", type, " Modes\n"))
  }
  if (type %in% c("T1", "last")) {
    img.mode = get.last.mode(x.in, y.in, verbose = verbose,
                             ...)
  }
  if (type %in% c("T2", "largest")) {
    img.mode = get.largest.mode(x.in, y.in, verbose = verbose,
                                ...)
  }
  img.mode.q = mean(img.voi < img.mode)
  if (verbose) {
    cat(paste0("Quantile ", type, " VOI\n"))
  }
  whitestripe = quantile(img.voi, probs = c(max(img.mode.q -
                                                  whitestripe.width.l, 0), min(img.mode.q + whitestripe.width.u,
                                                                               1)), na.rm = TRUE)
  whitestripe.ind = which((img > whitestripe[1]) & (img < whitestripe[2]),
                          arr.ind = arr.ind)
  err = FALSE
  if (length(whitestripe.ind) == 0) {
    warning(paste0("Length of White Stripe is 0 for ", type,
                   ", using whole brain normalization"))
    whitestripe.ind = which(img > mean(img))
    err = TRUE
  }
  mu.whitestripe = img.mode
  sig.whitestripe = sd(img[whitestripe.ind])
  mask.img = img
  mask.img[!is.na(mask.img) | is.na(mask.img)] = 0
  mask.img[whitestripe.ind] = 1
  if (inherits(img, "nifti")) {
    mask.img = cal_img(mask.img)
    mask.img = zero_trans(mask.img)
  }
  
  
  return(list(whitestripe.ind = whitestripe.ind, img.mode = img.mode,
              mask.img = mask.img, mu.whitestripe = mu.whitestripe,
              sig.whitestripe = sig.whitestripe, img.mode.q = img.mode.q,
              whitestripe = whitestripe, whitestripe.width = whitestripe.width,
              whitestripe.width.l = whitestripe.width.l, whitestripe.width.u = whitestripe.width.u,
              err = err))
}

# Performs hybrid T1/T2 version of the whiteStripe normalization
# tweaked from WhiteStripe package to return all indices : from T1, T2 images and intersection

whitestripe_hybrid <- function (t1, t2, ...){

  t1.ws = whitestripe(t1, type = "T1", breaks=2000, ...)
  t2.ws = whitestripe(t2, type = "T2", breaks=300, ...)
  whitestripe.ind = intersect(t1.ws$whitestripe.ind, t2.ws$whitestripe.ind)
  mask.img = t1
  mask.img[!is.na(mask.img) | is.na(mask.img)] = 0
  mask.img[whitestripe.ind] = 1
  if (inherits(t1, "nifti")) {
    mask.img = cal_img(mask.img)
    mask.img = zero_trans(mask.img)
  }
  return(list(whitestripe.ind = whitestripe.ind, 
              mask.img = mask.img, 
              WS.T1.ind = t1.ws$whitestripe.ind, 
              WS.T2.ind = t2.ws$whitestripe.ind,
              ret_t1 = t1.ws, 
              ret_t2 = t2.ws))
}

# Create intersection mask:
maskIntersect <- function(list, output.file = NULL, prob=1, 
                          reorient=FALSE, returnObject = TRUE, writeToDisk=TRUE, verbose=TRUE){
  
  if (!verbose) pboptions(type="none") 
  
  # Checks:
  if (is.atomic(list)){
    list <- as.list(list)
  }
  if (writeToDisk & is.null(output.file)){
    stop("output.file must be specified if writeToDisk is true.")
  }
  
  n <- length(list)
  inter  <- list[[1]]
  if (class(inter)=="nifti"){
    inter <- Reduce("+", list)
  } else if (class(inter)=="character"){
    inter <- readNIfTI(list[[1]], reorient=reorient)
    for (i in 2:n){
      inter <- inter + readNIfTI(list[[i]], reorient=reorient)
    }
  } else {
    stop("list must be either a list of nifti objects or a list of NIfTI file paths.")
  }
  
  # Creating the intersection map:
  cutoff <- floor(prob * n)
  inter[inter < cutoff]  <- 0 
  inter[inter >= cutoff] <- 1
  
  # Writing to disk:
  if (writeToDisk){
    filename <- gsub(".nii.gz|.nii", "", output.file)
    writeNIfTI(inter, filename)
  }
  
  # Returning object:
  if (returnObject){
    return(inter)
  }
}

.write_brain <- function(brain.norm, output.file, brain.mask, reorient=FALSE){
  if (is.character(brain.mask)){
    brain.mask <- readNIfTI(brain.mask, reorient=FALSE)
  }
  brain.mask[brain.mask==1] <- brain.norm
  #  brain.mask[brain.mask==1] <- brain.norm  + 1000.
  output.file <- gsub(".nii.gz|.nii", "", output.file)
  writeNIfTI(brain.mask, output.file)
}


# Assuming images are registered and normalized beforehand
# from J Fortin et al.
# https://github.com/Jfortin1/RAVEL/tree/master/R
###########################################################################
normalizeWS <- function(input.files=NULL, input.files2=NULL, output.files=NULL, brain.mask=NULL, 
                        WhiteStripe_Type=c("T1", "T2", "FLAIR", "HYBRID"), 
                        writeToDisk=FALSE, returnMatrix=FALSE, verbose=TRUE, 
                        reorient=FALSE){
  
  WhiteStripe_Type <- match.arg(WhiteStripe_Type)
  if (WhiteStripe_Type=="FLAIR") WhiteStripe_Type <- "T2"
  # RAVEL correction procedure:
  if (!verbose) pboptions(type="none") 
  
  if (!is.null(brain.mask)){
    brain.mask <- readNIfTI(brain.mask, reorient=reorient)
    brain.indices <- brain.mask==1
  } else {
    stop("brain.mask must be provided.")
  }
  
  
  
  cat("[normalizeWS] WhiteStripe intensity normalization is applied to each scan. \n")
  # Matrix of voxel intensities:
  if (WhiteStripe_Type == "T1"){
    
    if (is.null(output.files)){
      output.files <- gsub(".nii.gz|.nii","_WS.nii.gz", input.files)
    }
    V <- pblapply(input.files, function(x){
      brain <- readNIfTI(x, reorient=reorient)
      # stripped brain
      stripped = TRUE
      # # yes the brain is stripped but we want to reslice (pb of skull when crop doesnot work)
      # stripped = FALSE
      # slices = 80:120
      stripped_brain = brain
      stripped_brain[brain.mask == 0] <- 0
      indices <- whitestripe(stripped_brain, type=WhiteStripe_Type, breaks=2000, verbose=FALSE, stripped=stripped)
      # ajout
      #.write_brain(brain.norm=indices$mask.img, output.file = gsub(".nii.gz|.nii","_WSM.nii.gz",x), brain.mask=brain.mask, reorient=reorient)
      brain   <- whitestripe_norm(brain, indices$whitestripe.ind)
      #    brain <- as.vector(brain[brain.indices])
      list(brain=brain, indices=indices)
    })
    V <- do.call(cbind, V)
    if (writeToDisk){
      if (verbose) cat("[normalizeWS] Writing out the T1 corrected images \n")
      pblapply(1:ncol(V), function(i){
        writeNIfTI(V[,i]$brain, gsub(".nii.gz|.nii", "", output.files[i]))
        # .write_brain(brain.norm = V[,i]$brain, output.file = output.files[i],
        #              brain.mask=brain.mask, reorient=reorient)
        myMask <- V[, 1]$brain
        myMask[!is.na(myMask) | is.na(myMask)] <- NA
        myMask[V[, 1]$indices$whitestripe.ind] <- 1
        print(paste(length(V[, 1]$indices$whitestripe.ind), " voxels in T1 WS normalization."))
        pdf(paste0(gsub(".nii.gz|.nii", "_voxels.pdf", output.files[i])))
        orthographic(x=V[, 1]$brain, y=myMask, col.y="green")
        dev.off()
        writeNIfTI(myMask, gsub(".nii.gz|.nii", "_voxels", output.files[i]))
      })
    } 
    
  } else if (WhiteStripe_Type == "T2"){
    
    if (is.null(output.files)){
      output.files <- gsub(".nii.gz|.nii","_WS.nii.gz", input.files2)
    }
    V <- pblapply(input.files2, function(x){
      brain <- readNIfTI(x, reorient=reorient)
      # stripped brain
      stripped = TRUE
      # # yes the brain is stripped but we want to reslice (pb of skull when crop doesnot work)
      # stripped = FALSE
      # slices = 80:120
      stripped_brain = brain
      stripped_brain[brain.mask == 0] <- 0
      indices <- whitestripe(stripped_brain, type=WhiteStripe_Type, breaks=300, verbose=FALSE, stripped=stripped)
      # ajout
      #.write_brain(brain.norm=indices$mask.img, output.file = gsub(".nii.gz|.nii","_WSM.nii.gz",x), brain.mask=brain.mask, reorient=reorient)
      brain   <- whitestripe_norm(brain, indices$whitestripe.ind)
      #    brain <- as.vector(brain[brain.indices])
      list(brain=brain, indices=indices)
    })
    V <- do.call(cbind, V)
    if (writeToDisk){
      if (verbose) cat("[normalizeWS] Writing out the T2 corrected images \n")
      pblapply(1:ncol(V), function(i){
        writeNIfTI(V[,i]$brain, gsub(".nii.gz|.nii", "", output.files[i]))
        # .write_brain(brain.norm = V[,i]$brain, output.file = output.files[i],
        #              brain.mask=brain.mask, reorient=reorient)
        myMask <- V[, 1]$brain
        myMask[!is.na(myMask) | is.na(myMask)] <- NA
        myMask[V[, 1]$indices$whitestripe.ind] <- 1
        print(paste(length(V[, 1]$indices$whitestripe.ind), " voxels in T2 WS normalization."))
        pdf(paste0(gsub(".nii.gz|.nii", "_voxels.pdf", output.files[i])))
        orthographic(x=V[, 1]$brain, y=myMask, col.y="red")
        dev.off()
        writeNIfTI(myMask, gsub(".nii.gz|.nii", "_voxels", output.files[i]))
      })
    } 
  } else if (WhiteStripe_Type == "HYBRID"){
    output.files2 <- NULL
    if (is.null(output.files)){
      output.files <- gsub(".nii.gz|.nii","_WS_hybrid.nii.gz", input.files)
      output.files2 <- gsub(".nii.gz|.nii","_WS_hybrid.nii.gz", input.files2)
    }
    V <- pblapply(input.files, function(x, y){
      print(x)
      brain_t1 <- readNIfTI(x, reorient=reorient)
      print(unlist(y))
      brain_t2 <- readNIfTI(unlist(y), reorient=reorient)
      
      # stripped brain
      stripped = TRUE
      # # yes the brain is stripped but we want to reslice (pb of skull when crop doesnot work)
      # stripped = FALSE
      # slices = 80:120
      stripped_brain_t1 = brain_t1
      stripped_brain_t1[brain.mask == 0] <- 0
      stripped_brain_t2 = brain_t2
      stripped_brain_t2[brain.mask == 0] <- 0
      
      ### whitestripes hybrid
      indices <- whitestripe_hybrid(stripped_brain_t1, stripped_brain_t2, verbose=FALSE, stripped=stripped)
      
      # ajout
      #.write_brain(brain.norm=indices$mask.img, output.file = gsub(".nii.gz|.nii","_WSM.nii.gz",x), brain.mask=brain.mask, reorient=reorient)
      brain_t1 <- whitestripe_norm(brain_t1, indices$whitestripe.ind)
      brain_t2 <- whitestripe_norm(brain_t2, indices$whitestripe.ind)
      #    brain <- as.vector(brain[brain.indices])
      list(brain_t1=brain_t1, brain_t2=brain_t2, indices=indices)
    }, y = input.files2)
    V <- do.call(cbind, V)
    if (writeToDisk){
      if (verbose) cat("[normalizeWS] Writing out the T1/T2 corrected images \n")
      pblapply(1:ncol(V), function(i){
        writeNIfTI(V[,i]$brain_t1, gsub(".nii.gz|.nii", "", output.files[i]))
        writeNIfTI(V[,i]$brain_t2, gsub(".nii.gz|.nii", "", output.files2[i]))
        # .write_brain(brain.norm = V[,i]$brain, output.file = output.files[i],
        #              brain.mask=brain.mask, reorient=reorient)
        myMask1 <- V[, 1]$brain_t1
        myMask1[!is.na(myMask1) | is.na(myMask1)] <- NA
        myMask1[V[, 1]$indices$WS.T1.ind] <- 1
        myMask1[V[, 1]$indices$WS.T2.ind] <- 1000
        myMask1[V[, 1]$indices$whitestripe.ind] <- 500
        print(paste(length(V[, 1]$indices$WS.T1.ind), " voxels in T1 WS normalization."))
        print(paste(length(V[, 1]$indices$WS.T2.ind), " voxels in T2 WS normalization."))
        print(paste(length(V[, 1]$indices$whitestripe.ind), " voxels in intersection."))
        
        pdf(gsub(".nii.gz|.nii", "_voxels.pdf", output.files[i]))
        orthographic(x=V[, 1]$brain_t1, y=myMask1, col.y=rainbow(3))
        dev.off()
        
        writeNIfTI(myMask1, gsub(".nii.gz|.nii", "_voxels", output.files[i]))
      })
    } 
  }
  
  if (returnMatrix){
      zz = file(paste0("result_", type, ".csv"), "w")
      if (type == "HYBRID") {
          cat(paste0("Im\tMu\tSig"), 
              paste0("T1", "\t",V[,1]$indices$ret_t1$mu.whitestripe, "\t",V[,1]$indices$ret_t1$sig.whitestripe),
              paste0("T2", "\t",V[,1]$indices$ret_t2$mu.whitestripe, "\t",V[,1]$indices$ret_t2$sig.whitestripe),
              file=zz, sep='\n')

    } else {
        cat(paste0("Im\tMu\tSig"), paste0(type, "\t",V[,1]$indices$mu.whitestripe, "\t",V[,1]$indices$sig.whitestripe),
        file=zz, sep='\n')
    }
    close(zz)
    return(V)
  }	
}

###########################################################################
###########################################################################
###########################################################################
args <- commandArgs(trailingOnly = TRUE)
#args <- c(1:5)
#type = "HYBRID"
if (length(args) == 5){
  type = as.character(args[3])
  # # Image to standardized T1
  t1 = as.character(args[1])
  #t1 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model02/187962757123_enh-gado_T1w_bfc.nii.gz'
  # # Image to standardize FLAIR
  t2 = as.character(args[2])
  #t2 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model01/187962757123_rAxT2.nii.gz'
  # # Standardization type T1, FLAIR or HYBRID
  
  # # # binary mask of the brain (GM, WM)
  bmask = as.character(args[4])
  #bmask = '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/native_hatbox.nii.gz'
  pve2 = as.character(args[5])
  #pve2 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_betmask_pve_2.nii.gz'
  
} else if (length(args) == 4){
  type = as.character(args[2])
  if (type == "T1"){
    #t1 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model02/187962757123_enh-gado_T1w_bfc.nii.gz'
    t1 = as.character(args[1])
  } else if (type == "T2" | type == "FLAIR"){
    t2 = as.character(args[1])
    #t2 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model01/187962757123_rAxT2.nii.gz'
    
  }    
  # # # binary mask of the brain (GM, WM)
  bmask = as.character(args[3])
  #bmask = '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/native_hatbox.nii.gz'
  pve2 = as.character(args[4])
  #pve2 = '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_betmask_pve_2.nii.gz'
  #
}


reorient=FALSE
ws1 <- NULL
ws2 <- NULL

mask.wm = readNIfTI(pve2, reorient=reorient)
mask.gm = readNIfTI(gsub("_pve_2", "_pve_1", pve2), reorient=reorient)

if (type == "T1"){

    print("Performing T1 WhiteStripe normalization...")    

  res = normalizeWS(input.files=list(t1), brain.mask = bmask, WhiteStripe_Type = 'T1', 
                    writeToDisk = TRUE, reorient=reorient, returnMatrix=TRUE)
  
  
  # print densities    
  ws1 = readNIfTI(gsub(".nii.gz|.nii","_WS.nii.gz", t1), reorient=reorient)
  ws1.sel = ws1[mask.wm > .9]
  imghist = hist(ws1.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.wm <- density(ws1.sel)
  density.wm$y <- density.wm$y * multiplier[1]
  
  ws1.sel = ws1[mask.gm > .9]
  imghist = hist(ws1.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.gm <- density(ws1.sel)
  density.gm$y <- density.gm$y * multiplier[1]
  
  filename <- gsub(".nii.gz|.nii", ".pdf", t1)
  pdf(filename)
  plot(density.wm, type='l', col="red", xlim=c(min(density.gm$x), max(density.wm$x)),
       main=filename)
  lines(density.gm, type='l', col="green")
  legend("topleft", legend = c("White matter", "Grey matter"), fill = T, col = c("red", "green"))
  dev.off()
  
  filename <- gsub(".nii.gz|.nii", ".RData", t1)
  save(density.gm, density.wm, file=filename)
  
} else if (type == "FLAIR" | type == "T2"){

    print("Performing T2 WhiteStripe normalization...")    

  res = normalizeWS(input.files2=list(t2), brain.mask = bmask, WhiteStripe_Type = 'T2', 
                    writeToDisk = TRUE, reorient=reorient, returnMatrix=TRUE)
  
  # print densities
  
  ws2 = readNIfTI(gsub(".nii.gz|.nii","_WS.nii.gz", t2), reorient=reorient)
  ws2.sel = ws2[mask.wm > .9]
  imghist = hist(ws2.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.wm <- density(ws2.sel)
  density.wm$y <- density.wm$y * multiplier[1]
  
  ws2.sel = ws2[mask.gm > .9]
  imghist = hist(ws2.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.gm <- density(ws2.sel)
  density.gm$y <- density.gm$y * multiplier[1]
  
  filename <- gsub(".nii.gz|.nii", ".pdf", t2)
  print(filename)
  pdf(filename)
  plot(density.wm, type='l', col="red", xlim=c(min(density.gm$x), max(density.wm$x)),
       main=filename)
  lines(density.gm, type='l', col="green")
  legend("topleft", legend = c("White matter", "Grey matter"), lty=1, col = c("red", "green"))
  dev.off()
  
  filename <- gsub(".nii.gz|.nii", ".RData", t2)
  print(filename)
  save(density.gm, density.wm, file=filename)
} else if (type == "HYBRID"){
  
  print("Performing Hybrid WhiteStripe normalization...")    
  
  res = normalizeWS(input.files=list(t1), input.files2=list(t2), brain.mask = bmask, WhiteStripe_Type = 'HYBRID', 
                    writeToDisk = TRUE, reorient=reorient, returnMatrix=TRUE)
  
  # print densities
  print("Printing intensities...")    
  print("T1...")
  ws1 = readNIfTI(gsub(".nii.gz|.nii","_WS_hybrid.nii.gz", t1), reorient=reorient)
  ws1.sel = ws1[mask.wm > .9]
  imghist = hist(ws1.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.wm <- density(ws1.sel)
  density.wm$y <- density.wm$y * multiplier[1]
  
  ws1.sel = ws1[mask.gm > .9]
  imghist = hist(ws1.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.gm <- density(ws1.sel)
  density.gm$y <- density.gm$y * multiplier[1]
  
  filename <- gsub(".nii.gz|.nii", "_hybrid.pdf", t1)
  pdf(filename)
  plot(density.wm, type='l', col="red", xlim=c(min(density.gm$x), max(density.wm$x)),
       main=filename)
  lines(density.gm, type='l', col="green")
  legend("topleft", legend = c("White matter", "Grey matter"), lty=1, col = c("red", "green"))
  dev.off()
  
  filename <- gsub(".nii.gz|.nii", "_hybrid.RData", t1)
  save(density.gm, density.wm, file=filename)
  
  print("T2...")
  ws2 = readNIfTI(gsub(".nii.gz|.nii","_WS_hybrid.nii.gz", t2), reorient=reorient)
  ws2.sel = ws2[mask.wm > .9]
  imghist = hist(ws2.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.wm <- density(ws2.sel)
  density.wm$y <- density.wm$y * multiplier[1]
  
  ws2.sel = ws2[mask.gm > .9]
  imghist = hist(ws2.sel, breaks = 2000, plot = FALSE)
  multiplier <- imghist$counts / imghist$density
  density.gm <- density(ws2.sel)
  density.gm$y <- density.gm$y * multiplier[1]
  
  filename <- gsub(".nii.gz|.nii", "_hybrid.pdf", t2)
  pdf(filename)
  plot(density.wm, type='l', col="red", xlim=c(min(density.gm$x), max(density.wm$x)),
       main=filename)
  lines(density.gm, type='l', col="green")
  legend("topleft", legend = c("White matter", "Grey matter"), lty=1, col = c("red", "green"))
  dev.off()
  
  filename <- gsub(".nii.gz|.nii", "_hybrid.RData", t2)
  save(density.gm, density.wm, file=filename)
}


"""

# Script documentation
doc = """
Command:
python $HOME/gits/scripts/2017_rr/metastasis/m04_ws_std.py \
    -i1 /neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc.nii.gz \
    -i2 /neurospin/radiomics/studies/metastasis/base/187962757123/model01/187962757123_rAxT2.nii.gz \
    -t HYBRID
    -m /neurospin/radiomics/studies/metastasis/base/187962757123/model03/native_hatbox.nii.gz \
    -p /neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_betmask_pve_2.nii.gz \
    -d /neurospin/radiomics/studies/metastasis/base/187962757123/model05

"""


def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    if glob(os.path.join(dirarg, '*')) != []:
        raise argparse.ArgumentError(
            "The dir '{0}' is not empty!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--image1', metavar='FILE', required=False,
                    help='Image T1 to correct for bias field')
parser.add_argument('-i2', '--image2', metavar='FILE', required=False,
                    help='Image T2 or FLAIR to correct for bias field')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Normalizaion type : T1, T2, FLAIR or HYBRID')
parser.add_argument('-m', '--mask', metavar='FILE', required=True,
                    help='mask of slices obtained from MNI')
parser.add_argument('-p', '--pve', metavar='FILE', required=True,
                    help='pve_2 file obtained from fsl-fast')
parser.add_argument('-d', '--outdir', metavar='PATH', required=True,
                    type=is_dir,
                    help='Output directory to create the file in.')

def main():
    args = parser.parse_args()
    
    type_norm = args.type
    
    mask_bin = args.mask
    pve = args.pve
    OutDirPath = args.outdir

    # get an tmp dir
    tmpdir = tempfile.mkdtemp()
    prevdir = os.getcwd()
    os.chdir(tmpdir)

    try:
        if (type_norm == "T1"):
            work_in = os.path.basename(args.image1)
            shutil.copy(args.image1, work_in)
            with open('WS_code.R', 'w') as fp:
                fp.write(WS_code + '\n')
            cmd = ['Rscript', '--vanilla', 'WS_code.R', work_in, type_norm, mask_bin, pve]
            print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
            results = subprocess.check_call(cmd)
        elif (type_norm == "T2" or type_norm == "FLAIR"):
            work_in = os.path.basename(args.image2)
            shutil.copy(args.image2, work_in)
            with open('WS_code.R', 'w') as fp:
                fp.write(WS_code + '\n')
            cmd = ['Rscript', '--vanilla', 'WS_code.R', work_in, type_norm, mask_bin, pve]
            print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
            results = subprocess.check_call(cmd)
        elif (type_norm == "HYBRID"):
            work_in = os.path.basename(args.image1)
            work_in2 = os.path.basename(args.image2)
            shutil.copy(args.image1, work_in)
            shutil.copy(args.image2, work_in2)
            with open('WS_code.R', 'w') as fp:
                fp.write(WS_code + '\n')
            cmd = ['Rscript', '--vanilla', 'WS_code.R', work_in, work_in2, type_norm, mask_bin, pve]
            print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
            results = subprocess.check_call(cmd)
                
        #move
        flist = glob('*.nii.gz') + glob('*.pdf') + \
                glob('*RData') + glob('*.csv') + glob('*.R')
        print flist
        for f in flist:
           shutil.move(f, OutDirPath)
    
    except Exception:
        print 'WS standardization FAILED:\n%s', traceback.format_exc()

    # final housekeeping
    os.chdir(prevdir)
    shutil.rmtree(tmpdir)

main()
