# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:16:46 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
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

WS_code="""
# libraries
###########################################################################
library(WhiteStripe)
library(oro.nifti)
library(fslr)
library(pbapply)


# modify whitestripe function for debugging
###########################################################################
whitestripe <- function(
  img,
  type = c("T1", "T2", "last", "largest"),
  breaks = 2000, whitestripe.width = 0.05,
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

.write_brain <- function(brain.norm, output.file, brain.mask){
  if (is.character(brain.mask)){
    brain.mask <- readNIfTI(brain.mask, reorient=FALSE)
  }
  brain.mask[brain.mask==1] <- brain.norm + 1000.
  output.file <- gsub(".nii.gz|.nii", "", output.file)
  writeNIfTI(brain.mask, output.file)
}


# Assuming images are registered and normalized beforehand
# from J Fortin et al.
# https://github.com/Jfortin1/RAVEL/tree/master/R
###########################################################################
normalizeWS <- function(input.files, output.files=NULL, brain.mask=NULL, 
                        WhiteStripe_Type=c("T1", "T2", "FLAIR"), writeToDisk=FALSE, returnMatrix=FALSE, verbose=TRUE){
  
  WhiteStripe_Type <- match.arg(WhiteStripe_Type)
  if (WhiteStripe_Type=="FLAIR") WhiteStripe_Type <- "T2"
  # RAVEL correction procedure:
  if (!verbose) pboptions(type="none") 
  
  if (!is.null(brain.mask)){
    brain.mask <- readNIfTI(brain.mask, reorient=FALSE)
    brain.indices <- brain.mask==1
  } else {
    stop("brain.mask must be provided.")
  }
  
  if (is.null(output.files)){
    output.files <- gsub(".nii.gz|.nii","_WS.nii.gz", input.files)
  }
  
  cat("[normalizeWS] WhiteStripe intensity normalization is applied to each scan. \n")
  # Matrix of voxel intensities:
  V <- pblapply(input.files, function(x){
    brain <- readNIfTI(x, reorient=FALSE)
    # stripped brain
    stripped = TRUE
    stripped_brain = brain
    stripped_brain[brain.mask == 0] <- 0
    indices <- whitestripe(stripped_brain, type=WhiteStripe_Type, verbose=FALSE, stripped=stripped)
    brain   <- whitestripe_norm(brain, indices$whitestripe.ind)
    brain <- as.vector(brain[brain.indices])
    brain
  })
  V <- do.call(cbind, V)
  
  if (writeToDisk){
    if (verbose) cat("[normalizeWS] Writing out the corrected images \n")
    pblapply(1:ncol(V), function(i){
      .write_brain(brain.norm = V[,i], output.file = output.files[i], brain.mask=brain.mask)
    })
  } 
  if (returnMatrix){
    return(V)
  }	
}

###########################################################################
###########################################################################
###########################################################################
args <- commandArgs(trailingOnly = TRUE)


# Image to standardized T1
t1 = as.character(args[1])
# binary mask of the brain (GM, WM)
bmask = as.character(args[2])
#
res = normalizeWS(list(t1), brain.mask = bmask, WhiteStripe_Type = 'T1', writeToDisk = TRUE)

"""

# Script documentation
doc = """
Bias field correction
~~~~~~~~~~~~~~~~~~~~~

This code uses http://dx.doi.org/10.1016/j.nicl.2014.08.008
Shinohara RT, Sweeney EM, Goldsmith J, et al. Statistical normalization 
techniques for magnetic resonance imaging. 
NeuroImage : Clinical. 2014;6:9-19. doi:10.1016/j.nicl.2014.08.008.

It uses R and R packages (fslr and WhiteStripe) so far to pave the way to the
obtention of the radiomics features.

Steps:


Command:
python $HOME/gits/scripts/2017_rr/metastasis/WSstandardization \
    -i /neurospin/radiomics/studies/metastasis/base/187962757123/anat/187962757123_enh-gado_T1w.nii.gz \
    -d /tmp/existing_out_dir

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
parser.add_argument('--image', '-i', metavar='FILE',
                    help='Image to correct for bias field')
parser.add_argument('--outdir', '-d', metavar='PATH',
                    type=is_dir,
                    help='Output directory to create the file in.')
parser.add_argument('--param', '-p', metavar='FILE', nargs=1, type=str, default=None,
                    help='Parameter file containing the settings to be used in extraction')


"""

"""
def main():
    args = parser.parse_args()
#    args = parser.parse_args(['-d', '/tmp/existing_out_dir',
#                               '-i', "/neurospin/radiomics/studies/"
#                                     "metastasis/base/187962757123/anat/"
#                                     "187962757123_enh-gado_T1w.nii.gz"])
    

    # Initialize Logging if any


    # Initialize extractor
    try:
        if args.param is not None:
            raise Exception('Reading from param file not implemented yet.')
        # 
        ImageFilePath = args.image
        work_in = os.path.basename(ImageFilePath) 
        OutDirPath = args.outdir

        # get an tmp dir
        tmpdir = tempfile.mkdtemp()

        # cp input data for easy coding
        prevdir = os.getcwd()
        os.chdir(tmpdir)
        shutil.copyfile(ImageFilePath, work_in)

        # perform bet extraction and mask transcoding in float32
        mask_bet = 'betmask.nii.gz'
        mask_bin = mask_bet.replace('betmask', 'betmask_mask')
        cmd = ['fsl5.0-bet', work_in, mask_bet, '-m']
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        print ">>> ", os.getcwd(), " <<<", "Convert bet mask in float image: ",mask_bin
        nimg = ni.load(mask_bin)
        ni.save(ni.Nifti1Image(nimg.get_data().astype('float32'),
                                         affine=nimg.get_affine()),
                     mask_bin)

        # perform WS standardization with R scripts 
        with open('WS_code.R', 'w') as fp:
            fp.write(WS_code + '\n')
        cmd = ['Rscript', '--vanilla', 'WS_code.R', work_in, mask_bin]
        print ">>> ", os.getcwd(), " <<<", " ".join(cmd)
        results = subprocess.check_call(cmd)
        
        #move
        flist = glob('*_WS.nii.gz') + glob('*mask.nii.gz') + \
                ['Rplots.pdf', 'WS_code.R']
        for f in flist:
           shutil.move(f, OutDirPath)
        os.chdir(prevdir)
        shutil.rmtree(tmpdir)

    except Exception:
        print 'WS standardization FAILED:\n%s', traceback.format_exc()


if __name__ == "__main__":
    main()

