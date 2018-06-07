export WD=/neurospin/psy/canbind/data

cd $WD
mkdir derivatives
mkdir derivatives/spmsegment

export DST=$WD/derivatives/spmsegment
export SRC=$WD/sourcedata/canbind_bids

# Copy
rsync -avu $SRC/* $DST/


# 1) Basic QC on raw data
ls $SRC/sub-*/ses-*/anat/*.nii|wc
# 808
ls $DST/sub-*/ses-*/anat/*.nii|wc
#    808     808   84072

cd $DST

fsl5.0-slicesdir $DST/sub-*/ses-*/anat/*.nii

mkdir QC
mv slicesdir QC/slicesdir_raw


firefox QC/slicesdir_raw/index.html

1.2 Optional Center origin of every image to the Anterior CIngular with SPM

