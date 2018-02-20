#! /bin/sh


SITE=CBIC

ls /neurospin/psy_sbox/hbn/$SITE/sourcedata/*/anat/*T1w.nii.gz|while read src; do echo $src ;
# src=/neurospin/psy_sbox/hbn/$SITE/sourcedata/sub-NDARZN677EYE/anat/sub-NDARZN677EYE_acq-VNav_T1w.nii.gz
dst_file=$(echo $src|sed -e "s%sourcedata%derivatives/vbm%g")
#echo cp --$src $dst_file
dst_dir=$(dirname "$dst_file")
mkdir -p "$dst_dir" && cp $src "$dst_dir"
gunzip "$dst_file"
echo $src "$dst_file"
done

