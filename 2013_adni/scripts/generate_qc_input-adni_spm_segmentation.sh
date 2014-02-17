#!/usr/bin/env bash
for a in AD Control MCI;
do
  for b in $(ls /neurospin/cati/ADNI/ADNI_510/BVdatabase/$a/*/ -d);
  do
    c="$b/t1mri/default_acquisition/spm_new_segment/segmentation/";
    echo $(basename $b | sed 's/_.*//g') $(basename $b) $(ls $c/*Nat_greyProba.nii) $(ls $c/*Nat_whiteProba.nii) $(ls $c/*Nat_csfProba.nii) $(ls $c/*Nat_skullProba.nii) $(ls $c/*Nat_scalpProba.nii);
  done;
done
