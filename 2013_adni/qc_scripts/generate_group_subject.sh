#!/usr/bin/env bash
# Output can be put in subject_group.csv
echo PTID Group
for a in AD Control MCI;
do
	for b in $(ls /neurospin/cati/ADNI/ADNI_510/BVdatabase/$a/*/ -d);
	do
		c="$b/t1mri/default_acquisition/spm_new_segment/segmentation/";
		echo $(basename $b) $a
	done;
done
