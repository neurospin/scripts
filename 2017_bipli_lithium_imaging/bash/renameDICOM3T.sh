#!/bin/sh


if [ $# -lt 1 ]; then
	raw_dir="/neurospin/ciclops/projects/BIPLi7/ClinicalData/Raw_Data"
else
	nosort_dir=$1
fi



if $tosort; then
	echo "Running sort"

	#Sort for Bipli
	for subjpath in ${raw_dir}/*/; do
		for dirpath in ${subjpath}/DICOM3T/*/; do
			dirname=`basename $dirpath`
			newdirname=$(echo $dirname | sed -e "s/-/_/g")
			#echo ${dirpath}
			#echo ${subjpath}/DICOM3T/${newdirname}
			mv ${dirpath} ${subjpath}/DICOM3T/${newdirname}
		done
	done
fi
