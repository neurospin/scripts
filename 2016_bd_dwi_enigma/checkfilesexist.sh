#! /bin/bash


folder="creteil"

while IFS='' read -r line || [[ -n "$line" ]]; do
	i=0

	for word in $line; do
		echo $word
		nii=0
		bval=0
		bvec=0
		for niipath in ${folder}/*.nii; do
			niiname=`basename $niipath`
			if [ "$niifile" = "word" ]; then
				nii=1
			fi
		done
		if [ "$nii" = 1 ]; then
			echo "present"
		else
			echo "absent"
		fi
	done
done < "${folder}.txt"
		
