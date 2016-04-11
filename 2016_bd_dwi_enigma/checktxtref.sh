#! /bin/bash

if [ $# -lt 1 ]; then
	folder="creteil"
else
	folder=$1
fi

i=0

while IFS='' read -r line || [[ -n "$line" ]]; do
	#echo $line
	#a=( $line )
	#echo $a
	
	for word in $line; do
		label="$(echo -e "${word}" | tr -d '[[:space:]]')"
	done

	labels["$i"]="$label"
	i=$(( $i + 1 ))

done < "${folder}list.txt"

j=0
l=0

for niipath in ${folder}/*.nii*; do

	found=0
	niiname=`basename $niipath`
	niiname="$(echo -e "${niiname}" | tr -d '[[:space:]]')"
	#echo $niiname
	
	for label in ${labels[*]}; do
		#echo $label
		if [ "$niiname" = "${label}.nii" ] || [ "$niiname" = "${label}.nii.gz" ]; then
			found=1
		fi
	done
	
	if [ "$found" = 1 ]; then
		j=$(( $j + 1 ))
	else
		l=$(( $l + 1 ))
		echo " $niiname"
	fi
	
done

echo "$l nifti files missing from excel/txt"
	
