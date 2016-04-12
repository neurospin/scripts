#!/bin/sh

#Not Shakespeare sequel

searchthresh() {
	
	dirname=$1
	gethresh=false
	if [ $# -lt 2 ]; then
		paramfile="param.txt"
	else	
	while IFS='' read -r line || [[ -n "$line" ]]; do

			if $gethresh; then
					thresh="$line"
					gethresh=false
					return thresh
			fi
			if [ "$line" = "$dirname" ]; then
					gethresh=true          
			fi
			#echo "Text read from file: $line"
	done < "$paramfile"
	echo $thresh
	echo "thresh or no thresh"
}


filesnii=$(ls -l |grep nii | wc -l)
i=0
echo "$filesnii to do"

#Fractional Intensity threshold for the Bet
if [ $# -eq 0 ]
  then
    thresh=0.25
else
	thresh="$1"
fi


for filename in *.nii* ; do
	newname=$(echo $filename | sed -e "s/.nii/_brain.nii/g")
	bet ${filename} ${newname} -F -f ${thresh}
	i=$((i+1))
	echo "$i files done, $((filesnii-i)) remaining"	
done

