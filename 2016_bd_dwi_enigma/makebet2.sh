#!/bin/sh

#Not Shakespeare sequel

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

