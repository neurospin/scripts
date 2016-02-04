#!/bin/sh

#Not actually about Shakespeare.

	
	file="$1"

	i=0

	fslsplit ${file}
	for betfile in vol* ; do
	echo "Processing Image N°$i"
	bet $betfile vil${i}.nii.gz -f 0.5 -g 0
	i=$((i+1))
	rm $betfile

	done

	OIFS= $IFS
	IFS='.'

	#Get the filename (whatever is before the first dot "." )
	IN=$filename
	set -- $IN
	fname=$1

	fslmerge -t ${fname_bet}.nii.gz vil*.nii.gz
	rm vil*.nii.gz

	IFS=$OIFS
