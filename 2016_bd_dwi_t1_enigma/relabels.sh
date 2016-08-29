#! /bin/bash

declare -A labels


while IFS='' read -r line || [[ -n "$line" ]]; do
	#echo $line
	#a=( $line )
	#echo $a
	i=0
	for word in $line; do
		if [ "$i" -eq "0" ]; then
			#Make sure there is no whitespace
			label1="$(echo -e "${word}" | tr -d '[[:space:]]')"
			i=1
		else
			#Make sure there is no whitespace
			label2="$(echo -e "${word}" | tr -d '[[:space:]]')"
		fi
	done
	#oldlabel=${a[1]}
	#newlabel=${a[0]}
	#echo $label1
	#echo $label2
	labels["$label2"]="$label1"
	

done < "labels.txt"

if [ $# -lt 1 ]; then
	mann_dir="mannheim"
else
	mann_dir=$1
fi

echo ${mann_dir}

for oldpath in ${mann_dir}/*; do
	oldname=`basename $oldpath`
	oldid=${oldname%%.*}
	ext=${oldname#*.}
	newname=${labels[$oldid]}
	#echo "$oldname"
	if ! [ -z "$newname" ]; then
		#echo " $oldname $oldid $ext $newname"
		if [ -z "$ext" ] || [ "$ext" = "oldname" ]; then
			mv ${oldpath} ${mann_dir}/${newname}
		else 
			echo "$oldname $newname $ext"
			mv ${oldpath} ${mann_dir}/${newname} #.${ext}
		fi
	fi
done
