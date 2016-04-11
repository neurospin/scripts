#! /bin/bash


if [ $# -lt 1 ]; then
	old_cret="cretdicom"
else
	new_cret=$1
fi

if [ $# -lt 2 ]; then
	new_cret="creteil"
else
	new_cret=$1
fi


for dirpath in ${old_cret}/*/ ; do
	for file in ${dirpath}/*; do
	   if ! [ -d "$file" ]; then
		 mv ${file} ${new_cret}/
	   fi
	done
	
done
