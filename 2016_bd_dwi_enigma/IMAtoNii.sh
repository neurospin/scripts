#!/bin/sh

  ls -l |grep nii | wc -l
  filesnii=$(ls -l |grep IMA | wc -l)
  i=0
  echo "$filesnii to do"
  
  if ! [ -d "IMA" ]; then
	mkdir IMA
  fi
  
  for filename in *.IMA ; do
	echo $filename
	newname=$(echo $filename | sed -e "s/IMA/nii/g")
	echo $newname
	AimsFileConvert -i ${filename} -o ${newname}.nii
	mv ${filename} IMA
	i=$((i+1))
	echo "$i files done, $((filesnii-i)) remaining"
  done 
  

