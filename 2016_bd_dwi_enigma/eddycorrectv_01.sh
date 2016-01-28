#!/bin/sh
  OIFS= $IFS
  IFS='.'
  ls -l |grep nii | wc -l
  filesnii=$(ls -l |grep nii | wc -l)
  i=0
  echo "$filesnii to do"
  
  if ! [ -d "No_EddyCorrect" ]; then
	mkdir No_EddyCorrect
  fi
  if ! [ -d "Logs" ]; then
	mkdir Logs
  fi
#  if ! [ -d "tmp0000" ]; then
#	mkdir tmp0000
#  fi
  
  if [ *.nii.gz != "*.nii.gz" ]; then
	for filename in *.nii.gz ; do
		IN=$filename
		set -- $IN
		fname=$1
		eddy_correct ${fname} ${fname}_eddy.nii trilinear
		mv ${fname}.nii.gz No_EddyCorrect/${fname}_noeddy.nii.gz
		mv ${fname}_eddy.ecclog Logs/${fname}_eddy.ecclog
		i=$((i+1))
		echo "$i files done, $((filesnii-i)) remaining"
	done 
  fi
  
 # if [ *.nii  !=  "*.nii" ]; then 
	for filename in *.nii ; do
		IN=$filename
		set -- $IN
		fname=$1
		eddy_correct ${fname} ${fname}_eddy.nii trilinear
		mv ${fname}.nii No_EddyCorrect/${fname}_noeddy.nii
		mv ${fname}.ecclog Logs/${fname}_eddy.ecclog
		i=$((i+1))
		echo "$i files done, $((filesnii-i)) remaining"
	done 
 # fi

  
  
  IFS=$OIFS
