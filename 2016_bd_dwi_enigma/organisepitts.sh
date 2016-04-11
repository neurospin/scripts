#!/bin/sh

#This is the template of the file that is to go through all of our databases,
#Creteil, galway, grenoble, mannheim, pittsburgh, and udine, and organise them in the format:
# dwi/site/ID_name.nii or t1/site/ID_match.nii


#############
# Functions##
#############

### Function to search through strings.
#Important: DO NOT use this function for long strings, as it is slow and unefficient. 
#If we need it for long ones (100 or more characters) we'll need to make a new one.
searchstring () {
        mystring=$1
        substring=$2
        stringsize=${#mystring}
        substringsize=${#substring}
        max=$(($stringsize-$substringsize))
        searching=true
        #echo $max`seq 2 $max`
        for i in `seq 0 $max`; do
                #echo $mystring $i $substringsize
                subtest=${mystring:i:substringsize}
                if [ "$subtest" == "$substring" ]; then
                        #echo "found" 
                        return $i
                        #searching=false
                fi
        done
        #echo "not found"
        i=-1
        return $i
}

#Picks up the threshold value for each site specified in the param file 
#By default, param.txt
searchthresh() {
	
	dirname=$1
	gethresh=false
	if [ $# -lt 2 ]; then
		paramfile="param.txt"
	else
		paramfile=$2
	fi
	while IFS='' read -r line || [[ -n "$line" ]]; do

			if $gethresh; then
					thresh="$line"
					echo $thresh
					gethresh=false
					#return $thresh
			fi
			if [ "$line" = "$dirname" ]; then
					gethresh=true          
			fi
			#echo "Text read from file: $line"
	done < "$paramfile"

}

toruneddy() {
	
	dirname=$1
	geteddy=false
	if [ $# -lt 2 ]; then
		paramfile="parameddy.txt"
	else
		paramfile=$2
	fi
	while IFS='' read -r line || [[ -n "$line" ]]; do

			if $geteddy; then
					runeddy="$line"
					echo $runeddy
					geteddy=false
					#return $geteddy
			fi
			if [ "$line" = "$dirname" ]; then
					geteddy=true          
			fi
			#echo "Text read from file: $line"
	done < "$paramfile"

}

#### Function to check if dir exists, then make it
# Make after Checking the Directory
mkcdir() {
	if ! [ -d $1 ]; then
		mkdir $1
	fi	
}




#############
#  MAIN #####
#############


#####
#Parameters
#####
#Prints Additional details (not fully implemented)
verbose=true
#Converts found DICOM images to nifti (recommended)
convertIMAtonii=true
#Sorts the files from input path to output path, rest of pipeline assumes this has happened
tosort=true
#Eddy_correct for the nifti files (cannot run without doing sort)
toeddy=false
parameddy="parameddy.txt"
#Brain image extraction for the nifti files (cannot run without doing sort)
tobet=false
parambet="parambet.txt"
#Runs the dtifit which outputs the FA files (cannot run without doing Bet)
todtifit=false



#Check for input, if there are arguments, the first argument is the directory we grab the data from (root of the sites data)
#To improve: add error checks if args are not dir
#By default, the input folder is "sandbox"
if [ $# -lt 1 ]; then
	nosort_dir="sandbox"
else
	nosort_dir=$1
fi

#Same as above, check for second argument for result directory (after sort)
#By default, the output folder is "sorted". It is not necessary for that folder to exist yet.
if [ $# -lt 2 ]; then
	sorted_dir="sorted"
else
	sorted_dir=$2
fi

#If new directory doesn't exist, make it.
#Only works if all dirs are made except one. Ex: creates "dwi" not "myfold/dwi" if myfold doesn't exist

mkcdir $sorted_dir

mkcdir ${sorted_dir}/dwi

mkcdir ${sorted_dir}/t1

#This launches the sorting of all the input files into the output directory
if $tosort; then
	echo "Running sort"


	#Sort for pittsburgh

	if [ -d "${nosort_dir}/pittsburgh" ]; then
		
		pittsdir=${nosort_dir}/pittsburgh
		
		mkcdir ${sorted_dir}/dwi/pittsburgh

		mkcdir ${sorted_dir}/t1/pittsburgh 
		
		newpittsDWI=${sorted_dir}/dwi/pittsburgh
		newpittsT1=${sorted_dir}/t1/pittsburgh
		
		mkcdir ${newpittsT1}/IMAfiles
		mkcdir ${newpittsDWI}/IMAfiles

		
		for dirpath in ${pittsdir}/*/; do
			echo $dirpath
			newdirname=`basename $dirpath`
			echo $newdirname
			newdirname=${newdirname:9}
			echo $newdirname
			if $verbose; then
				echo "Copying $newdirname"
			fi
			
			cp -r ${dirpath}/dti* ${newpittsDWI}/${newdirname}
			cp -r ${dirpath}/axi* ${newpittsT1}/${newdirname}
			
			convertIMAtonii=true
			
			if $convertIMAtonii; then
			
				echo "Running dcm2nii in ${newpittsDWI}/${newdirname}"
			
				dcm2nii -4 ${newpittsDWI}/${newdirname}/*	
				dcm2nii -4 ${newpittsT1}/${newdirname}/*	
				mkcdir ${newpittsDWI}/IMAfiles/${newdirname}
				mkcdir ${newpittsT1}/IMAfiles/${newdirname}
				
				
				mv ${newpittsDWI}/${newdirname}/*nii* ${newpittsDWI}/${newdirname}.nii.gz
				mv ${newpittsDWI}/${newdirname}/*bval* ${newpittsDWI}/${newdirname}.bval
				mv ${newpittsDWI}/${newdirname}/*bvec* ${newpittsDWI}/${newdirname}.bvec
				
				mv ${newpittsT1}/${newdirname}/*nii* ${newpittsT1}/${newdirname}.nii.gz
				
				fslreorient2std ${newpittsDWI}/${newdirname}.nii.gz ${newpittsDWI}/${newdirname}.nii.gz
				fslreorient2std ${newpittsT1}/${newdirname}.nii.gz ${newpittsT1}/${newdirname}.nii.gz

				
				mv ${newpittsT1}/${newdirname} ${newpittsT1}/IMAfiles/	
				mv ${newpittsDWI}/${newdirname} ${newpittsDWI}/IMAfiles/
				
			fi	

		done
	fi


fi



#Eddy correct the files inside the sorted directories
if $toeddy; then
	echo "Running Eddy"
	for dirpath in ${sorted_dir}/dwi/*/; do
		
		
		i=0
		dirname=`basename $dirpath`
		echo $dirname
		toruneddy $dirname $parameddy
	
		if $runeddy; then
			filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
			echo "$filesnii to do in $dirname" 
			if ! [ -d "${dirpath}/No_EddyCorrect" ]; then
				mkdir ${dirpath}/No_EddyCorrect
			fi
			if ! [ -d "${dirpath}/Logs" ]; then
				mkdir ${dirpath}/Logs
			fi

			for filepath in ${dirpath}/*.nii* ; do
				
				filename=`basename $filepath`
				fileid=${filename%%.*}
				file_ext=${filename#*.}
				newpath=${dirpath}${fileid}
				echo ${filepath}
				eddy_correct ${filepath} ${newpath}__eddy.${file_ext} trilinear
				mv ${filepath} ${dirpath}/No_EddyCorrect/${filename}
				mv ${newpath}__eddy.ecclog ${dirpath}/Logs/${fileid}__eddy.ecclog
				i=$((i+1))
				echo "$i files done, $((filesnii-i)) remaining"
			done 
		fi
	done	  
fi



if $tobet; then
	echo "Running Bet"
	for dirpath in ${sorted_dir}/dwi/*/; do
		
		mkcdir ${dirpath}/NoBet
		mkcdir ${dirpath}/Brainmask
		
		i=0
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		echo $dirname
		searchthresh $dirname $parambet
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to bet in $dirname with a threshold of $thresh" 
		
		
		for filepath in ${dirpath}*.nii* ; do
			
			filename=`basename $filepath`
			newpath=$(echo $filepath | sed -e "s/.nii/__brain.nii/g")
			bet ${filepath} ${newpath} -F -f ${thresh}
			i=$((i+1))
			
			mv ${filepath} ${dirpath}/NoBet/${filename}
			mv ${dirpath}/${filename%%.*}__brain_mask* ${dirpath}/Brainmask
			echo "$i files done, $((filesnii-i)) remaining"	
		done
	done

	for dirpath in ${sorted_dir}/t1/*/; do
		
		mkcdir ${dirpath}/NoBet
		mkcdir ${dirpath}/Brainmask
		
		i=0
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		echo $dirname
		searchthresh $dirname $parambet
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to bet in $dirname with a threshold of $thresh" 
		
		
		for filepath in ${dirpath}*.nii* ; do
			
			filename=`basename $filepath`
			newpath=$(echo $filepath | sed -e "s/.nii/__brain.nii/g")
			bet ${filepath} ${newpath} -F -f ${thresh}
			i=$((i+1))
			
			mv ${filepath} ${dirpath}/NoBet/${filename}
			mv ${dirpath}/${filename%%.*}__brain_mask* ${dirpath}/Brainmask
			echo "$i files done, $((filesnii-i)) remaining"	
		done
	done

fi

#Needs for the images to be nifti, needs to be Betted (brain mask), needs the bvec and bval
if $todtifit; then
	echo "Running DTIFIT"
	for dirpath in ${sorted_dir}/dwi/*/; do
	
		i=0
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to dtfit in $dirname" 	

		for filepath in ${dirpath}*.nii* ; do
		
			filename=`basename $filepath`
			Idname=${filename%%__*}
			mkcdir ${dirpath}${Idname}
			echo "This is my entry file: $filepath"
			echo "This is my brain file: ${dirpath}Brainmask/${Idname}"
			echo "This is my bvec file: ${dirpath}${Idname}.bvec"
			echo "This is my bval file: ${dirpath}${Idname}.bval"
			echo "This is my output: ${dirpath}${Idname}/dtifit"
			dtifit -k ${filepath} -m ${dirpath}Brainmask/${Idname}__* -r ${dirpath}${Idname}.bvec -b ${dirpath}${Idname}.bval -o ${dirpath}${Idname}/dtifit
			i=$((i+1))
			echo "$i files done, $((filesnii-i)) remaining"	
		done
	done
fi
