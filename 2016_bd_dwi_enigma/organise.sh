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
#Eddy_correct for the nifti files
toeddy=false
#Brain image extraction for the nifti files
tobet=false
paramfile="param.txt"



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


	#Sort for mannheim
	if [ -d "${nosort_dir}/mannheim" ]; then
	
		
		mkcdir ${sorted_dir}/dwi/mannheim
		mkcdir ${sorted_dir}/t1/mannheim

		mkcdir ${sorted_dir}/dwi/mannheim/IMAfiles
		
		mkcdir ${sorted_dir}/t1/mannheim/IMAfiles
		
		sortpathdwi=${sorted_dir}/dwi/mannheim
		sortpatht1=${sorted_dir}/t1/mannheim
		
		for dirpath in ${nosort_dir}/mannheim/*/ ; do
			
			dirname=`basename $dirpath`
			if $verbose; then
				echo "Copying $dirname"
			fi		
	
			mkcdir ${sortpathdwi}/${dirname}
			mkcdir ${sortpatht1}/${dirname}
			
			cp ${dirpath}/DTI/* ${sortpathdwi}/${dirname}
			cp ${dirpath}/MPR/* ${sortpatht1}/${dirname}

		done
		
		#### This could become a function if proven to be useful: converts IMA images to niftii by merging them together, and then sorts them out.
		#Though if that becomes a function, some serious restructuring would be in order...
		convertIMAtonii=true
		if $convertIMAtonii; then	

				
			for dirpath in ${sortpathdwi}/*/ ; do

				Id_name=`basename $dirpath`
				echo $Id_name
				
				if ! [ $Id_name = "IMAfiles" ]; then
					
					mkcdir ${sortpathdwi}/IMAfiles/${Id_name}
					dcm2nii -4 ${dirpath}*.IMA
					mv ${dirpath}*.IMA* ${sortpathdwi}/IMAfiles/${Id_name}
					for filepath in ${dirpath}/*; do
						#echo $filename
						filename=`basename $filepath`
						mv ${filepath} ${sortpathdwi}/${Id_name}.${filename#*.}
					done
					
					rm -r ${dirpath}
						
				fi
			done

			for dirpath in ${sortpatht1}/*/ ; do

				Id_name=`basename $dirpath`
				echo "this is the Idname: $Id_name"
				
				if ! [ $Id_name = "IMAfiles" ]; then
					mkcdir ${sortpatht1}/IMAfiles/${Id_name}
					dcm2nii -4 ${dirpath}*.IMA
					
					echo ${dirpath}
					mv ${dirpath}*.IMA* ${sortpatht1}/IMAfiles/${Id_name}
					for filepath in ${dirpath}/*; do
						#echo "this is the filepath: $filepath"
						filename=`basename $filepath`
						#echo "this is the destination: ${Id_name}.${filename#*.} "
						mv ${filepath} ${sortpatht1}/${Id_name}.${filename#*.}
					done
					rm -r ${dirpath}
				fi
			done
		
		fi			
	fi

	#Sort for Grenoble
	if [ -d "${nosort_dir}grenoble" ]; then
			
		if [ -d "${nosort_dir}grenoble/images_nifti/patients_controles" ]; then

			oldpath=${nosort_dir}grenoble/images_nifti/patients_controles
		
			mkcdir ${sorted_dir}/dwi/grenoble_Philipps

			mkcdir ${sorted_dir}/t1/grenoble_Philipps		
			
			#check for dwi files
			for filepath in ${oldpath}/*DTI*Ed*; do
				
				filename=`basename $filepath`
				fileid=${filename#*BipEd}
				fileid=${fileid:0:2}
				newname=BipEd${fileid}.${filename#*.}
				
				if $verbose; then
					echo "Copying $newname"
				fi

				cp ${filepath} ${sorted_dir}/dwi/grenoble_Philipps/${newname}
			done	
		
			#check for t1 files
			for filepath in ${oldpath}/*Anat*Ed*; do
				
				filename=`basename $filepath`
				fileid=${filename#*BipEd}
				fileid=${fileid:0:2}
				
				#In some of the T1 files, the ID is after the SECOND BipEd, not the first,
				#With the first one just indicating BipEdLLClear
				#Update on that=> Our two unidentified subjects still get LL: just
				#Delete them until we can actually USE them.
				if [ "$fileid" = LL ]; then
					fileid=${filename##*BipEd}
					fileid=${fileid:0:2}
				fi
				newname=BipEd${fileid}.${filename#*.}
				
				if $verbose; then
					echo "Copying $newname"
				fi
				
				#Ignore the unidentified subjects (2 of them in our current data set)
				if ! [ "$fileid" = "LL" ]; then
					cp ${filepath} ${sorted_dir}/t1/grenoble_Philipps/${newname}
				fi
			done
		fi	
	fi
	
	
	#Sort for Creteil
	if [ -d "${nosort_dir}/creteil" ]; then

		mkcdir ${sorted_dir}/dwi/creteil
	
		mkcdir ${sorted_dir}/t1/creteil

		#check for dwi files
		for filepath in ${nosort_dir}/creteil/hardi*; do
			
			filename=`basename $filepath`
			#In Creteil t1 files, the ID_match is the sixth to twelfth value of the file name
			newfilename=${filename:6:6}.${filename#*.}
			if $verbose; then
				echo "Copying $newfilename"
			fi

			cp ${filepath} ${sorted_dir}/dwi/creteil/${newfilename}
		done	
		
		#check for t1 files
		for filepath in ${nosort_dir}/creteil/t1*; do
			#In Creteil t1 files, the ID_match is the third to ninth value of the file name

			filename=`basename $filepath`
			newfilename=${filename:3:6}.${filename#*.}
			
			if $verbose; then
				echo "Copying $newfilename"
			fi		
			
			cp ${filepath} ${sorted_dir}/t1/creteil/${newfilename}
		done
	fi




	#Sort for pittsburgh

	if [ -d "${nosort_dir}/pittsburgh" ]; then
		
		pittsdir=${nosort_dir}/pittsburgh
		
		mkcdir ${sorted_dir}/dwi/pittsburgh

		mkcdir ${sorted_dir}/t1/pittsburgh 
		
		newpittsDWI=${sorted_dir}/dwi/pittsburgh
		newpittsT1=${sorted_dir}/t1/pittsburgh

		
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

		done
	fi


	#Sort for udine

	if [ -d "${nosort_dir}/udine" ]; then
		udine1_5T=${nosort_dir}/udine/1.5T
		udine3T=${nosort_dir}/udine/3T
		
		mkcdir ${sorted_dir}/dwi/udine1.5T 
		mkcdir ${sorted_dir}/t1/udine1.5T
		mkcdir ${sorted_dir}/dwi/udine3T
		mkcdir ${sorted_dir}/t1/udine3T

		
		newudineDWI_1_5T=${sorted_dir}/dwi/udine1.5T
		newudineT1_1_5T=${sorted_dir}/t1/udine1.5T
		newudineDWI_3T=${sorted_dir}/dwi/udine3T
		newudineT1_3T=${sorted_dir}/t1/udine3T
		echo $newudineDWI_1_5T
		
		for dirname in ${udine1_5T}/*/; do
			echo $dirname
			for filepath in ${dirname}DTI/*; do
				filename=`basename $filepath`
				newfilename=`basename $dirname`.${filename#*.}
				if $verbose; then
					echo "Copying $newfilename"
				fi	
				cp ${filepath} ${newudineDWI_1_5T}/${newfilename}

			done
			for filepath in ${dirname}T1/*; do
				echo $filepath
				filename=`basename $filepath`
				newfilename=`basename $dirname`.${filename#*.}
				if $verbose; then
					echo "Copying $newfilename"
				fi	
				cp ${filepath} ${newudineT1_1_5T}/${newfilename}

			done
		done
			
		for dirname in ${udine3T}/*/; do
			for filepath in ${dirname}DIFFUSION/*; do
		
				filename=`basename $filepath`
				newfilename=`basename $dirname`.${filename#*.}
				if $verbose; then
					echo "Copying $newfilename"
				fi	
				cp ${filepath} ${newudineDWI_3T}/${newfilename}

			done
			for filepath in ${dirname}T1/*; do
		
				filename=`basename $filepath`
				newfilename=`basename $dirname`.${filename#*.}
				if $verbose; then
					echo "Copying $newfilename"
				fi	
				cp ${dirname}/T1/* ${newudineT1_3T}/${newfilename}

			done
		done
	fi
	
	


fi



#Eddy correct the files inside the sorted directories
if $toeddy; then
	for dirpath in ${sorted_dir}/dwi/*/; do
		
		i=0
		dirname=`basename $dirpath`
		echo $dirname
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
			eddy_correct ${filepath} ${newpath}_eddy.${file_ext} trilinear
			mv ${filepath} ${dirpath}/No_EddyCorrect/${filename}
			mv ${newpath}_eddy.ecclog ${dirpath}/Logs/${fileid}_eddy.ecclog
			i=$((i+1))
			echo "$i files done, $((filesnii-i)) remaining"
		done 
	done	  
fi



if $tobet; then
	for dirpath in ${sorted_dir}/dwi/*/; do
		
		mkcdir ${dirpath}/NoBet
		mkcdir ${dirpath}/Brainmask
		
		i=0
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		echo $dirname
		searchthresh $dirname $paramfile
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to bet in $dirname with a threshold of $thresh" 
		
		
		for filepath in ${dirpath}*.nii* ; do
			
			filename=`basename $filepath`
			newpath=$(echo $filepath | sed -e "s/.nii/_brain.nii/g")
			bet ${filepath} ${newpath} -F -f ${thresh}
			i=$((i+1))
			
			mv ${filepath} ${dirpath}/NoBet/${filename}
			mv ${dirpath}/${filename%%.*}_brain_mask* ${dirpath}/Brainmask
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
		searchthresh $dirname $paramfile
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to bet in $dirname with a threshold of $thresh" 
		
		
		for filepath in ${dirpath}*.nii* ; do
			
			filename=`basename $filepath`
			newpath=$(echo $filepath | sed -e "s/.nii/_brain.nii/g")
			bet ${filepath} ${newpath} -F -f ${thresh}
			i=$((i+1))
			
			mv ${filepath} ${dirpath}/NoBet/${filename}
			mv ${dirpath}/${filename%%.*}_brain_mask* ${dirpath}/Brainmask
			echo "$i files done, $((filesnii-i)) remaining"	
		done
	done

fi
