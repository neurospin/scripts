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
tosort=false
#Eddy_correct for the nifti files (cannot run without doing sort)
toeddy=false
parameddy="parameddy.txt"
#Brain image extraction for the nifti files (cannot run without doing sort)
tobet=false
paramrunbet="paramrunbet.txt"
parambetdwi="parambetdwi.txt"
parambett1="parambett1.txt"
#Aligns the T1 to the ICBM Atlas (provided by John Hopkins University, 1mm resolution)
towarpwT1=false
towarpwT1suite=true
#bdpaddress="~/Documents/BrainSuite15c/bdp/bdp.sh"
bdpaddress="../BrainSuite15c/bdp/bdp.sh"

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
				#echo "this is the Idname: $Id_name"
				
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
				bash fdt_rotate_bvecs.sh ${dirpath}${fileid}.bvec ${dirpath}${fileid}__eddy.bvec ${newpath}__eddy.ecclog
				mv ${newpath}__eddy.ecclog ${dirpath}/Logs/${fileid}__eddy.ecclog
				mv ${dirpath}${fileid}.bvec ${dirpath}/No_EddyCorrect/${fileid}.bvec
				i=$((i+1))
				echo "$i files done, $((filesnii-i)) remaining"
			done 
		fi
	done	  
fi



if $tobet; then
	echo "Running Bet"
	
	
	#for filepath in ${sorted_dir}/t1/mannheim/*.nii*
	#	python fixt1.py $filepath
	#done
	
	for dirpath in ${sorted_dir}/dwi/*/; do
		
		mkcdir ${dirpath}/NoBet
		mkcdir ${dirpath}/Brainmask
		
		i=0
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		echo $dirname
		toruneddy $dirname $paramrunbet
		searchthresh $dirname $parambetdwi
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

#*	for dirpath in ${sorted_dir}/t1/*/; do
#		
#		mkcdir ${dirpath}/NoBet
#		mkcdir ${dirpath}/Brainmask
#		
#		i=0
#		dirname=`basename $dirpath`
#		dirname=$(echo $dirname | sed -e "s/\///g")
#		echo $dirname
#		searchthresh $dirname $parambett1
#		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
#		echo "$filesnii to bet in $dirname with a threshold of $thresh" 
#		
#		
#		for filepath in ${dirpath}*.nii* ; do
#			
#			filename=`basename $filepath`
#			newpath=$(echo $filepath | sed -e "s/.nii/__brain.nii/g")
#			bet ${filepath} ${newpath} -S -f ${thresh} -g 0
#			#bet ${newpath} ${newpath} -R -f ${thresh} -g 0
#			i=$((i+1))
#			
#			mv ${filepath} ${dirpath}/NoBet/${filename}
#			#mv ${dirpath}/${filename%%.*}__brain_mask* ${dirpath}/Brainmask
#			fslreorient2std ${newpath} ${newpath}
#			
#			echo "$i files done, $((filesnii-i)) remaining"	
#		done
#	done

fi

if $towarpwT1suite; then
	echo "Dealing with susceptibility artefacts by using the T1"
	for dwipath in ${sorted_dir}/dwi/mannheim/; do
		dirname=`basename $dwipath`
		dirname=$(echo $dirname | sed -e "s/\///g")		
		echo $dirname		
		t1path=${sorted_dir}/t1/${dirname}
		
		i=0
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to warp in $dirname" 
		for filepath in ${dwipath}*.nii* ; do
			filename=`basename $filepath`
			filename=${filename%%__*}
			bash $bdpaddress ${t1path}/${filename}*.bfc.nii.gz --tensor --odf --nii ${filepath} -g ${dwipath}/${filename}*.bvec -b ${dwipath}/${filename}*.bval 
			mkcdir ${t1path}/${filename}
			mv ${t1path}/${filename}.* ${t1path}/${filename}/
			echo ${t1path}/${filename}/
		done
	done
fi	

			
			

if $towarpwT1; then
	echo "Dealing with susceptibility artefacts by using the T1"
	for dirpath in ${sorted_dir}/t1/*/; do
		
		
		dirname=`basename $dirpath`
		dirname=$(echo $dirname | sed -e "s/\///g")
		echo $dirname		
		dwipath=${sorted_dir}/dwi/${dirname}/
		
		mkcdir ${dirpath}/Matfiles
		mkcdir ${dwipath}/Matfiles
		mkcdir ${dirpath}/Nowarp
		mkcdir ${dwipath}/Nowarp
		i=0

		#searchthresh $dirname $parambett1
		filesnii=$(find ${dirpath} -type f -print | grep nii | wc -l) 
		echo "$filesnii to warp in $dirname" 
		
		
		
		for filepath in ${dirpath}*.nii* ; do
			
			filename=`basename $filepath`
			filename=${filename%%__*}
			newpath=$(echo $filepath | sed -e "s/.nii/__warpedtoicbm.nii/g")
			flirt -in ${filepath} -ref /usr/share/data/jhu-dti-whitematter-atlas/JHU/JHU-ICBM-DWI-1mm.nii.gz -omat ${dirpath}/Matfiles/${filename%%.*}.omat -out ${newpath} -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
			i=$((i+1))
			bet ${newpath} ${newpath} -f 0.5 -g 0
			echo "first flirt done"
			
			flirt -in ${dwipath}${filename}*.nii* -ref ${newpath} -omat ${dwipath}Matfiles/${filename%%.*}2.omat -out ${dwipath}${filename%%.*}_warped.nii.gz -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 9  -interp trilinear
			echo "second flirt done"
			rm ${dwipath}${filename%%.*}_warped.nii.gz

			echo "flirt -in ${dwipath}${filename}*.nii* -applyxfm -init ${dwipath}Matfiles/${filename%%.*}2.omat -out ${dwipath}${filename%%.*}_warped2.nii.gz -paddingsize 0.0 -interp trilinear -ref ${newpath}"
			flirt -in ${dwipath}${filename}*.nii* -applyxfm -init ${dwipath}Matfiles/${filename%%.*}2.omat -out ${dwipath}${filename%%.*}__warped.nii.gz -paddingsize 0.0 -interp trilinear -ref ${newpath}

			
			#flirt -in ${dwipath}${filename}*.nii* -ref ${newpath} -out ${dwipath}${filename%%.*}_warped2.nii.gz -init ${dwipath}Matfiles/${filename%%.*}2.omat -applyxfm
			echo "third flirt done"
			
			
			#mv ${dirpath}/${filename%%.*}__brain_mask* ${dirpath}/Brainmask
			
			
			#flirt -in ${filepath} -ref /usr/share/data/jhu-dti-whitematter-atlas/JHU/JHU-ICBM-DWI-1mm.nii.gz -omat ${dirpath}/Matfiles/${filename%%.*}1.omat -out ${newpath} -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
			#flirt -in ${dwipath}${filename} -ref ${filepath} -omat ${dwipath}Matfiles/${filename%%.*}2.omat -out ${dwipath}${filename%%.*}_warped.nii.gz -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
			#convert_xfm -concat ${dirpath}/Matfiles/${filename%%.*}1.omat -omat ${dwipath}Matfiles/${filename%%.*}3.omat ${dwipath}Matfiles/${filename%%.*}2.omat
			#flirt -in ${dwipath}${filename} 
			
			#mv ${dwipath}${filename} ${dirpath}/Nowarp/${filename}
			
			echo "$i files done, $((filesnii-i)) remaining"	
		done		
		
	done
fi	

#Needs for the images to be nifti, needs to be Betted (brain mask), needs the bvec and bval, if they have some susceptibility artefacts, the warp to t1 or some other method must be employed
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
