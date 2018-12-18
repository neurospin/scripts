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
		mkdir -p $1
	fi	
}

rcm() {
	if ! [ -z "$(ls -A $1)" ]; then
	   rm $1/*
	fi	
}


reorganizebvec() {
	
	dirname=$1
	gethresh=false
	if [ $# -lt 1 ]; then
		bvecinput="/neurospin/ciclops/projects/BIPLi7/ClinicalData/Processed_Data/2017_01_24/Diffusion/03-Bet/cmrr_diff_b1500_60dir_TE67_eddy.bvec"
	else
		bvecinput=$1
	fi
	if [ $# -lt 2 ]; then
		bvecoutput="/neurospin/ciclops/projects/BIPLi7/ClinicalData/Processed_Data/2017_01_24/Diffusion/04-Flirt/cmrr_diff_b1500_60dir_TE67_eddy.bvec"
	else
		bvecoutput=$2
	fi

	touch ${bvecoutput}
	while IFS='' read -r line || [[ -n "$line" ]]; do
			newline=$(echo $line | sed -e "s/	/ /g")
			newline=$(echo $newline | sed -e "s/\./0\./g")
			#echo ${newline}
			echo ${newline} >> ${bvecoutput}
			i=1
	done < "$bvecinput"

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
preparefield_map=false
#Eddy_correct for the nifti files (cannot run without doing sort)
toeddy=false
#parameddy="parameddy.txt"
#Brain image extraction for the nifti files (cannot run without doing sort)
tobet=false
toflirt=false
tofugue=false
todtifit=false

fullsort=false
if ${fullsort}; then
	preparefield_map=true
	tosort=true;
	toeddy=true;
	tobet=true;
	#toflirt=true;
	tofugue=true;
	todtifit=true;
fi
#paramrunbet="paramrunbet.txt"
#parambetdwi="parambetdwi.txt"
#parambett1="parambett1.txt"
#Aligns the T1 to the ICBM Atlas (provided by John Hopkins University, 1mm resolution)


totbss=false
valuecheck=false
ROIextract=true

fulltbss=false
if ${fulltbss}; then
	totbss=true
	valuecheck=true
	ROIextract=true
fi
#bdpaddress="~/Documents/BrainSuite15c/bdp/bdp.sh"
bdpaddress="../BrainSuite15c/bdp/bdp.sh"

#Runs the dtifit which outputs the FA files (cannot run without doing Bet)




#Check for input, if there are arguments, the first argument is the directory we grab the data from (root of the sites data)
#To improve: add error checks if args are not dir
#By default, the input folder is "sandbox"

if [ $# -lt 1 ]; then
	raw_dir="/neurospin/ciclops/projects/BIPLi7/Clinicaldata/Raw_Data"
	sorted_dir="/neurospin/ciclops/projects/BIPLi7/Clinicaldata/Processed_Data"
else
	raw_dir=$1/Raw_Data
	sorted_dir=$1/Processed_Data
fi

if [ $# -lt 2 ]; then
	analysisdir="/neurospin/ciclops/projects/BIPLi7/Clinicaldata/Analysis/"
else
	sorted_dir=$2
fi

if [ $# -lt 3 ]; then
	#specificdir="2*"
	specificdir="2018_12_07"
else
	specificdir=$3
fi



#If new directory doesn't exist, make it.
#Only works if all dirs are made except one. Ex: creates "dwi" not "myfold/dwi" if myfold doesn't exist

mkcdir $sorted_dir

#mkcdir ${sorted_dir}/t1
convertIMAtonii=true

fieldmapfolder=Diffusion/00-Fieldmap
rawfolder=Diffusion/01-Raw
eddyfolder=Diffusion/02-Eddy
betfolder=Diffusion/03-Bet
flirtfolder=Diffusion/04-Flirt
fuguefolder=Diffusion/04-Fugue
dtifitfolder=Diffusion/05-Dtifit
logsfolder=Diffusion/Log

#reorganizebvec

#This launches the sorting of all the input files into the output directory
if $tosort; then
	echo "Running sort"


	#Sort for Bipli
	for subjpath in ${raw_dir}/${specificdir}/; do
		
		subjname=`basename $subjpath`
		
		mkcdir ${sorted_dir}/${subjname}/Diffusion
		mkcdir ${sorted_dir}/${subjname}/Diffusion/IMAfiles
		
		#echo ${sorted_dir}/${subjname}/Diffusion
		
		if $verbose; then
			echo "Copying $subjname"
		fi
			
		
		
		if $convertIMAtonii; then
		
			diff_sname=${subjpath}/DICOM3T/cmrr*
			diff_bname=${subjpath}/DICOM3T/CMRR*
			#diff_folder=DICOM3T/${diff_name}
			for diff_folder in ${diff_sname}*/ ${diff_bname}*/; do 
				echo ${diff_folder}
				if [ -d "$diff_folder" ]; then
					mkcdir ${sorted_dir}/${subjname}/${rawfolder}
					rcm ${sorted_dir}/${subjname}/${rawfolder}
					#echo "Running dcm2nii in ${subjpath}${diff_folder}/*"
					cp ${diff_folder}/* ${sorted_dir}/${subjname}/Diffusion/IMAfiles
					#echo ${sorted_dir}/${newdirname}/Diffusion/IMAfiles
				
					#dcm2nii -4 ${diff_folder}/*
					dcm2nii -4 ${sorted_dir}/${subjname}/Diffusion/IMAfiles/*
					
					#mv ${diff_folder}/*nii.gz* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.nii.gz
					#mv ${diff_folder}/*bval* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.bval
					#mv ${diff_folder}/*bval* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.bvec
					diff_name=`basename $diff_folder`
					#echo ${sorted_dir}/${subjname}/Diffusion/IMAfiles/*nii.gz*
					#echo ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.nii.gz
					mv ${sorted_dir}/${subjname}/Diffusion/IMAfiles/*nii.gz* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.nii.gz
					mv ${sorted_dir}/${subjname}/Diffusion/IMAfiles/*bval* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.bval
					mv ${sorted_dir}/${subjname}/Diffusion/IMAfiles/*bvec* ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.bvec
					
					fslreorient2std ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.nii.gz ${sorted_dir}/${subjname}/${rawfolder}/${diff_name}.nii.gz
				fi; 
			done

			
		fi
	done
fi

if $preparefield_map; then
	echo "Running Field map calculation"
	for subjpath in ${raw_dir}/${specificdir}*/; do
		
		betval=0.7
		echodif=2.46
		
		subjname=`basename $subjpath`
		Magname=B0MAP_1
		Phasename=B0MAP_2
		DICOMfolder=DICOM3T
		Magfolder=${subjpath}/${DICOMfolder}/${Magname}
		Phasefolder=${subjpath}/${DICOMfolder}/${Phasename}
		dcm2nii -4 ${Magfolder}/*
		dcm2nii -4 ${Phasefolder}/*
		fieldfoldersubj=${sorted_dir}/${subjname}/${fieldmapfolder}
		mkcdir ${fieldfoldersubj}
		rcm ${fieldfoldersubj}
		mv ${Magfolder}/*.nii* ${fieldfoldersubj}/B0MAP_Mag.nii.gz
		mv ${Phasefolder}/*.nii* ${fieldfoldersubj}/B0MAP_Phase.nii.gz
		bet ${fieldfoldersubj}/B0MAP_Mag.nii.gz ${fieldfoldersubj}/B0MAP_Mag_bet.nii.gz -F -f 0.7
		fsl_prepare_fieldmap SIEMENS ${fieldfoldersubj}/B0MAP_Phase.nii.gz ${fieldfoldersubj}/B0MAP_Mag_bet.nii.gz ${fieldfoldersubj}/Fieldmap.nii.gz ${echodif}
	done
fi
		
#Eddy correct the files inside the sorted directories
if $toeddy; then
	echo "Running Eddy"
	for subjpath in ${sorted_dir}/${specificdir}*/; do
		i=0
		subjname=`basename $subjpath`
		for filepath in ${subjpath}/${rawfolder}/*.nii* ; do
		
			filename=`basename $filepath`
			fileid=${filename%%.*}
			file_ext=${filename#*.}
			newfilepath=${subjpath}/${eddyfolder}/${fileid}
			mkcdir ${sorted_dir}/${subjname}/${eddyfolder}
			mkcdir ${sorted_dir}/${subjname}/${logsfolder}
			rcm ${sorted_dir}/${subjname}/${eddyfolder}
			rcm  ${sorted_dir}/${subjname}/${logsfolder}
			eddy_correct ${filepath} ${newfilepath}_eddy.${file_ext} 0 trilinear
			#echo "Eddy example"
			#echo  ${filepath} ${newfilepath}_eddy.${file_ext}
			bash fdt_rotate_bvecs.sh ${subjpath}/${rawfolder}/${fileid}.bvec ${newfilepath}_eddy.bvec ${newfilepath}_eddy.ecclog
			mv ${newfilepath}_eddy.ecclog ${subjpath}/${logsfolder}/${fileid}_eddy.ecclog
			cp ${subjpath}/${rawfolder}/*bval ${subjpath}/${eddyfolder}/
		done 
	done	  
fi



if $tobet; then
	echo "Running Bet"
	for subjpath in ${sorted_dir}/${specificdir}*/; do
		i=0
		subjname=`basename $subjpath`
		for filepath in ${subjpath}/${eddyfolder}/*.nii* ; do
			thresh=0.2
			filename=`basename $filepath`
			#newpath=$(echo $filepath | sed -e "s/.nii/__brain.nii/g")
			fileid=${filename%%.*}
			file_ext=${filename#*.}
			newfilepath=${subjpath}/${betfolder}/${fileid}
			echo ${newfilepath}
			mkcdir ${sorted_dir}/${subjname}/${betfolder}
			rcm ${sorted_dir}/${subjname}/${betfolder}
			bet ${filepath} ${newfilepath}.${file_ext} -m -F -f ${thresh}
			mv ${newfilepath}_mask* ${subjpath}/${logsfolder}/
			#echo "bet example"
			#echo ${filepath} ${newfilepath}.${file_ext} -F -f ${thresh}
			cp ${subjpath}/${eddyfolder}/*bval ${subjpath}/${betfolder}/
			cp ${subjpath}/${eddyfolder}/*bvec ${subjpath}/${betfolder}/
		done
	done
fi

if $tofugue; then
	echo "Running Fugue"
	for subjpath in ${sorted_dir}/${specificdir}*/; do
		i=0
		dwell=0.00246
		subjname=`basename $subjpath`
		for filepath in ${subjpath}/${betfolder}/*.nii* ; do
			mkcdir ${subjpath}/${fuguefolder}
			rcm ${subjpath}/${fuguefolder}
			filename=`basename $filepath`
			fileid=${filename%%.*}
			file_ext=${filename#*.}
			fugue -i ${sorted_dir}/${subjname}/${betfolder}/*.nii.gz --dwell=${dwell} --loadfmap=${fieldfoldersubj}/Fieldmap.nii.gz -u ${sorted_dir}/${subjname}/${fuguefolder}/${fileid}.${file_ext}
			#echo -i ${sorted_dir}/${subjname}/${betfolder}/*.nii.gz --dwell=${dwell} --loadfmap=${fieldfoldersubj}/Fieldmap.nii.gz -u ${sorted_dir}/${subjname}/${fuguefolder}/${fileid}_fugue.${file_ext}
			cp ${subjpath}/${betfolder}/*bval ${subjpath}/${fuguefolder}/
			cp ${subjpath}/${betfolder}/*bvec ${subjpath}/${fuguefolder}/
		done
	done
fi


if $toflirt; then
	echo "Running flirt"
	for subjpath in ${sorted_dir}/${specificdir}*/; do
		i=0
		subjname=`basename $subjpath`
		for filepath in ${subjpath}/${betfolder}/*.nii* ; do
			filename=`basename $filepath`
			fileid=${filename%%.*}
			file_ext=${filename#*.}
			newfilepath=${subjpath}/${flirtfolder}/${fileid}
			echo ${newfilepath}
			mkcdir ${subjpath}/${flirtfolder}
			rcm ${subjpath}/${flirtfolder}
			flirt -in ${filepath} -ref /volatile/fsl/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain -out ${newfilepath}.${file_ext} -omat ${newfilepath}.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
			flirt -in ${filepath} -ref /volatile/fsl/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain -out ${newfilepath}.${file_ext} -init ${newfilepath}.mat -applyxfm
			cp ${subjpath}/${betfolder}/*bval ${subjpath}/${flirtfolder}/
			#cp ${subjpath}/${betfolder}/*bvec ${subjpath}/${flirtfolder}/
			reorganizebvec ${subjpath}/${betfolder}/*bvec ${newfilepath}.bvec
		done
	done
fi

#Needs for the images to be nifti, needs to be Betted (brain mask), needs the bvec and bval, if they have some susceptibility artefacts, the warp to t1 or some other method must be employed
if $todtifit; then
	echo "Running DTIFIT"
	for subjpath in ${sorted_dir}/${specificdir}*/; do
		i=0
		subjname=`basename $subjpath`
		inputfolder=${fuguefolder}
		for filepath in ${subjpath}/${inputfolder}/*.nii* ; do
			filename=`basename $filepath`
			fileid=${filename%%.*}
			file_ext=${filename#*.}
			newfilepath=${subjpath}/${dtifitfolder}/${fileid}
			#echo ${newfilepath}
			echo ${subjpath}
			mkcdir ${subjpath}/${dtifitfolder}
			#echo -k ${filepath} -m ${subjpath}/${logsfolder}/*_mask* -r ${subjpath}/${inputfolder}/${Idname}*bvec -b ${subjpath}/${inputfolder}/${Idname}*bval -o ${newfilepath}.${file_ext}
			#echo ${subjpath}/${dtifitfolder}/*
			rcm ${subjpath}/${dtifitfolder}
			dtifit -k ${filepath} -m ${subjpath}/${logsfolder}/*_mask* -r ${subjpath}/${inputfolder}/${Idname}*bvec -b ${subjpath}/${inputfolder}/${Idname}*bval -o ${newfilepath}.${file_ext}
			mkcdir ${analysisdir}
			mkcdir ${analysisdir}/run_tbss
			cp ${newfilepath}*FA* ${analysisdir}/run_tbss/${subjname}_FA.nii.gz
		done
	done
fi

if $totbss; then

	#FSLPATH="/volatile/fsl/"
	Cingfile=$FSLDIR/data/standard/LowerCingulum_1mm
	part1=true
	if $part1; then
		echo "Running TBSS"
		previousdir=${PWD}
		cd ${analysisdir}/run_tbss/
		tbss_1_preproc *.nii.gz
		tbss_2_reg -t ${analysisdir}/ENIGMA_targets/ENIGMA_DTI_FA.nii.gz
		tbss_3_postreg -S
		mkcdir ${analysisdir}/ENIGMA_targets_edited/
		rcm ${analysisdir}/ENIGMA_targets_edited/
		fslmerge -t ./all_FA_QC ./FA/*FA_to_target.nii.gz
		fslmaths ./all_FA_QC -bin -Tmean -thr 0.9 ${analysisdir}/ENIGMA_targets_edited/mean_FA_mask.nii.gz
		fslmaths ${analysisdir}/ENIGMA_targets/ENIGMA_DTI_FA.nii.gz -mas ${analysisdir}/ENIGMA_targets_edited/mean_FA_mask.nii.gz ${analysisdir}/ENIGMA_targets_edited/mean_FA.nii.gz
		fslmaths ${analysisdir}/ENIGMA_targets/ENIGMA_DTI_FA_skeleton.nii.gz -mas ${analysisdir}/ENIGMA_targets_edited/mean_FA_mask.nii.gz ${analysisdir}/ENIGMA_targets_edited/mean_FA_skeleton.nii.gz
		cd ${analysisdir}/ENIGMA_targets_edited
		tbss_4_prestats -0.049
	fi
	
	part2=true
	if $part2; then
		cd ${analysisdir}/run_tbss/
		cd ${previousdir}
		mkcdir ${analysisdir}/run_tbss/skeletons
		rcm ${analysisdir}/run_tbss/skeletons
		for filepath in ${analysisdir}/run_tbss/FA/*_target.nii.gz; do
			filename=`basename $filepath`
			fileid=${filename%%_FA*}
			file_ext=${filename#*.}
			FA_indivfolder=${analysisdir}/run_tbss/FA_individ
			echo ${fileid}
			mkcdir ${FA_indivfolder}/${fileid}/stats
			mkcdir ${FA_indivfolder}/${fileid}/FA
			rcm ${FA_indivfolder}/${fileid}/stats
			rcm ${FA_indivfolder}/${fileid}/FA
			cp ${filepath} ${FA_indivfolder}/${fileid}/FA/
			fslmaths ${FA_indivfolder}/${fileid}/FA/${fileid}*FA_to_target.nii.gz -mas ${analysisdir}/ENIGMA_targets_edited/mean_FA_mask.nii.gz ${FA_indivfolder}/${fileid}/FA/${fileid}_masked_FA.nii.gz
			tbss_skeleton -i ${FA_indivfolder}/${fileid}/FA/${filedid}*masked_FA* -p 0.049 ${analysisdir}/ENIGMA_targets_edited/mean_FA_skeleton_mask_dst ${Cingfile} ${FA_indivfolder}/${fileid}/FA/${fileid}_masked_FA.nii.gz ${FA_indivfolder}/${fileid}/stats/${fileid}_masked_FAskel.nii.gz -s ${analysisdir}/ENIGMA_targets_edited/mean_FA_skeleton_mask.nii.gz
			cp ${FA_indivfolder}/${fileid}/stats/${fileid}_masked_FAskel.nii.gz ${analysisdir}/run_tbss/skeletons 
		done
	fi
fi

if $valuecheck; then
	maindir=${analysisdir}/run_tbss/
	list=`find $maindir -wholename "*/FA/*_masked_FA.nii.gz"`

	## insert full path to mean_FA, skeleton mask and distance map
	## based on ENIGMA-DTI protocol this should be:
	ENIGMAdir=${analysisdir}/ENIGMA_targets_edited/
	mean_FA=${analysisdir}/run_tbss/stats/"mean_FA_mask.nii.gz"
	mask=${analysisdir}/run_tbss/stats/"mean_FA_skeleton.nii.gz"
	dst_map=${ENIGMAdir}/"mean_FA_skeleton_mask_dst.nii.gz"
	##############
	### from here it should be working without further adjustments
	if 	[ -f ${analysisdir}/Proj_Dist.txt ]; then
		rm ${analysisdir}/Proj_Dist.txt
	fi
	echo "ID" "Mean_Squared" "Max_Squared" >> Proj_Dist.txt
	#echo ${list}
 
## for each FA map
    for FAmap in ${list}; do
		#echo ${FAmap}
		echo ${list}
		#base=`echo $FAmap | awk 'BEGIN {FS="/"}; {print $NF}' | awk 'BEGIN {FS="_"}; {print $1}'`
		filename=`basename $FAmap`
		base=${filename%%_masked*}
		valuecheckfold=${analysisdir}/run_tbss/valuecheck/
		mkcdir ${valuecheckfold}
		dst_out=${valuecheckfold}"dst_vals_"$base""
		# get Proj Dist images
		tbss_skeleton -d -i $mean_FA -p 0.2 $dst_map $FSLDIR/data/standard/LowerCingulum_1mm $FAmap $dst_out
		#echo -d -i $mean_FA -p 0.2 $dst_map $FSLDIR/data/standard/LowerCingulum_1mm.nii.gz $FAmap $dst_out
#	 	
		#X direction
		Xout=${valuecheckfold}""squared_X_"$base"
		file=""$dst_out"_search_X.nii.gz"
		fslmaths $file -mul $file $Xout
		#echo ${Xout}
		#echo ${file}
		#echo ${base}
#	 
		#Y direction
		Yout=${valuecheckfold}""squared_Y_"$base"
		file=""$dst_out"_search_Y.nii.gz"
		fslmaths $file -mul $file $Yout
	 
		#Z direction
		Zout=${valuecheckfold}""squared_Z_"$base"
		file=""$dst_out"_search_Z.nii.gz"
		fslmaths $file -mul $file $Zout
 
		#Overall displacement
		Tout=${valuecheckfold}"Total_ProjDist_"$base""
		fslmaths $Xout -add $Yout -add $Zout $Tout
	 
		# store extracted distances
		mean=`fslstats -t $Tout -k $mask -m`  
		max=`fslstats -t $Tout -R | awk '{print $2}'`
		echo "$base $mean $max" >> Proj_Dist.txt
	 
		# remove X Y Z images
		## comment out for debugging
		rm ${valuecheckfold}/dst_vals_*.nii.gz
		rm ${valuecheckfold}/squared_*.nii.gz
	 
		echo "file $Tout done"
    done
fi

if $ROIextract; then
	
	maindir=${analysisdir}/run_tbss/
	csvoutputdir1=${maindir}ENIGMA_ROI_part1
	part1=false
	if ${part1}; then
		#make an output directory for all files
		maindir=${analysisdir}/run_tbss/
		csvoutputdir1=${maindir}ENIGMA_ROI_part1
		mkcdir ${csvoutputdir1}
		statsdir=${maindir}stats/
		ENIGMAdir=${analysisdir}/ENIGMA_targets_edited/

		for subjectpath in ${maindir}/FA_individ/2*/
		do
			subjectname=`basename $subjectpath`
			./singleSubjROI_exe ENIGMA_look_up_table.txt ${statsdir}/mean_FA_skeleton.nii.gz ${ENIGMAdir}/JHU-ICBM-labels-1mm.nii.gz ${csvoutputdir1}/${subjectname}_ROIout ${maindir}/skeletons/${subjectname}_masked_FAskel.nii.gz
		done
	fi
	part2=false
	if ${part2}; then
		#######
		## part 2 - loop through all subjects to create ROI file 
		##			removing ROIs not of interest and averaging others
		#######

		#make an output directory for all files
		csvoutputdir2=${maindir}/ENIGMA_ROI_part2
		mkcdir ${csvoutputdir2}

		# you may want to automatically create a subjectList file 
		#    in which case delete the old one
		#    and 'echo' the output files into a new name
		rm ${maindir}/subjectList.csv

		for subjectpath in ${maindir}/FA_individ/2*/
		do
			subjectname=`basename $subjectpath`
			./averageSubjectTracts_exe ${csvoutputdir1}/${subjectname}_ROIout.csv ${csvoutputdir2}/${subjectname}_ROIout_avg.csv
			# can create subject list here for part 3!
			#echo ${csvoutputdir1}/${subjectname}_ROIout.csv ${csvoutputdir2}/${subjectname}_ROIout_avg.csv
			echo ${subjectname},${csvoutputdir2}/${subjectname}_ROIout_avg.csv >> ${maindir}/subjectList.csv
		done
	fi
	part3=true
	if ${part3}; then
		#######
		## part 3 - combine all 
		#######
		Table=${maindir}/ALL_Subject_Info.txt
		subjectIDcol=DateAcq
		subjectList=${maindir}/subjectList.csv
		outTable=${maindir}/combinedROItable.csv
		Ncov=2
		covariates="Age;Sex"
		Nroi="all" #2
		rois="IC;EC"

		#location of R binary 
		Rbin=R

		#Run the R code
		${Rbin} --no-save --slave --args ${Table} ${subjectIDcol} ${subjectList} ${outTable} ${Ncov} ${covariates} ${Nroi} ${rois} <  ./combine_subject_tables.R  
	fi
fi
