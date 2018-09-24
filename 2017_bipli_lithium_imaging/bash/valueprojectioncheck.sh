#!/bin/sh
# Emma Sprooten for ENIGMA-DTI
# run in a new directory eg. Proj_Dist/
# create a text file containing paths to your masked FA maps
# output in Proj_Dist.txt
 
# make sure you have FSL5!!!
 
###### USER INPUTS ###############
## insert main folder where you ran TBSS
## just above "stats/" and "FA/"
maindir="/enigmaDTI/TBSS/run_tbss/"
list=`find $maindir -wholename "*/FA/*_masked_FA.nii.gz"`
 
## insert full path to mean_FA, skeleton mask and distance map
## based on ENIGMA-DTI protocol this should be:
mean_FA="/enigmaDTI/TBSS/ENIGMA_targets/mean_FA_MyMasked.nii.gz"
mask="/enigmaDTI/TBSS/ENIGMA_targets/mean_FA_skeleton_MyMasked.nii.gz"
dst_map="/enigmaDTI/TBSS/ENIGMA_targets/enigma_skeleton_mask_dst.nii.gz"
 
##############
### from here it should be working without further adjustments
 
rm Proj_Dist.txt
echo "ID" "Mean_Squared" "Max_Squared" >> Proj_Dist.txt
 
 
## for each FA map
    for FAmap in ${list}   
    do
	base=`echo $FAmap | awk 'BEGIN {FS="/"}; {print $NF}' | awk 'BEGIN {FS="_"}; {print $1}'`
        dst_out="dst_vals_"$base""
 
	# get Proj Dist images
        tbss_skeleton -d -i $mean_FA -p 0.2 $dst_map $FSLDIR/data/standard/LowerCingulum_1mm $FAmap $dst_out
 
	#X direction
	Xout=""squared_X_"$base"
	file=""$dst_out"_search_X.nii.gz"
	fslmaths $file -mul $file $Xout
 
	#Y direction
	Yout=""squared_Y_"$base"
	file=""$dst_out"_search_Y.nii.gz"
	fslmaths $file -mul $file $Yout
 
	#Z direction
        Zout=""squared_Z_"$base"
        file=""$dst_out"_search_Z.nii.gz"
	fslmaths $file -mul $file $Zout
 
	#Overall displacement
	Tout="Total_ProjDist_"$base""
	fslmaths $Xout -add $Yout -add $Zout $Tout
 
	# store extracted distances
	mean=`fslstats -t $Tout -k $mask -m`  
	max=`fslstats -t $Tout -R | awk '{print $2}'`
        echo "$base $mean $max" >> Proj_Dist.txt
 
        # remove X Y Z images
        ## comment out for debugging
        rm ./dst_vals_*.nii.gz
        rm ./squared_*.nii.gz
 
	echo "file $Tout done"
    done
