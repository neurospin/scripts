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

	

#############
#  MAIN #####
#############


#Check for input, if there are arguments, the first argument is the directory we grab the data from (root of the sites data)
#To improve: add error checks if args are not dir
if [ $# -lt 1 ]; then
	nosort_dir="sandbox"
else
	nosort_dir=$1
fi

#Same as above, check for second argument for result directory (after sort)
if [ $# -lt 2 ]; then
	sorted_dir="sorted"
else
	sorted_dir=$2
fi

#If new directory doesn't exist, make it.
#Only works if all dirs are made except one. Ex: creates "dwi" not "myfold/dwi" if myfold doesn't exist
if ! [ -d $sorted_dir ]; then
	mkdir $sorted_dir
	if ! [ -d "${sorted_dir}/dwi" ]
		mkdir ${sorted_dir}/dwi
	fi
	if ! [ -d "${sorted_dir}/t1" ]
		mkdir ${sorted_dir}/t1
	fi
fi

#Creteil sort
if [ -d "${nosort_dir}/creteil" ]; then
	mkdir $sorted_dir/dwi/creteil
	mkdir $sorted_dir/t1/creteil
	#check for dwi files
	for filename in ${nosort_dir}/creteil/hardi*; do
		#In Creteil hardi files, the ID_match is the sixth to twelfth value of the file name
		newfilename=${filename:6:6}
		cp ${nosort_dir}/creteil/${filename} ${sorted_dir}/dwi/creteil/${newfilename}
	done	
	#check for t1 files
	for filename in ${nosort_dir}/creteil/t1*
		#In Creteil t1 files, the ID_match is the third to ninth value of the file name
		newfilename=${filename:3:6}
		cp ${nosort_dir}/creteil/${filename} ${sorted_dir}/t1/creteil/${newfilename}
	done
fi

#Sort for galway
#To come back to, the files in galway in dti look like t1 files, 
#no need to do anything else till I know what's going on and how to proceed
if [ -d "${nosort_dir}/galway" ]; then
	for filename in ${nosort_dir}/galway/dti/*
		newfilename="nu_${filename:0:5}"
		cp ${nosort_dir}/galway/dti/${filename} ${sorted_dir}/t1/galway/${newfilename}
	for filename in ${nosort_dir}/galway/t1/*
		
		newfilename=${filename:6:6}
		cp ${nosort_dir}/galway/${filename} ${sorted_dir}/t1/galway/${newfilename}
fi



