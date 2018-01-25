This readme describes the pipeline and codes used for the BipLi project. Most of the codes described were done in matlab,
though the reconstruction of the Twisted Projection Imaging dat file to an usuable nifti was coded in Python27.

1°) The pipeline itself

The first step that will be described here is to turn the dat files of the Twisted projection imaging (TPI) into nifti files.
This reconstruction is done through regridding, although it also creates csv files that may be used for FISTA reconstruction in future versions.

Then, these nifti files will be denoised using the BTK process "BTKNLMDenoising". 
This denoising will make the Lithium signal more clearly visible, though the reliability of such a method must still be more throughly discussed.
We may also discuss using a mask of the signal to improve the process, though due to the low resolution this is unlikely to change much.

Once this is done, we may now do a correction based on the calculated T1 value of the Lithium of this patient and the phantom acquisitions.

Once this is over, the main scripts doing the repositioning of the lithium files to the MNI space can be launched. This is done using spm processes.
Specifically: First, the Lithium data matrix is compared to the Lithium 7T anatomical image to ensure that they are aligned in spm.
Then, the 7T anatomical image is coregistered to the 3T anatomical image, (using only translations and rotations), and the rotation matrix is then applied to
the matrix of the Lithium image to place it in the 3T space.
Finally, the 3T image is transformed (with deformations) to the MNI space using the Segment function of spm12. Said transformation is then applied to the Lithium images, 
placing them all in the same MNI space.



launch_reconstruct_TPI.m
	This is the first function that converts the raw dat files of Siemens into usable nifti files. It is mostly used to call the different subjects
	and name the outputs appropriately, while the actual conversion is handled by the python code it calls.
	Its inputs are:
		-Rawdatdir:	The address string of the directory where raw information is stored. It (so far) supposes that the files are stored like this:
			{Rawdatdir}\{subjectname}\twix7T
		All files in the twix folder of a subject (recognized by the "TPI" in name), will then be converted to nifti.
		
		-Processeddir:	The address string of the directory where the results are stored. To make classification easy,
			it gives a similar structure to the files, that is:
			{Processeddir}\{subjectname}\TPI\Reconstruct_gridding\01-Raw
			The appropriate folders are created if they do not yet exist
		
		-Codedir: The address string of the directory where the python codes are stored
			(Normally, this can be found in the same directory as this readme, and is named "ReconstructionTPI" so if Codedir is not specified, it will be assigned this address)
		
		-Pythonexe: the address string of the Python27 exe
			(Eventually, will try to make code Python3 compatible and make this line redundant if Python is already called by default, no matter what version it is)
			(If this variable is not specified, it will assume the "python" line is sufficient)

	It should be noted the nii files, though stored in their appropriate directories, are just sequentially named as 'Patient01, Patient 02, etc'
	It should also be noted that the flip angle degree is assumed to be specified in the dat file name, and the nii and FISTA csv files will be named accordingly.
	All echoes found within the dat file will be separately reconstructed and named as different niftis.
	This version creates the FISTA csv files, which can be theoretically used for FISTA reconstructions, though the reconstruction has not provided satisfactory results thus far.

Once this is done and you have installed BTK, you can use the script runBTK.m to quickly go through the different patients and run the Denoising Filter.
runBTK.m takes as input the address of the BTK installation as well as that of the "Processeddir" folder, and applies the NLMDenoising to all lithium files found within the "Raw-01" folders
and places them in new "Filtered-02" folders.

rawtoquantif.m will be used for the correction from the raw signal to the assumed concentration value based on the T1/T2 values and the previously obtained phantom values. 
This script goes through the Filtered-02 folders, if they are empty it will go through the Raw-01 folders. It will then simply multiply the values of the signal based on the values
stored in rawtoquantif.xlsx, stored in the "info_pipeline" folder.


calculate_all_00
       This is the "main" function that handles all the used files and launches the other processes.
	It uses:
		-Lifiles	(cells with the address strings of all Lifiles to be processed)
		-Lioutputdir7T	(string of the address directory where Lithium in anat 7T space are placed)
		-Lioutputdir3T	(string of the address directory where Lithium in anat 3T space are placed)
		-Lioutputdirmni (string of the address directory where Lithium in MNI space are placed)
		-Anat7Tfile	(string of the address of the anatomical 7T Hydrogen image)
		-Anat3Tfile	(string of the address of the anatomical 3T Hydrogen image)
		-TPMfile	(string of the address of the anatomical MNI image space)
		-segmentfile	(string of the base address of the .mat file used for )
		-normfile	(string of the base address of the .mat file used for the placement in MNI space)
		-keepniifiles	(an integer between 0 and 3 that specifies how much additional nii files are kept after the process)
			( 3 keeps all the files, 2 erases the mw files of spm_segment, 1 additionally erases the sum of the mw files, 0 erases the )

calculate_translation_mat_01
	This .m file is just to ensure the lithium files are aligned with the anatomical images in spm

calculate_coreg_mat_02
	This .m file does a quick calculation of the linear coregistration matrix from the anat7T to the anat 3T ( as ascertained by the spm coregistration process )

calculate_deform_field_03
	This .m file launches the spm process used to do the segmentation between the anatomical 3T space and the MNI space ( needs (for now) a template of the segment batch (.mat file) and the address of the used MNI space image)

apply_deformf_field_04
	This .m file applies the deformation field calculated in 03 to an image in 3T space ( usually the new Lithium files)