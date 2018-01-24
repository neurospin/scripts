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