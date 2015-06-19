export SUBJECTS_DIR=/neurospin/cati/ADNI/ADNI_510/FSdatabase/

cd /neurospin/brainomics/2013_adni/freesurfer_assembled_data

freesurfer_init


subjects=$(ls -d /neurospin/cati/ADNI/ADNI_510/FSdatabase/*_S_*| grep -v minf| sed 's#.*/##')

# lh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi lh --meas thickness --out ${s}_lh.mgh ; done

# rh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi rh --meas thickness --out ${s}_rh.mgh ; done

########################################################
# smooths by 6mm fwhm

mkdir /neurospin/brainomics/2013_adni/freesurfer_assembled_data_smoothed
export SUBJECTS_DIR=/neurospin/cati/ADNI/ADNI_510/FSdatabase/

cd /neurospin/brainomics/2013_adni/freesurfer_assembled_data_smoothed

freesurfer_init


subjects=$(ls -d /neurospin/cati/ADNI/ADNI_510/FSdatabase/*_S_*| grep -v minf| sed 's#.*/##')

# lh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi lh --meas thickness --fwhm 6 --out ${s}_lh.sm6.mgh ; done

# rh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi rh --meas thickness --fwhm 6 --out ${s}_rh.sm6.mgh ; done

# could have been done more directly with
#mri_surf2surf --hemi lh --s fsaverage --sval 114_S_1106_S22859_I49911_lh.sm6.mgh --tval 114_S_1106_S22859_I49911_lh.sm6.mgh --fwhm-trg 6 --noreshape --no-cortex

