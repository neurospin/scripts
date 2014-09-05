export SUBJECTS_DIR=/neurospin/cati/ADNI/ADNI_510/FSdatabase/

cd /neurospin/brainomics/2013_adni/freesurfer_assembled_data

freesurfer_init


subjects=$(ls -d /neurospin/cati/ADNI/ADNI_510/FSdatabase/*_S_*| grep -v minf| sed 's#.*/##')

# lh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi lh --meas thickness --out ${s}_lh.mgh ; done

# rh
for s in $subjects ; do mris_preproc --s $s --target fsaverage --hemi rh --meas thickness --out ${s}_rh.mgh ; done

