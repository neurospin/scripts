user_func /home/ed203246/git/scripts/2013_adni/ADAS11-MCIc-CTL/02_tvenet_cs.py

# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/ADAS11-MCIc-CTL
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/config.json

# generate nifti files
cd /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL


/home/ed203246/git/scripts/2013_adni/share/weigths_map_npy_to_nii.py config.json --keys "0.001_0.3335_0.3335_0.333_-1.0
0.001_0.5_0.5_0.0_-1.0
0.001_0.09_0.81_0.1_-1.0
0.005_0.09_0.81_0.1_-1.0"



#######################
## ENET 0.001_0.5_0.5_0.0_-1.0
bv_env /home/ed203246/git/scripts/2013_adni/share/weigths_map_mesh.py --input /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/results/weigths_map/beta_0.001_0.5_0.5_0.0_-1.0.nii.gz

/home/ed203246/git/scripts/brainomics/image_clusters_analysis.py --input /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/results/weigths_map/beta_0.001_0.5_0.5_0.0_-1.0.nii_thresholded:0.003706/beta_0.001_0.5_0.5_0.0_-1.0.nii.gz

#######################
## ENET TV
bv_env /home/ed203246/git/scripts/2013_adni/share/weigths_map_mesh.py --input /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/results/weigths_map/beta_0.001_0.3335_0.3335_0.333_-1.0.nii.gz

cd /neurospin/brainomics/2013_adni/ADAS11-MCIc-CTL/results/weigths_map/beta_0.001_0.5_0.5_0.0_-1.0.nii_thresholded:0.003706/
mv clusters.nii.gz clusters.nii.gz-
ln -s beta_0.001_0.5_0.5_0.0_-1.0.nii_clusters_values.nii.gz clusters.nii.gz
mv clusters.mesh clusters.mesh-
ln -s beta_0.001_0.5_0.5_0.0_-1.0.nii_clusters.gii clusters.mesh

