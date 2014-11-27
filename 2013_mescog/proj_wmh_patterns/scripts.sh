cd /home/ed203246/data/mescog/wmh_patterns/results/meshs
fsl5.0-fslsplit l1_+_l2_+_TV_opppc0.nii ./l1_+_l2_+_TV_opppc0 -t

~/git/scripts/brainomics/image_clusters_analysis.py --input l1_+_l2_+_TV_opppc00001.nii.gz

~/git/scripts/brainomics/image_clusters_analysis.py --input l1_+_l2_+_TV_opppc00002.nii.gz

l1_+_l2_+_TV_opppc00001

~/git/scripts/brainomics/image_clusters_rendering.py l1_+_l2_+_TV_opppc00001 l1_+_l2_+_TV_opppc00002

