Project to study patterns in the white matter hyper-intensities (WMH).

# History

The first batch of results have been presented to MESCOG meeting in January 2014.
They are stored in `/neurospin/mescog/proj_wmh_patterns.2014_01`.
In those we always used the French subpopulation as training and the German one as test.

Since then we have 

# Data

Data are stored in `/neurospin/mescog/proj_wmh_patterns`.
We use the images in processed by Marco as input.

We have found that some subjects are very far from the average.
We have tried to detect outliers with the scripts `00_outliers_detection.py` and `00_inspect_outliers.py` but this is no longer used since Marco cleaned them.

# Scripts

* `00_build_dataset.py`: concatenates images and create data set.
  * input: `/neurospin/mescog/neuroimaging/original/munich/CAD_norm_M0/WMH_norm/*M0-WMH_norm.nii.gz`
  * output: `/neurospin/mescog/datasets/CAD-WMH-MNI.npy`, `/neurospin/mescog/datasets/CAD-WMH-MNI-subjects.txt`
* `00_quality_control.py`: compute some basic statistics
* `01_clustering.prepare_dataset.py`: split dataset in french and german subsets and center them
  * output: `/neurospin/mescog/proj_wmh_patterns/french.npy`, `/neurospin/mescog/proj_wmh_patterns/germans.npy`, `/neurospin/mescog/proj_wmh_patterns/french.center.npy`, `/neurospin/mescog/proj_wmh_patterns/germans.center.npy`
* `01_clustering.PCA.py`: apply PCA
  * output: in `/neurospin/mescog/proj_wmh_patterns/PCA`
* `01_clustering.Ward.py`: apply Ward clustering
 * output: in `/neurospin/mescog/proj_wmh_patterns/clustering/Ward`
* `01_clustering.kmeans.py`: apply k-means clustering
 * output: in `/neurospin/mescog/proj_wmh_patterns/clustering/kmeans`
* `02_mulm_plot_PCA.py` `02_mulm_plot_PCA.R`: plots for PCA interpretation
