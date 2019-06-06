# PCA (98.4%, 0.26%)
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py -1 -1 -1

# =======================================================================
# No l1 Increase TV from 0.001 to 1

# Same as PCA
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.000001 1 0.001

# Same as PCA (98.4%, 0.003%) for PC1. Strange PC2: whole brain
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.000001 1 0.01

# (81.8%, 1.8%) PC1 similar to PCA. PC2 Anterior temporal (PC1 corr PC2 ?), PATTERN1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.000001 1 0.1

# (34%, 0.6%) PC1 similar to PCA. PC2 hard to interpret. too much TV, PATTERN2
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.000001 1 1

# =======================================================================
# Very small l1
# 0.001 * l1max, 1, Increase TV from 0.1 to 1


# (81% 1.8%), PC1 similar to PCA. PC2 Anterior temporal (PC1 corr PC2 ?), PATTERN1 but better since more sparsity
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.00045588511697379 1 0.1

# (41%, 0.5%), PC1 is PC2 of pattern 1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.00045588511697379 1 0.5

# (34%, 0.7%) PC1 similar to PCA. PC2 hard to interpret. too much TV, PATTERN2
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.00045588511697379 1 1

# =======================================================================
# Small l1
# 0.01 * l1max, 1, Increase TV from 0.1 to 1

# (82% 1.8%), PC1 similar to PCA. PC2 Anterior temporal (PC1 corr PC2 ?), PATTERN1 but better since more sparsity (WINNER)
# /neurospin/brainomics/2019_rundmc_wmh/analysis/201905_rundmc_wmh_pca/models/pca_enettv_0.0046_1.000_0.100
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.0045588511697379 1 0.1

# (40%, 0.5%),
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.0045588511697379 1 0.5

# (34%, 1.2%), PC1 is PC2 of pattern 1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.0045588511697379 1 1

# =======================================================================
# medium l1
# 0.1 * l1max, 1, Increase TV from 0.1 to 1
# (81%, 1%), PC1 similar to PCA. PC2 Anterior temporal (PC1 corr PC2 ?) smaller, PATTERN1 but even more sparsity
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.045588511697379 1 0.1

# (39%, 0.3%, 1.67), PC1 is PC2 of pattern 1, + PC3
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.045588511697379 1 0.5

# (32%, 0.7%, 17%), PC1 is PC2 of pattern 1, PC3 to be explored ???
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.045588511697379 1 1

# like PCA
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py 0.000001 1 0.000001

