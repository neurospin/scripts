This directory is to be used with matlab
The explanations of each file can be found at the top of each file

Example.m
Contains an example of use MEGHASurf.m (vertex-wise analysis, such as analyse the sulcal depth extracted with Freesurfer)

if the phenotype is sulc (or curv or curv.pial)
rh.sulc/ and lh.sulc/ should be replaced in WriteMap.m and ExtractSurf by the correct corresponding directory such as rh.curv/

The variables should be self explanatory but among them are:
the GRM matrix (format txt file for MEGHA), the GRM id matrix, the covariate file, the radical (ImgFileL\Rh) of the filename for the data of each subject, the labels of each subject (ImgSubj), the number of permutation (Nperm), the output file

It is worth noting that if we just need the Pval and heritability estimate we should set Nperm to 1, because we only need to perform a lot of permuation to find the significant cluster

Example2.m
Contains and example of use of MEGHA.m (analyse iteratively multiple phenotype such as the max depth for each sulcus)
The variables should be self explanatory but among them are:
the GRM matrix (format txt file for MEGHA), the GRM id matrix, the covariate file, the file containing all the phenotype concatenated, the number of permutation (Nperm), the output file

It is worth noting that if we just need the Pval and heritability estimate we should set Nperm to 1, because we only need to perform a lot of permuation to find the significant phenotype
