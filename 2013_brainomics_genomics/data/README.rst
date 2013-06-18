Expected data
=============

Imagen Genotyping information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From the IMAGEN disks go to :
qc/imput/1kG_HlM_VF_23mai2013/genetics_imputation/1KGPref

- get the bim file obtained from the merge of the two Illumina platforms and liftover for Build37.
- apply maf and hwe (0.05 and 10e-4)	to obtain snp_b37_intersect_maf5hwe4_qcsubject[bed bim fam] files

Imagen SNP metainformation
~~~~~~~~~~~~~~~~~~~~~~~~~~
bed file (relative to Bedtools not plink tools!).

- from the bim file
   awk 'BEGIN{FS="\t"}{printf("chr%s\t%d\t%s\t%s\n",$1,$4-1,$4,$2)}' snp_b37_intersect_maf5hwe4_qcsubject.bim > snps_b37_ref.bed
