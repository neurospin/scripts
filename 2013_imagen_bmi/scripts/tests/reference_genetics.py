# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:35:13 2014

@authors: Mathieu Dubois and Helene Lajous

Extract SNP information (name and position) for each gene with Vincent
Frouin's recipe.

AIMS:   - check IMAGEN database
        - get a file with all snps of interest.

The recipe uses shell commands so this script just wraps them in a Python
script.

Results are stored in files 'reference_genetics/gene_name.snp'.
"""


import os
from subprocess import Popen, PIPE, STDOUT


# Pathnames
INPUT_DIR = "/neurospin/brainomics/bioinformatics_resources/data/"
INPUT_REFGENE_META = os.path.join(INPUT_DIR,
                                  "genetics",
                                  "hg19.refGene.meta")
INPUT_SNP138 = os.path.join(INPUT_DIR,
                            "snps",
                            "cleaned_snp138Common.txt")

OUTPUT_DIR = "reference_genetics"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Main genes associated to BMI according to the literature
TEST_GENES = ['BDNF', 'CADM2', 'COL4A3BP', 'ETV5', 'FAIM2', 'FANCL', 'FTO',
              'GIPR', 'GNPDA2', 'GPRC5B', 'HMGCR', 'KCTD15', 'LMX1B',
              'LRP1B', 'LINGO2', 'MAP2K5', 'MC4R', 'MTCH2', 'MTIF3', 'NEGR1',
              'NPC1', 'NRXN3', 'NTRK2', 'NUDT3', 'POC5', 'POMC', 'PRKD1',
              'PRL', 'PTBP2', 'PTER', 'QPCTL', 'RPL27A', 'SEC16B', 'SH2B1',
              'SLC39A8', 'SREBF2', 'TFAP2B', 'TMEM160', 'TMEM18', 'TNNI3K',
              'TOMM40', 'ZNF608']


# Use this command to find the position of each gene
# The chromosome is given by field 4,
# Start and end positions are respectively given by fields 2 and 3.
CHR_CMD_PATTERN = "awk '{{if ($1==\"{gene}\") print $4,$2,$3}}' {meta}"
# Use this command to find SNPs associated to BMI genes
SNP_CMD_PATTERN = "awk '{{if (($2==\"{chrom}\") && ($4>={start}) && ($4<={stop})) print $4,$5; }}' {snps}"


# Get snps' info for each gene
gene_filenames = []
for gene in TEST_GENES:
    print "Gene:", gene
    # Find chromosome and position for each gene
    cmd = CHR_CMD_PATTERN.format(gene=gene,
                                 meta=INPUT_REFGENE_META)
    proc = Popen(cmd, shell=True,
                 stdin=PIPE,
                 stdout=PIPE,
                 stderr=STDOUT,
                 close_fds=True)
    output = proc.stdout.read()
    chrom, start, stop = output.split()
    # Find SNPs included in each gene
    cmd = SNP_CMD_PATTERN.format(chrom=chrom,
                                 start=start,
                                 stop=stop,
                                 snps=INPUT_SNP138)
    proc = Popen(cmd, shell=True,
                 stdin=PIPE,
                 stdout=PIPE,
                 stderr=STDOUT,
                 close_fds=True)
    output = proc.stdout.read()
    snp_info = output.split('\n')[:-1]

    # Write results for each gene in a txt file
    snp_file_path = os.path.join(OUTPUT_DIR, "%s.snp" % gene)
    with open(snp_file_path, "w") as f:
        for snp in snp_info:
            print >> f, snp
    gene_filenames.append(snp_file_path)

# Write results in a single txt file for all genes
snp_file_path = os.path.join(OUTPUT_DIR, 'all_SNPs.snp')
with open(snp_file_path, 'w') as outfile:
    for fname in gene_filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)