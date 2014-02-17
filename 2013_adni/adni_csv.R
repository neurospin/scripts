require(ADNIMERGE)
OUTPUT_DIR="/neurospin/brainomics/2013_adni_preprocessing/clinic"
OUTPUT_PATH=file.path(OUTPUT_DIR, "adnimerge.csv")
write.csv2(adnimerge, OUTPUT_PATH)