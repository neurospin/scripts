require(ADNIMERGE)
OUTPUT_DIR="/neurospin/brainomics/2013_adni/clinic"
OUTPUT_PATH=file.path(OUTPUT_DIR, "adnimerge.csv")
write.csv(adnimerge, OUTPUT_PATH, row.names=FALSE)
