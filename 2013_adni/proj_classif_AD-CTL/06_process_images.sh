cd /neurospin/brainomics/2013_adni/proj_classif/tv/split_vizu/images

ls *.png|while read input; do
convert  $input -trim /tmp/toto.png;
convert  /tmp/toto.png -transparent white $input;
done

