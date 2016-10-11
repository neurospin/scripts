scripts
=======

These are the scripts used to manage the Enigma pipeline on the folders referred to as follow:

Creteil, Grenoble_Philipps, mannheim, pittsburgh, udine1.5T, udine3T

This is for the information found in /neurospin/cati/bipolar/sandbox/niftii . All processes will remain within /neurospin/cati/bipolar

Main Process: organise.sh
============

The main file to run is organise.sh. 

It is a bash file that takes as inputs the folder in which the data is currently located, and the folder 
in which we want to reorganize said data.

For example, in our case if the data for different sites, such as creteil is located at neurospin/cati/bipolar/sandbox/niftii/creteil, and we want them to be reorganized somewhere in the current folder ( /neurospin/cati/bipolar ), the inputs would be:

sandbox/niftii/	./


There are four main components to the organise.sh script.
(Those can be turned on or off inside the code parameters, ex: if {tosort=true} the sort will launch first, otherwise it will be skipped.)

It should be noted that these components are assumed to be launched in a certain order, and that the "Sort the data" must be done first, as all other components use the same organisational data as the one created by "Sort the data"

Sort the data
-------

First, it will take the original data (for example, creteil)
and sort it in the format:

sorted/dwi/creteil/{files}	for the diffusion weighted images

sorted/t1/creteil/{files}	for the T1 images

This will be done differently for all sites, in such a manner as to have the file names corresponding to the NAME of the subjects found in BASE_NEW (can be found in /neurospin/cati/bipolar)

Eddy_correct
-------

Then, it will run the eddy correction on the dwi files found in sorted/dwi.

Here, it will refer to parameddy.txt . If the folder is named in parameddy.txt followed by "true",
it will do the eddy correction. Otherwise, it will proceed to the next folder.

Bet
-------

Here, it will run the Bet on all dwi found in the different folders under sorted/.
This time, it will refer to parambet.txt for the dwi
: the txt files will specify the folders to bet followed by
the value of the threshold of the Bet

Dtifit
-------

This part has been discontinued due to its lack of dealing properly with susceptibility artefacts
This will run the Dtifit on the sorted folder. Possible additions to follow.


The Mannheim naming problem: relabels.sh
========

One of the problems of mannheim is that, originally, many of the files were renamed when compared to their
original excel reference. Thankfully, Samuel provided a file specifying which file was turned into which, allowing us to relabel
the mannheim files according to their original name.

This script, relabels.sh, takes as input the location of the folder in which mannheim is located, and uses the file "labels.txt"
(hard-coded). Then, it will rename all the folders and files corresponding to one name and convert it to the other name.

It can thus be used on the original data or the post-sort data.


Referring to the excel file: checkfilesexist.sh and checktxtref.sh
========


Those bash files are used to check the correspondance between the files we possess and the files mentioned by the excel.

They take as input the folder

checkfilesexist.sh will try to find all the files mentioned by name in a txt file (copy/paste from the excel)
For example, creteillist.txt will have the identity names of the files mentioned in excel as belonging to creteil.

So

bash checkfilesexist.sh creteil 

will check creteil and refer to creteillist.txt .

It will then mention the files it did not find through an echo and finish with saying the number of files found total.


checktxtref.sh will instead look at the files inside the folder and see if they are mentioned in the txt file. It will then 
mention the files for which it did not find a reference in the txt and the number of such files missing all told.






