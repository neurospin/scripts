#!/bin/sh


#pretty simple code, just launch it from within the folder with the anatomist images you want to make into a movie, or cd into the right folder and launch these commands from ubuntu terminal.

for image in ./*; do mv ${image} ${image}.jpeg ; done
mencoder "mf://*.jpeg" -mf fps=5 -o output.avi -ovc lavc -lavcopts vcodec=mpeg4
