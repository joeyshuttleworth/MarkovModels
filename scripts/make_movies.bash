#!/bin/bash

declare -a StringArray=("p1_p2" "p3_p4" "p5_p6" "p7_p8" )

for value in ${StringArray[@]}; do
	ffmpeg -framerate 2 -i ${value}_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p ${value}.mp4
done;

