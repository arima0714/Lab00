#!/bin/bash

process=$1
iteration=$2
size=$3

baseDir="$HOME/LULESH/build/profiles/"
outputDir="$HOME/LULESH/build/profiles/csv_files/"

targetDir="$baseDir/p$process/i$iteration/s$size/"
targetFileName="pprof_s_p"$process"i"$iteration"s"$size
outputFileName="pprof_s_p"$process"i"$iteration"s"$size".csv"

echo "targetPath="$targetDir"/"$targetFileName
cd $targetDir && cat $targetDir"/"$targetFileName  | sed -n '/mean/,$p' | sed -e 's/.TAU application/.TAU_application/g' | sed -e 's/char /char_/g' | sed -e 's/void /void_/g' | sed -e 's/double /double_/g' | sed -e 's/int /int_/g' | sed -e 's/Real_t /Real_t_/g' | sed -e 's/, /_/g' | sed "4,5d" | sed "1,2d" | awk -v OFS=, '{print $7, $4}' > $outputDir"/"$outputFileName

