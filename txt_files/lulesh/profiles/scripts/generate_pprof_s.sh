#!/bin/bash

process=$1
iteration=$2
size=$3

baseDir="$HOME/LULESH/build/profiles/"

targetDir="$baseDir/p$process/i$iteration/s$size"
fileName="pprof_s_p"$process"i"$iteration"s"$size

echo "targetPath="$targetDir"/"$fileName

cd $targetDir && pprof -s > $fileName

