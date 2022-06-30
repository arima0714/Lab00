#!/bin/bash

# shellcheck source=conf_v3.txt
source conf_pprof_csv.txt

scriptsDir="$HOME/LULESH/build/profiles/scripts/"

for process in "${processes[@]}"
do
    for iteration in "${iterations[@]}"
    do
        for size in "${sizes[@]}"
        do
            $scriptsDir/generate_pprof_s.sh $process $iteration $size
            $scriptsDir/generate_csv_from_pprof_s.sh $process $iteration $size
        done
    done
done
