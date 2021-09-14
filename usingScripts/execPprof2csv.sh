#!/bin/bash

programSizes=("E" "F")
numOfCores=("32" "64" "128" "256")

for programSize in "${programSizes[@]}"
do
    for numOfCore in "${numOfCores[@]}"
    do
        echo "programSize=$programSize, numOfCore=$numOfCore"
    done
done
