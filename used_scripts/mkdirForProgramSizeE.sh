numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
for numOfCore in "${numOfCores[@]}"
do
    for benchmarkName in "${benchmarkNames[@]}"
    do
        echo "numOfCore="$numOfCore
        echo "benchmarkNames="$benchmarkName
    done
done