numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=E
for numOfCore in "${numOfCores[@]}"
do
    for benchmarkName in "${benchmarkNames[@]}"
    do
        dirName=./$problemSize/$numOfCore/$benchmarkName
        echo $dirName
    done
done

# 使い方：`bin/`で実行する
