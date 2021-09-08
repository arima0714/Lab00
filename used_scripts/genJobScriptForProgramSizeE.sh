#!/bin/bash

# 使い方：
# 実行される処理：`bin/<コア数>/<問題サイズ(E)>/<ベンチマーク名>/<バイナリ>`のようにバイナリを設置する

numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=E

<executeEnvironment>
<numOfExecuteEnvironment>
<executingHour>
<executeShellScript>
<numOfCore>
<processPerNode>

for numOfCore in "${numOfCores[@]}"
do
    dirName=./$problemSize/$numOfCore/$benchmarkName/
    binName=$benchmarkName.$problemSize.x
    cp $binName $dirName
done
