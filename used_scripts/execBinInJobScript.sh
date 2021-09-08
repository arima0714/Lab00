#!/bin/bash
numOfCore=$1
benchmarkNames=(lu is ep cg mg ft)
problemSize=E

baseDir=$PWD
echo $baseDir
echo 'numOfCore='$numOfCore

for benchmarkName in "${benchmarkNames[@]}"
do
    # ディレクトリを実行環境用のディレクトリに移動
    dirName=./$problemSize/$numOfCore/$benchmarkName
    cd $dirName
    # ベンチマークバイナリを実行
    binName=$benchmarkName.$problemSize.x
    if [ -e $binName ]; then
        echo $binName
    fi
    # 元のディレクトリに戻る
    cd $baseDir
done
