#!/bin/bash

# 使い方：`bin/<コア数>/<問題サイズ>/<ベンチマーク名>`のような構成のディレクトリが既にある状態で`bin/`で実行する
# 引数：問題サイズ（アルファベット1文字）
# 実行される処理：`bin/<コア数>/<問題サイズ(E)>/<ベンチマーク名>/<バイナリ>`の様にバイナリを設置する

numOfCores=(2 4 8 16 32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=$1

for numOfCore in "${numOfCores[@]}"
do
    for benchmarkName in "${benchmarkNames[@]}"
    do
        dirName=./$problemSize/$numOfCore/$benchmarkName/
        binName=$benchmarkName.$problemSize.x
        cp $binName $dirName
    done
done
