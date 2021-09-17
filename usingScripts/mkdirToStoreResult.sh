#!/bin/bash

# 使い方：`bin/`で実行する
# 引数：問題サイズ（アルファベット1文字）
# 実行される処理：`bin/<コア数>/<問題サイズ>/<ベンチマーク名>`の様な構成のディレクトリを作成する

numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=$1
for numOfCore in "${numOfCores[@]}"
do
    for benchmarkName in "${benchmarkNames[@]}"
    do
        dirName=./$problemSize/$numOfCore/$benchmarkName
        mkdir -p $dirName
    done
done
