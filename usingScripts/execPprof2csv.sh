#!/bin/bash

# 使用方法：`./bin/`で実行する。
# 引数：実行するスクリプトのパス
# 処理内容：`pprof -s`による出力をCSV化するスクリプトをまとめて実行するスクリプト

programSizes=("E" "F")
numOfCores=("32" "64" "128" "256")

for programSize in "${programSizes[@]}"
do
    for numOfCore in "${numOfCores[@]}"
    do
        echo "programSize=$programSize, numOfCore=$numOfCore"
        $1 "$programSize" "$numOfCore"
    done
done
