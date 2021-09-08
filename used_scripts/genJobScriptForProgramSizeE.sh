#!/bin/bash

# 使い方：`bin/`で実行する。
# 実行される処理：ジョブスクリプトを作成する。

numOfCores=(32 64 128 256)

# <executeEnvironment>
# <numOfExecuteEnvironment>
# <executingHour>
# <executeShellScript>
# <numOfCore>
# <processPerNode>

for numOfCore in "${numOfCores[@]}"
do
    # 各コア数ごとに異なる処理を行う
    if [ 32 -eq "$numOfCore" ]; then
        # 32コア
        echo "numOfCore=$numOfCore(32)"
    elif [ 64 -eq "$numOfCore" ]; then
        # 64コア
        echo "numOfCore=$numOfCore(64)"
    elif [ 128 -eq "$numOfCore" ]; then
        # 128コア
        echo "numOfCore=$numOfCore(128)"
    elif [ 256 -eq "$numOfCore" ]; then
        # 256コア
        echo "numOfCore=$numOfCore(256)"
    else
        echo "There is an inappropriate element."
    fi
done
