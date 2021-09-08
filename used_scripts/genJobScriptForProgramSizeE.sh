#!/bin/bash

# 使い方：`bin/`で実行する。
# 実行される処理：ジョブスクリプトを作成する。

numOfCores=(32 64 128 256)

jobScriptBaseFileName="jobScriptForProblemSizeE"

for numOfCore in "${numOfCores[@]}"
do
    # 各コア数ごとにジョブスクリプトを作成する
    jobScriptFileName=$jobScriptBaseFileName$numOfCore
    cp $jobScriptBaseFileName $jobScriptFileName
    # <executeShellScript>

    # 各コア数ごとに異なる処理を行う
    if [ 32 -eq "$numOfCore" ]; then
        # 32コア
        echo "numOfCore=$numOfCore(32)"
        # <executeEnvironment>
        executeEnvironment="q_core"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=8
        # <executingHour>
        executingHour=23
        # <numOfCore>
        numOfCore=32
        # <processPerNode>
        processPerNode=4
    elif [ 64 -eq "$numOfCore" ]; then
        # 64コア
        echo "numOfCore=$numOfCore(64)"
        # <executeEnvironment>
        executeEnvironment="q_core"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=16
        # <executingHour>
        executingHour=14
        # <numOfCore>
        numOfCore=64
        # <processPerNode>
        processPerNode=4
    elif [ 128 -eq "$numOfCore" ]; then
        # 128コア
        echo "numOfCore=$numOfCore(128)"
        # <executeEnvironment>
        executeEnvironment="f_node"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=5
        # <executingHour>
        executingHour=8
        # <numOfCore>
        numOfCore=128
        # <processPerNode>
        processPerNode=28
    elif [ 256 -eq "$numOfCore" ]; then
        # 256コア
        echo "numOfCore=$numOfCore(256)"
        # <executeEnvironment>
        executeEnvironment="f_node"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=10
        # <executingHour>
        executingHour=5
        # <numOfCore>
        numOfCore=256
        # <processPerNode>
        processPerNode=28
    else
        echo "There is an inappropriate element."
    fi
done
