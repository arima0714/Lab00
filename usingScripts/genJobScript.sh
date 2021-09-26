#!/bin/bash

# 使い方：`bin/usingScripts/`で実行する。
# 引数1：問題サイズ（アルファベット1文字）
# 引数2：ベースとなるジョブスクリプトのパス
# 実行される処理：ジョブスクリプトを作成する。

numOfCores=(2 4 8 16 32 64 128 256)

programSize=$1
jobScriptBaseFileName=$2
jobScriptBaseDir="./"

for numOfCore in "${numOfCores[@]}"
do
    # 各コア数ごとにジョブスクリプトを作成する
    jobScriptFileName=$jobScriptBaseFileName$numOfCore
    # 保存するディレクトリを保持する変数
    saveDir="./$programSize/$numOfCore/"
    jobScriptPath=$saveDir$jobScriptFileName
    cp $jobScriptBaseDir$jobScriptBaseFileName $jobScriptPath
    # <executeShellScript>
    executeShellScript="execBenchmarkExcludeBTSPonF.sh"
    # 各コア数ごとに異なる処理を行う
    if [ 2 -eq "$numOfCore" ]; then
        # 32コア
        echo "numOfCore=$numOfCore(2)"
        # <executeEnvironment>
        executeEnvironment="q_core"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=1
        # <executingHour>
        executingHour=10
        # <numOfCore>
        numOfCore=2
        # <processPerNode>
        processPerNode=2
    if [ 4 -eq "$numOfCore" ]; then
        # 32コア
        echo "numOfCore=$numOfCore(4)"
        # <executeEnvironment>
        executeEnvironment="q_core"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=1
        # <executingHour>
        executingHour=6
        # <numOfCore>
        numOfCore=4
        # <processPerNode>
        processPerNode=4
    if [ 8 -eq "$numOfCore" ]; then
        # 32コア
        echo "numOfCore=$numOfCore(8)"
        # <executeEnvironment>
        executeEnvironment="q_core"
        # <numOfExecuteEnvironment>
        numOfExecuteEnvironment=2
        # <executingHour>
        executingHour=4
        # <numOfCore>
        numOfCore=8
        # <processPerNode>
        processPerNode=4
    elif [ 32 -eq "$numOfCore" ]; then
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
        executingHour=23
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
        executingHour=16
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
        executingHour=10
        # <numOfCore>
        numOfCore=256
        # <processPerNode>
        processPerNode=28
    else
        echo "There is an inappropriate element."
    fi
    # <executeEnvironment>
    sed -i -e "s/<executeEnvironment>/$executeEnvironment/g" $jobScriptPath
    # <numOfExecuteEnvironment>
    sed -i -e "s/<numOfExecuteEnvironment>/$numOfExecuteEnvironment/g" $jobScriptPath
    # <executingHour>
    sed -i -e "s/<executingHour>/$executingHour/g" $jobScriptPath
    # <executeShellScript>
    sed -i -e "s/<executeShellScript>/$executeShellScript/g" $jobScriptPath
    # <numOfCore>
    sed -i -e "s/<numOfCore>/$numOfCore/g" $jobScriptPath
    # <processPerNode>
    sed -i -e "s/<processPerNode>/$processPerNode/g" $jobScriptPath

done
