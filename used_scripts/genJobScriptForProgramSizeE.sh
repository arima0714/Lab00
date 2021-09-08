#!/bin/bash

# 使い方：`bin/`で実行する。
# 実行される処理：ジョブスクリプトを作成する。

numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=E

# <executeEnvironment>
# <numOfExecuteEnvironment>
# <executingHour>
# <executeShellScript>
# <numOfCore>
# <processPerNode>

for numOfCore in "${numOfCores[@]}"
do
    dirName=./$problemSize/$numOfCore/$benchmarkName/
    binName=$benchmarkName.$problemSize.x
    # 各コア数ごとに異なる処理を行う
done
