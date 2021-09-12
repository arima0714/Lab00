#!/bin/bash

# 使い方：ジョブスクリプト内で実行する。実行時引き数は2つあり下記の通り。ジョブスクリプトはコア数を示すディレクトリでqsubするように実行する。
# 引数１：コア数（総プロセス数）
# 引数２：ノード当たりの実行プロセス数
# 実行される処理：引き数で渡されたコア数・ノード当たりのプロセス数に応じた環境で、ベンチマークプログラムを実行する

numOfCore=$1
A=$2
B=$numOfCore
benchmarkNames=(lu ep cg mg ft)
problemSize=F

baseDir=$PWD

export PATH="$PATH:/home/9/20IA1328/tau-2.29/x86_64/bin"
export TAU_MAKEFILE=/home/9/20IA1328/tau-2.29/x86_64/lib/Makefile.tau-papi-mpi-pdt
export TAU_OPTIONS=-optRevert
export PATH="$PATH:/home/9/20IA1328/pdtoolkit-3.25.1/x86_64//bin"
export TAU_THROTTLE=0

ls
pwd

for benchmarkName in "${benchmarkNames[@]}"
do
    # ディレクトリを実行環境用のディレクトリに移動
    dirName=./$benchmarkName
    cd $dirName || exit
    # ベンチマークバイナリを実行
    binName=$benchmarkName.$problemSize.x
    if [ -e $binName ]; then
        # 実行
        # ノードあたりAプロセスMPI全B プロセスを使用
        mpirun -npernode $A -n $B -x LD_LIBRARY_PATH ${binName}
    else
        echo "execute ""$binName"" failed"
    fi
    # 元のディレクトリに戻る
    cd $baseDir || exit
done
