#!/bin/bash
numOfCore=$1
numOfCoreInNode=$2
benchmarkNames=(lu is ep cg mg ft)
problemSize=E

baseDir=$PWD
echo $baseDir
echo 'numOfCore='$numOfCore

export PATH=\"\$PATH:/home/9/20IA1328/tau-2.29/x86_64/bin\"
export TAU_MAKEFILE=/home/9/20IA1328/tau-2.29/x86_64/lib/Makefile.tau-papi-mpi-pdt
export TAU_OPTIONS=-optRevert
export PATH=\"\$PATH:/home/9/20IA1328/pdtoolkit-3.25.1/x86_64//bin\"
export TAU_THROTTLE=0
# 実行
# ノードあたりAプロセスMPI全B プロセスを使用
mpirun -npernode \$A -n \$B -x LD_LIBRARY_PATH \${BenchmarkFileName}

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
