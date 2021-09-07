numOfCores=(32 64 128 256)
benchmarkNames=(bt sp lu is ep cg mg ft)
problemSize=E
for numOfCore in "${numOfCores[@]}"
do
    for benchmarkName in "${benchmarkNames[@]}"
    do
        dirName=./$problemSize/$numOfCore/$benchmarkName/
        binName=$benchmarkName.$problemSize.x
        cp $binName $dirName
    done
done

# 使い方：`bin/<コア数>/<問題サイズ(E)>/<ベンチマーク名>`のような構成のディレクトリが既にあるうえで`bin/`で実行する
# 実行される処理：`bin/<コア数>/<問題サイズ(E)>/<ベンチマーク名>/<バイナリ>`のようにバイナリを設置する
