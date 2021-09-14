#!/bin/bash

# 関数：exec_pprof_s()
# 処理内容：カレントディレクトリに存在する実験データから、`pprof -s`コマンドで集計したテキストファイル`<pprof_s.txt>`を出力する

function exec_pprof_s() {
	echo "$PWD"
}

# 関数：convert_pprof2csv()
# 処理内容：カレントディレクトリに存在する`exec_pprof_s()`の実行結果をCSVに変換するスクリプト

function convert_pprof2csv() {
	echo "$PWD"
}

# 使用方法：`./bin/`で実行する。引数は下記の通り。
# 引数１：問題サイズ（E, F）
# 引数２：コア数（32, 64, 128, 256）
# 処理内容：集めたpprof_*.txtに該当するファイルをCSV形式(pprof_*.csv)に変換するスクリプト

programSize=$1
numOfCore=$2
benchmarkNames=("cg" "dt" "ep" "ft" "is" "lu" "mg")

for benchmarkName in "${benchmarkNames[@]}"
do
	dirPath="./$programSize/$numOfCore/$benchmarkName/"
	echo "dirPath=""$dirPath"
	exec_pprof_s
	convert_pprof2csv
done
