#!/bin/bash

# 処理内容：一気に全種類の実行ファイルをコンパイルするためのコマンド列を出力するスクリプト
# 使用方法：`NPB3.4-MPI/` で実行する。引数は無し。

benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
classes=("S" "W" "A" "B" "C" "D" "E" "F")

for benchmark in "${benchmarks[@]}"
do
	for class in "${classes[@]}"
	do
		if [ "${benchmark}" = "is" ] && [ "${class}" = "F" ]; then
			echo "echo \"${benchmark}に${class}はありません\""
		elif [ "${benchmark}" = "dt" ] && [ "${class}" = "E"  -o "${class}" = "F" ]; then
			echo "echo \"${benchmark}には${class}はありません\""
		else
			echo "make ${benchmark} CLASS=${class}"
		fi
	done
done
