#!/bin/bash

# 一気に全種類の実行ファイルをコンパイルするためのコマンド列を出力するスクリプト

benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
classes=("S" "W" "A" "B" "C" "D" "E" "F")

for benchmark in "${benchmarks[@]}"
do
	for class in "${classes[@]}"
	do
		if [ "${benchmark}" = "is" -a "${class}" = "F" ]; then
			echo "echo \"${benchmark}に${class}はありません\""
		elif [ "${benchmark}" = "dt" ] && [ "${class}" = "E" -o "${class}" = "F" ]; then
			echo "echo \"${benchmark}には${class}はありません\""
		else
			echo "make ${benchmark} CLASS=${class}"
		fi
	done
done
