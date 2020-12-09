# まだプロファイルを取得できていないものを列挙するスクリプト

# 基本的な配列・変数の宣言
benchmarks=("bt" "sp")
bin_dir=${PWD}"/"
classes=("S" "W" "A" "B" "C" "D" "E" "F")
classes=("A" "B" "C" "D")
processes=("25")

for benchmark in ${benchmarks[@]}
do
	for class in ${classes[@]}
	do
		# BenchMarkNameとは実行ファイル名のこと
		BenchMarkName=${benchmark}.${class}."x"
		# 実行ファイルが実際にある時のみ下記の処理を行う条件分岐
		if [ -e ${bin_dir}${BenchMarkName} ]; then
			# プロセス数ごとにプロファイルを取得するためのループ
			for process in ${processes[@]}
			do
				# pprofのサマリ(=プロファイル)を保存する際のファイル名
				pprof_filename=${bin_dir}"pprof_${benchmark}${class}${process}.txt"
				# プロファイルが存在しなければ、プロファイル名を出力
				if [ ! -e "${pprof_filename}" ]; then
					echo "${pprof_filename}"
				fi
			done
		fi
	done
done

exit

