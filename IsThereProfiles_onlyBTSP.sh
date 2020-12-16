# まだプロファイルを取得できていないものを列挙するスクリプト

# 基本的な配列・変数の宣言
benchmarks=("bt" "sp")
bin_dir=${PWD}"/"
classes=("S" "W" "A" "B" "C" "D" "E" "F")
classes=("A" "B" "C" "D")
processes=("1" "4" "9" "16" "25" "36" "49" "64" "81" "100" "121" "144" "169" "196" "225")

for benchmark in ${benchmarks[@]}
do
	for class in ${classes[@]}
	do
			# プロセス数ごとにプロファイルを取得するためのループ
			for process in ${processes[@]}
			do
				# pprofのサマリ(=プロファイル)を保存する際のファイル名
				pprof_filename=${bin_dir}"txt_files/pprof_${benchmark}${class}${process}.txt"
				# プロファイルが存在しなければ、プロファイル名を出力
				if [ ! -e "${pprof_filename}" ]; then
					echo "${pprof_filename}"
				fi
			done
	done
done

exit

