# 集めたpprof_*.txtに該当するファイルをCSV形式(pprof_*.csv)に変換するスクリプト

# cat pprof_spC8.txt | sed -n '/mean/,$p'| sed -e 's/char /char_/g' | sed -e 's/void /void_/g' | sed -e 's/double /double_/g' | sed -e 's/int /int_/g' | sed -e 's/.TAU application/.TAU_application/g' | sed -e 's/, /_/g' | sed "4,5d" | sed "1,2d" | awk -v OFS=, '{print $7, $4}'

# 基本的な配列・変数の宣言
benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
classes=("S" "W" "A" "B" "C" "D" "E" "F")
processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
bin_dir=${PWD}"/"

# profile.* は不要なので削除
rm profile.*

for benchmark in ${benchmarks[@]}
do
	for class in ${classes[@]}
	do
		# BenchMarkNameとは実行ファイル名のこと
		BenchMarkName=${benchmark}.${class}."x"
		# 実行ファイルが実際にある時のみ下記の処理を行う条件分岐
		if [ -e ${bin_dir}${BenchMarkName} ]; then
			# プロセス数ごとにジョブを投入するので、そのためのループ
			for process in ${processes[@]}
			do
				# JobScriptファイルをベンチマークを実行するたびに作成する
				AllProcess=$process
				if [ `expr $process` -le 4 ]; then
					ProcessPerNode=$process
					NumResource=1
				else 
					ProcessPerNode=4
					NumResource=`expr $process / 4`
				fi

				# 既にプロファイルが存在しなければジョブを投入する
				if [ -e "pprof_${benchmark}${class}${process}.txt" ]; then
					echo pprof_"${benchmark}${class}${process}".txt
				else
					echo "該当するファイルはありませんでした"
				fi
			done
		fi
	done
done



