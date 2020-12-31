# ジョブスクリプトを作成し、ベンチマークを実行するスクリプト

# 基本的な配列・変数の宣言
benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
benchmarks=("cg" "dt" "ep" "ft" "is" "lu" "mg")
bin_dir=${PWD}"/"
classes=("S" "W" "A" "B" "C" "D" "E" "F")
classes=( "A" "B" "C" "D")
processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
processes=("1" "2" "4")

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
				if [ `expr $process` -le 4 ]; then
					ProcessPerNode=$process
					NumResource=1
				else 
					ProcessPerNode=4
					NumResource=`expr $process / 4`
				fi
				# pprofのサマリを保存する際のファイル名
				pprof_filename=${bin_dir}"pprof_${benchmark}${class}${process}.txt"
				# 既にプロファイルが存在しなければジョブを投入する
				if [ ! -e "${pprof_filename}" ]; then
					rm profile.*
					echo " mpirun -n ${process} -x LD_LIBRARY_PATH ${BenchMarkName} && pprof -s > ${pprof_filename}"
					eval " mpirun -n ${process} -x LD_LIBRARY_PATH ${BenchMarkName}"
					eval "pprof -s > ${pprof_filename}"
				fi
			done
		fi
	done
done

exit
