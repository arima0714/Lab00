# ジョブスクリプトを作成し、ベンチマークを実行するスクリプト

# 基本的な配列・変数の宣言
benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
classes=("S" "W" "A" "B" "C" "D" "E" "F")
processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
bin_dir=${PWD}"/"

# 今回はCG , LU , SPのみで行う
benchmarks=("cg" "sp" "lu")
processes=("1" "2" "4" "8")

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
				# sedコマンドでジョブスクリプトで変更するものを実際に変更する
				echo "#!/bin/sh" > JobScript
				echo "# カレントディレクトリでジョブを実行する"
				echo "#$ -cwd" >> JobScript
				echo "# 実行ファイル名" >> JobScript
				echo "BenchmarkFileName="${BenchMarkName} >> JobScript
				echo "# ノード毎プロセス数"${ProcessPerNode} >> JobScript
				echo "A="${ProcessPerNode} >> JobScript
				echo "# 合計プロセス数"${AllProcess} >> JobScript
				echo "B="${AllProcess} >> JobScript
				echo "# 資源タイプC4 "${NumResource}"ノードを使用" >> JobScript
				echo "#$ -l q_core="${NumResource} >> JobScript
				echo "# 実行時間を指定" >> JobScript
				echo "#$ -l h_rt=0:09:30" >> JobScript
				echo "# moduleコマンドの初期化" >> JobScript
				echo ". /etc/profile.d/modules.sh" >> JobScript
				echo "# CUDA環境の読み込み" >> JobScript
				echo "module load cuda" >> JobScript
				echo "# OpenMPI環境の読み込み" >> JobScript
				echo "module load openmpi" >> JobScript
				echo "# PAPI環境の読み込み" >> JobScript
				echo "module load papi" >> JobScript
				echo "# gcc環境の読み込み" >> JobScript
				echo "module load gcc" >> JobScript
				echo "export PATH=\"\$PATH:/home/9/20IA1328/tau-2.29/x86_64/bin\"" >> JobScript
				echo "export TAU_MAKEFILE=/home/9/20IA1328/tau-2.29/x86_64/lib/Makefile.tau-papi-mpi-pdt" >> JobScript
				echo "export TAU_OPTIONS=-optRevert" >> JobScript
				echo "export PATH=\"\$PATH:/home/9/20IA1328/pdtoolkit-3.25.1/x86_64//bin\"" >> JobScript
				echo "export TAU_THROTTLE=0" >> JobScript
				echo "# 実行" >> JobScript
				echo "# ノードあたりAプロセスMPI全B プロセスを使用" >> JobScript
				echo "mpirun -npernode \$A -n \$B -x LD_LIBRARY_PATH \${BenchmarkFileName}" >> JobScript

				# 既にプロファイルが存在しなければジョブを投入する
				if [ ! -e "pprof_${benchmark}${class}${process}.txt" ]; then
					rm profile.*
					if [ ${process} -lt 8 ]; then
						qsub JobScript
					else
						qsub -g tgh-20IAN JobScript
					fi
					echo ${BenchMarkName}"をキューに投入しました"
					sleep 10m
					pprof -s > pprof_"${benchmark}${class}${process}".txt
				fi
			done
		fi
	done
done

