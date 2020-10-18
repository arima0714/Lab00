# CSVファイルを指定ディレクトリに集めるスクリプト
classes=("A" "B" "C" "D")

bench_marks=("is" "ep" "cg" "mg" "ft" "bt" "sp" "lu")

processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")

for bench_mark in "${bench_marks[@]}"
do
	for class in "${classes[@]}"
	do
		for process in "${processes[@]}"
		do
			dir_name="tau_${bench_mark}/x${process}_${class}/"
			file_name="${dir_name}${bench_mark}_x${process}_${class}.csv"
			echo ${file_name}
			cp ${file_name} /home/9/20IA1328/NPB3.4.1/NPB3.4-MPI/bin/CSVs/
		done
	done
done

