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
			cd ${dir_name}
			file_name="profile.0.0.0"
			if [ ! -s ${file_name} ]; then
				rm JobScript.*
				rm function_summary_mean.txt
				rm pprof_result
			fi
			cd /home/9/20IA1328/NPB3.4.1/NPB3.4-MPI/bin
		done
	done
done

