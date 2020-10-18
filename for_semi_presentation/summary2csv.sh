classes=("A" "B" "C" "D")

bench_marks=("is" "ep" "cg" "mg" "ft" "bt" "sp" "lu")

processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")

for bench_mark in "${bench_marks[@]}"
do
	for class in "${classes[@]}"
	do
		for process in "${processes[@]}"
		do
			base_dir_name="tau_${bench_mark}/x${process}_${class}/"
			file_name="${base_dir_name}function_summary_mean.txt"
			rm "${file_name}.csv"
			echo "${file_name}"
			cat ${file_name} | sed '5d' | sed '4d' | sed '2d' | sed '1d' > "${base_dir_name}function_summry_mean.csv"
		done
	done
done


