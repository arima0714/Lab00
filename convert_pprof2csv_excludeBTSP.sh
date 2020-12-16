# 集めたpprof_*.txtに該当するファイルをCSV形式(pprof_*.csv)に変換するスクリプト

# cat pprof_spC8.txt | sed -n '/mean/,$p'| sed -e 's/char /char_/g' | sed -e 's/void /void_/g' | sed -e 's/double /double_/g' | sed -e 's/int /int_/g' | sed -e 's/.TAU application/.TAU_application/g' | sed -e 's/, /_/g' | sed "4,5d" | sed "1,2d" | awk -v OFS=, '{print $7, $4}'

# 基本的な配列・変数の宣言
benchmarks=("bt" "cg" "dt" "ep" "ft" "is" "lu" "mg" "sp")
classes=("S" "W" "A" "B" "C" "D" "E" "F")
processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
bin_dir=${PWD}"/"

txt_files_dir="${bin_dir}txt_files/"
csv_files_dir="${bin_dir}csv_files/"

# profile.* は不要なので削除
# rm profile.*

for benchmark in ${benchmarks[@]}
do
	for class in ${classes[@]}
	do
		# プロセス数ごとにジョブを投入するので、そのためのループ
		for process in ${processes[@]}
		do
			# ジョブスクリプト用のプロセス数などに関する記述
			AllProcess=$process
			# pprof_*.txt のファイル名を格納する
			pprof_s_FileName="${txt_files_dir}pprof_${benchmark}${class}${process}.txt"
			csv_FileName="${csv_files_dir}pprof_${benchmark}${class}${process}.csv"
			if [ `expr $process` -le 4 ]; then
				ProcessPerNode=$process
				NumResource=1
			else 
				ProcessPerNode=4
				NumResource=`expr $process / 4`
			fi
			# pprof_*.txt が存在するときに行う処理 
			if [ -e ${pprof_s_FileName} ]; then
				echo ${csv_FileName}
				cat ${pprof_s_FileName} | sed -n '/mean/,$p'| sed -e 's/char /char_/g' | sed -e 's/void /void_/g' | sed -e 's/double /double_/g' | sed -e 's/int /int_/g' | sed -e 's/.TAU application/.TAU_application/g' | sed -e 's/, /_/g' | sed "4,5d" | sed "1,2d" | awk -v OFS=, '{print $7, $4}' > "${csv_FileName}"
			fi
		done
	done
done

