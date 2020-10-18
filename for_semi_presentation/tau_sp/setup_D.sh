classes=("A" "B" "C" "D")
benchamark_name=("sp" "bt")

# for benchmark in "${benchamark_name[@]}"
# do 
# 	echo "$benchmark"
# done
# for v in "${classes[@]}"
# do
#     echo "$v"
# done

processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
for process in "${processes[@]}"
do
# ディレクトリ作成
	dir_name="x${process}_D"
	cp -rf ./template ${dir_name}
# ディレクトリ移動
	cd ${dir_name}
# JobScript編集
## 5,6行目を1,2,4のいずれかにする
### if process == 1
### elseif process == 2
### else process == 4
	if [ $process -eq 1 ]; then
		sed -i "s/ProcessPerNode/1/g" JobScript
	elif [ $process -eq 2 ]; then
		sed -i "s/ProcessPerNode/2/g" JobScript
	else
		sed -i "s/ProcessPerNode/4/g" JobScript
	fi
## 7,8行目を実行プロセス数にする
	sed -i "s/AllProcess/${process}/g" JobScript
## 10,11行目を資源数にする
	tmpNumResource=`expr ${process} \/ 4`	
### if process < 4
### else 
	if [ $process -lt 4 ]; then
		sed -i "s/NumResource/1/g" JobScript
	else
		sed -i " s/NumResource/${tmpNumResource}/g" JobScript
	fi
# ジョブ投入
###	echo "qsub -g tgh-20IAN JobScript"
# ディレクトリ移動
	cd ..
done

