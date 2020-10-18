classes=("A" "B" "C" "D")

processes=("1" "2" "4" "8" "16" "32" "64" "128" "256")
for class in "${classes[@]}"
do
	for process in "${processes[@]}"
	do
		# ディレクトリ作成
		dir_name="x${process}_${class}"
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
		## クラスを設定
		sed -i " s/CLASS/${class}/g" JobScript
		# ディレクトリ移動
		cd ..
	done
done

