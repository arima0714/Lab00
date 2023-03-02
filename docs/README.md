# 環境構築

1. トップディレクトリで `docker-compose up` を実行
2. `localhost:8787` にアクセス

* Visual Studio Code のリモート開発にも一応対応している（ふだんはVisual Studio Code で開発していたが、動作が不安定な部分もある。）。

# 利用方法

`testing.ipynb` を開く

利用方法はtesting_pub.ipynbに有るようにする。

より抽象度を下げると
1. CSVの読み込み
2. モデルの作成
3. 予測が順に行われている

具体例としてベンチマークプログラムMGを利用する場合を示す。

```
# ベンチマークプログラムMGもプロファイルをCSVにしたディレクトリのパスをリスト化
list_csvDir_mg = [
    "./csv_files/mg_1st/",
    "./csv_files/mg_2nd/",
    "./csv_files/mg_3rd/",
]


# NPBのMGの初期変数
mg_size :list[int] = [32, 64, 128, 256, 512]
mg_nit: list[int] = [4, 10, 20, 35, 50]
# NPBのMGの初期変数（学習用）
train_mg_process :list[int] = [4, 8, 16, 32, 64]
train_mg_size :list[int] = [4, 8, 16, 32, 64]
train_mg_nit :list[int] = [5, 10, 15, 20, 25]
# NPBのMGの初期変数（予測用）
test_mg_process :list[int] = [128, 256, 512]
test_mg_size :list[int] = [128, 256, 512]
test_mg_nit :list[int] = [30, 40, 50]

# MGの説明変数
expVars_mg :list[str] = ["process", "problem_size", "nit"]

# MGの学習用DF
trainDF_MG :pd.DataFrame = ret_averaged_rawDF_mg(list_process=train_mg_process, list_size=train_mg_size, list_nit=train_mg_nit, list_csvDir=list_csvDir_mg, resVar="Exclusive")
# MGの予測用DF
testDF_MG :pd.DataFrame = ret_averaged_rawDF_mg(list_process=test_mg_process, list_size=test_mg_size, list_nit=test_mg_nit, list_csvDir=list_csvDir_mg, resVar="Exclusive")

benchmarkName :str = "mg"

# 変数mg_contensはクラスオブジェクトとなっており、学習用データ、予測用データ、説明変数名のリスト、目的変数、
mg_contents = ContentsForExtraP(trainDF = trainDF_MG, testDF= testDF_MG, resVar = "Exclusive", expVars= expVars_mg ,resVarPerCall = "ExclusivePerCall", benchmarkName = benchmarkName)

# 適合度, 重み付きMAPE, MAPE, 予測精度, 関数ごとのMAPEがキー：バリュー＝文字列：データフレームに対応する形で格納
mg_dict = mg_contents.exec_all()

mg_ex_適合度 :pd.DataFrame = mg_dict["適合度"]
mg_ex_重み付きMAPE :pd.DataFrame = mg_dict["重み付きMAPE"]
mg_ex_MAPE :pd.DataFrame = mg_dict["MAPE"]
mg_ex_予測精度 :pd.DataFrame = mg_dict["予測精度"]
mg_ex_MAPE_on_functionName :pd.DataFrame = mg_dict["関数ごとのMAPE"]

# ファイルの出力
outputFilePath :str = f"./outputs/{dt_now.strftime('%Y%m%d%H%M%S')}_{benchmarkName}_ex.xlsx"
with pd.ExcelWriter(
    outputFilePath,
    engine="xlsxwriter",
    engine_kwargs={'options': {'strings_to_numbers': True}},
    ) as writer:

    for _key in mg_dict.keys():
        mg_dict[_key].to_excel(writer, sheet_name=_key)

```