import streamlit as st
import subprocess
import pandas as pd


@st.cache
def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
    subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])


# ライブラリノートを実行可能な形式に変換
gen_lib()
# 変換されたライブラリノートをimport
import libLab00 as lib

dimension = st.sidebar.selectbox("プロットする次元", ("２次元", "３次元"))

if dimension == "２次元":  # 2次元グラフの描画

    st.markdown("# ２次元グラフのプロット")

elif dimension == "３次元":  # 3次元グラフの描画

    st.markdown("# ３次元グラフのプロット")

    # 生データの取得
    benchmarkName = ["cg"]
    classes = ["A", "B", "C", "D"]
    processes = [2, 4, 8, 16, 32, 64, 128, 256]
    csvDirPath = "../csv_files/"
    rawDataDF = lib.returnCollectedExistingData(benchmarkNames=benchmarkName, classes=classes, processes=processes, csvDirPath=csvDirPath)
    # 問題サイズを数値化
    programSize = rawDataDF['benchmarkClass'].tolist()
    programSizeInNum = lib.convertBenchmarkClasses_problemSizeInNPB(inputList=programSize)
    # プロット用のDFを作成

    # プロット

else:
    pass
