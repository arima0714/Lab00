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
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("# ３次元グラフのプロット")

    # 生データの取得
    benchmarkName = [st.selectbox(options=["cg", "ep", "ft", "is", "lu", "mg"], label="ベンチマーク名")]
    classes = ["A", "B", "C", "D"]
    processes = [2, 4, 8, 16, 32, 64, 128, 256]
    csvDirPath = "../csv_files/"
    rawDataDF = lib.returnCollectedExistingData(benchmarkNames=benchmarkName, classes=classes, processes=processes, csvDirPath=csvDirPath)
    # 問題サイズを数値化
    programSize = rawDataDF['benchmarkClass'].tolist()
    programSizeInNum = lib.convertBenchmarkClasses_problemSizeInNPB(inputList=programSize)
    rawDataDF["benchmarkClassInNum"] = programSizeInNum
    # プロット用のDFを作成
    functionNames = sorted(list(set(rawDataDF['functionName'].tolist())))
    functionName = st.selectbox(options=functionNames, label='関数名')
    DFperFunctionName = rawDataDF[rawDataDF["functionName"]==functionName]
    numCore = DFperFunctionName["process"].tolist()
    programSize = DFperFunctionName["benchmarkClassInNum"].tolist()
    functionCallCount = DFperFunctionName["functionCallNum"].tolist()
    DFtoPlot = pd.DataFrame({"問題サイズ":programSize, "コア数":numCore, "関数コール回数":functionCallCount})
    # プロット
    enableLogX = st.checkbox(label="問題サイズの軸の対数化")
    enableLogY = st.checkbox(label='コア数の軸の対数化')
    enableLogZ = st.checkbox(label='関数コール回数の軸の対数化')
    fig = px.scatter_3d(DFtoPlot, x='問題サイズ', y='コア数', z='関数コール回数', log_x=enableLogX, log_y=enableLogY, log_z=enableLogZ)
    fig.update_layout(width=700, height=700)
    st.write(fig)

else:
    pass
