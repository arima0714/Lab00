from numpy import minimum
import statistics
import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache
def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
    subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])


# ライブラリノートを実行可能な形式に変換
gen_lib()
# 変換されたライブラリノートをimport
import libLab00 as lib

dimension = st.sidebar.selectbox("プロットする次元", ("２次元", "３次元"))

# 生データの取得
benchmarkName = [
    st.selectbox(options=["cg", "ep", "ft", "is", "lu", "mg"], label="ベンチマーク名")
]
classes = ["A", "B", "C", "D"]
processes = [2, 4, 8, 16, 32, 64, 128, 256]
csvDirPath = "../csv_files/"
rawDataDF = lib.returnCollectedExistingData(
    benchmarkNames=benchmarkName,
    classes=classes,
    processes=processes,
    csvDirPath=csvDirPath,
)
# 問題サイズを数値化
programSize = rawDataDF["benchmarkClass"].tolist()
programSizeInNum = lib.convertBenchmarkClasses_problemSizeInNPB(inputList=programSize)
rawDataDF["benchmarkClassInNum"] = programSizeInNum
# プロット用のDFを作成
functionNames = sorted(list(set(rawDataDF["functionName"].tolist())))
functionName = st.selectbox(options=functionNames, label="関数名")
DFperFunctionName = rawDataDF[rawDataDF["functionName"] == functionName]
numCore = DFperFunctionName["process"].tolist()
programSize = DFperFunctionName["benchmarkClass"].tolist()
programSizeInNum = DFperFunctionName["benchmarkClassInNum"].tolist()
functionCallCount = DFperFunctionName["functionCallNum"].tolist()
DFtoPlot = pd.DataFrame(
    {
        "問題サイズ": programSizeInNum,
        "コア数": numCore,
        "関数コール回数": functionCallCount,
        "問題サイズ（文字）": programSize,
    }
)
# プロット

if dimension == "２次元":  # 2次元グラフの描画

    fixedTarget = st.selectbox("コア数と問題サイズのどちらを固定するか？", ["コア数", "問題サイズ"])
    notFixed = "コア数" if fixedTarget == "問題サイズ" else "問題サイズ"
    fixedVar = None

    if fixedTarget == "問題サイズ":
        # 問題サイズを固定する場合は、問題サイズ(文字)->問題サイズ(数値)->実際に固定
        choiceList = sorted(list(set(DFtoPlot["問題サイズ（文字）"].tolist())))
        pass
    elif fixedTarget == "コア数":
        # コア数を固定する場合は、コア数の数値で固定
        choiceList = sorted(list(set(DFtoPlot["コア数"].tolist())))
        pass
    else:
        pass

    enableLogX = st.checkbox(label="X軸（横軸）の対数化")
    enableLogY = st.checkbox(label="Y軸（縦軸）の対数化")

    choosedVar = st.selectbox("固定する値", choiceList)

    DFtoPlotIn2D = DFtoPlot[DFtoPlot[fixedTarget] == choosedVar]

    DFtoPlotIn2D

    fig = px.scatter(
        DFtoPlotIn2D, x=notFixed, y="関数コール回数", log_x=enableLogX, log_y=enableLogY
    )

    st.markdown("# ２次元グラフのプロット")

elif dimension == "３次元":  # 3次元グラフの描画

    st.markdown("# ３次元グラフのプロット")

    enableLogX = st.checkbox(label="問題サイズの軸の対数化")
    enableLogY = st.checkbox(label="コア数の軸の対数化")
    enableLogZ = st.checkbox(label="関数コール回数の軸の対数化")

    plotType = st.selectbox(options=["scatter", "mesh"], label="プロットするタイプの選択")
    if plotType == "scatter":
        fig = px.scatter_3d(
            DFtoPlot,
            x="問題サイズ",
            y="コア数",
            z="関数コール回数",
            color=DFtoPlot["問題サイズ（文字）"].tolist(),
            log_x=enableLogX,
            log_y=enableLogY,
            log_z=enableLogZ,
        )
    elif plotType == "mesh":
        x = DFtoPlot["問題サイズ"].tolist()
        y = DFtoPlot["コア数"].tolist()
        z = DFtoPlot["関数コール回数"].tolist()
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    opacity=0.70,
                    showscale=True,
                )
            ],
            layout={}
        )

        if enableLogX:
            fig.update_scenes(xaxis_type="log")
        if enableLogY:
            fig.update_scenes(yaxis_type="log")
        if enableLogZ:
            fig.update_scenes(zaxis_type="log")

    fig.update_layout(
        width=800,
        height=800,
        autosize=False,
        scene={"xaxis_title": "問題サイズ", "yaxis_title": "コア数", "zaxis_title": "関数コール回数"},
    )

else:
    pass

st.write(fig)
