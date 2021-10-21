import matplotlib.pyplot as plt
import streamlit as st
import subprocess
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def app():
    @st.cache
    def gen_lib():
        subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
        subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])

    # ライブラリノートを実行可能な形式に変換
    gen_lib()
    # 変換されたライブラリノートをimport
    import libLab00 as lib

    st.title("３次元および２次元プロット")

    dimension = st.selectbox("プロットする次元", ("２次元", "３次元"))

    # 生データの取得
    benchmarkName = [
        st.selectbox(options=["cg", "ep", "ft", "is", "lu", "mg"], label="ベンチマーク名")
    ]
    classes = ["S", "W", "A", "B", "C", "D", "E", "F"]
    processes = [2, 4, 8, 16, 32, 64, 128, 256]
    csvDirPath = "../csv_files/"
    rawDataDF = lib.returnCollectedExistingData(
        benchmarkNames=benchmarkName,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )

    st.markdown("## 抽出条件の選択")

    ## 2列にして、コア数・問題サイズを選択する
    ### 列の作成
    columnForCore, columnForSize = st.columns(2)
    ### コア数に関する列の設定
    columnForCore.subheader("コア数")
    #### 実際にデータ内にあるコア数のリストを作成

    enable001 = columnForCore.checkbox("コア数1", value=True)
    enable002 = columnForCore.checkbox("コア数2")
    enable004 = columnForCore.checkbox("コア数4")
    enable008 = columnForCore.checkbox("コア数8")
    enable016 = columnForCore.checkbox("コア数16")
    enable032 = columnForCore.checkbox("コア数32")
    enable064 = columnForCore.checkbox("コア数64")
    enable128 = columnForCore.checkbox("コア数128")
    enable256 = columnForCore.checkbox("コア数256")

    numOfCoreSet = set()
    if enable001:
        numOfCoreSet.add(1)
    if enable002:
        numOfCoreSet.add(2)
    if enable004:
        numOfCoreSet.add(4)
    if enable008:
        numOfCoreSet.add(8)
    if enable016:
        numOfCoreSet.add(16)
    if enable032:
        numOfCoreSet.add(32)
    if enable064:
        numOfCoreSet.add(64)
    if enable128:
        numOfCoreSet.add(128)
    if enable256:
        numOfCoreSet.add(256)

    numOfCoreList = sorted(list(numOfCoreSet))
    ### 問題サイズに関する列の設定
    columnForSize.subheader("問題サイズ")
    #### 実際にデータ内にある問題サイズのリストを作成

    enableA = columnForSize.checkbox("問題サイズA", value=True)
    enableB = columnForSize.checkbox("問題サイズB")
    enableC = columnForSize.checkbox("問題サイズC")
    enableD = columnForSize.checkbox("問題サイズD")
    enableE = columnForSize.checkbox("問題サイズE")
    enableF = columnForSize.checkbox("問題サイズF")

    programsizeSet = set()
    if enableA:
        programsizeSet.add("A")
    if enableB:
        programsizeSet.add("B")
    if enableC:
        programsizeSet.add("C")
    if enableD:
        programsizeSet.add("D")
    if enableE:
        programsizeSet.add("E")
    if enableF:
        programsizeSet.add("F")
    programsizeList = sorted(list(programsizeSet))

    # 問題サイズを数値化
    programSize = rawDataDF["benchmarkClass"].tolist()
    programSizeInNum = lib.convertBenchmarkClasses_problemSizeInNPB(
        inputList=programSize
    )
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
                        colorbar={
                            "bordercolor": "black",
                            "borderwidth": 5,
                            "outlinecolor": "black",
                            "outlinewidth": 5,
                        },
                        contour={"color": "black", "show": True, "width": 5},
                        lighting={"roughness": 1},
                        # flatshading=True,
                    )
                ],
                layout={},
            )

            if enableLogX:
                fig.update_scenes(xaxis_type="log")
            if enableLogY:
                fig.update_scenes(yaxis_type="log")
            if enableLogZ:
                fig.update_scenes(zaxis_type="log")

            fig2 = plt.figure()
            ax = fig2.gca(projection="3d")
            ax.plot_trisurf(x, y, z, alpha=0.9, color="g", antialiased=False)
            st.write(fig2)

        fig.update_layout(
            width=800,
            height=800,
            autosize=False,
            scene={
                "xaxis_title": "問題サイズ",
                "yaxis_title": "コア数",
                "zaxis_title": "関数コール回数",
                "aspectmode": "cube",
            },
        )

    else:
        pass

    st.write(fig)

    st.dataframe(DFtoPlot[["コア数", "関数コール回数", "問題サイズ（文字）", "問題サイズ"]])
