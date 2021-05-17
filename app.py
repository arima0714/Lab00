import streamlit as st
import subprocess
import pandas as pd


def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "lib.ipynb"])
    subprocess.run(["mv", "lib.py", "libLab00.py"])


# ライブラリノートを実行可能な形式に変換
gen_lib()
# 変換されたライブラリノートをimport
import libLab00 as lib

# コア数と問題サイズのどちらを固定するかを選択
fixedTarget = st.selectbox("コア数と問題サイズのどちらを固定するか？", ["コア数", "問題サイズ"])

# ベンチマークを選択
benchmark = st.selectbox("ベンチマークを選択", lib.benchmarks)

if benchmark == "bt" or benchmark == "sp":
    processes = lib.processes_onlyBTSP
else:
    processes = lib.processes_excludeBTSP
benchmarkClasses = lib.benchmarkClasses

if fixedTarget == "問題サイズ":
    fixed = st.selectbox("どの問題サイズで固定するか？", lib.benchmarkClasses)
else:
    fixed = st.selectbox("どのコア数で固定するか？", processes)

if fixedTarget == "問題サイズ":
    targetRawDF = lib.returnRawDFperBenchmark(
        Benchmark=benchmark, fix="Class", Processes=processes, FixedBenchmarkClass=fixed
    )
else:
    targetRawDF = lib.returnRawDFperBenchmark(
        Benchmark=benchmark, fix="Process", benchmarkClass=benchmarkClasses
    )

st.markdown(f"# {fixedTarget}を{fixed}に固定した時のベンチマーク{benchmark}の関数コール回数")
st.table(targetRawDF)

functionNames = targetRawDF.index.tolist()
functionName = st.selectbox("関数名を選択", functionNames)
st.markdown(f"# 関数{functionName}の関数コール回数")
targetFunctionDF = targetRawDF.loc[[functionName]]
st.table(targetFunctionDF)

raw_x = targetFunctionDF.columns.tolist()
raw_y = [targetFunctionDF.at[functionName, x] for x in raw_x]
if fixedTarget == "コア数":
    raw_x = lib.ConvertBenchmarkClasses(raw_x)


# グラフのプロット
## 準備
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

fig = plt.figure(figsize=(12, 9))
ax = plt.axes()
## 実データをsklearn用に変形
raw_x = np.array(raw_x).reshape(-1, 1)
raw_y = np.array(raw_y).reshape(-1, 1)
## 説明変数と目的変数に分割
notTrain_x, train_x, target_x = raw_x[0], raw_x[1:-1], raw_x[-1]
notTrain_y, train_y, target_y = raw_y[0], raw_y[1:-1], raw_y[-1]
## 実データをそれぞれプロット
plt.scatter(train_x, train_y, marker="o", label="予測に用いた関数コール回数")
plt.scatter(target_x, target_y, marker="o", label="予測したい関数コール回数の実測値")
plt.scatter(notTrain_x, notTrain_y, marker="o", label="最初のデータを除外した時に予測に用いなかった関数コール回数")
## モデル式をプロットするために変数”plot_x”を用意する
plot_x_min = 0.01
plot_x_max = 256.0
### モデル式としてプロットする最低値の設定
_min = st.slider(
    "モデル式に表示する最小値", min_value=plot_x_min, max_value=plot_x_max, step=0.3, value=1.0
)
plot_x = np.linspace(_min, plot_x_max, 500)
plot_x = np.array(plot_x).reshape(-1, 1)


"""
# モデルの選択
"""
## 最初のデータを除外したモデル式のプロット
if st.checkbox("線形モデル(コア数1もしくは問題サイズAのデータを除外)"):
    model_lin = lib.ModelLin(train_x, train_y, benchmark, functionName, test_ratio=0)
    model_lin.calc_lr()
    plot_y_lin = model_lin.predict(plot_x)
    plt.plot(plot_x, plot_y_lin, label="線形モデル(コア数1もしくは問題サイズAのデータを除外)")
if st.checkbox("対数モデル(コア数1もしくは問題サイズAのデータを除外)"):
    model_log10 = lib.ModelLog10(
        train_x, train_y, benchmark, functionName, test_ratio=0
    )
    model_log10.calc_lr()
    plot_y_log10 = model_log10.predict(plot_x)
    plt.plot(plot_x, plot_y_log10, label="対数モデル(コア数1もしくは問題サイズAのデータを除外)")
if st.checkbox("反比例モデル(コア数1もしくは問題サイズAのデータを除外)"):
    model_ip = lib.ModelIP(train_x, train_y, benchmark, functionName, test_ratio=0)
    model_ip.calc_lr()
    plot_y_ip = model_ip.predict(plot_x)
    plt.plot(plot_x, plot_y_ip, label="反比例モデル(コア数1もしくは問題サイズAのデータを除外)")
if st.checkbox("反比例モデル ver2(コア数1もしくは問題サイズAのデータを除外)"):
    model_ip2 = lib.ModelIp_mk2(train_x, train_y, target_x, target_y)
    model_ip2.calc_lr()
    plot_y_ip2 = model_ip2.predict(plot_x)
    plt.plot(plot_x, plot_y_ip2, label="反比例モデル ver2(コア数1もしくは問題サイズAのデータを除外)")
if st.checkbox("線形飽和モデル(コア数1もしくは問題サイズAのデータを除外)"):
    model_branch = lib.ModelBranch(
        train_x, train_y, benchmark, functionName, test_ratio=0
    )
    model_branch.calc_lr()
    plot_y_branch = model_branch.predict(plot_x)
    plt.plot(plot_x, plot_y_branch, label="線形飽和モデル(最初のデータを除外)")

## 最初のデータを含む学習データと最後のデータのみの試験用データに分割
train_x, target_x = raw_x[:-1], raw_x[-1]
train_y, target_y = raw_y[:-1], raw_y[-1]
## ロバスト回帰によるモデル式のプロット
if st.checkbox("線形モデル(ロバスト回帰)"):
    model_lin_rob = lib.ModelLin_rob(
        train_x=train_x, train_y=train_y, target_x=target_x, target_y=target_y
    )
    model_lin_rob.calc_hr()
    plot_y_lin_rob = model_lin_rob.predict(plot_x)
    plt.plot(plot_x, plot_y_lin_rob, label="線形モデル（ロバスト回帰）")
if st.checkbox("反比例モデル(ロバスト回帰)"):
    model_ip_rob = lib.ModelIp_rob(
        train_x=train_x, train_y=train_y, target_x=target_x, target_y=target_y
    )
    model_ip_rob.calc_hr()
    plot_y_ip_rob = model_ip_rob.predict(plot_x)
    plt.plot(plot_x, plot_y_ip_rob, label="線形モデル（ロバスト回帰）")
if st.checkbox("対数モデル(ロバスト回帰)"):
    model_log_rob = lib.ModelLog10_rob(
        train_x=train_x, train_y=train_y, target_x=target_x, target_y=target_y
    )
    model_log_rob.calc_hr()
    plot_y_log_rob = model_log_rob.predict(plot_x)
    plt.plot(plot_x, plot_y_log_rob, label="対数モデル（ロバスト回帰）")
if st.checkbox("線形モデル"):
    model_lin = lib.ModelLin(train_x, train_y, benchmark, functionName, test_ratio=0)
    model_lin.calc_lr()
    plot_y_lin = model_lin.predict(plot_x)
    plt.plot(plot_x, plot_y_lin, label="線形モデル")
if st.checkbox("対数モデル"):
    model_log10 = lib.ModelLog10(
        train_x, train_y, benchmark, functionName, test_ratio=0
    )
    model_log10.calc_lr()
    plot_y_log10 = model_log10.predict(plot_x)
    plt.plot(plot_x, plot_y_log10, label="対数モデル")
if st.checkbox("反比例モデル"):
    model_ip = lib.ModelIP(train_x, train_y, benchmark, functionName, test_ratio=0)
    model_ip.calc_lr()
    plot_y_ip = model_ip.predict(plot_x)
    plt.plot(plot_x, plot_y_ip, label="反比例モデル")
if st.checkbox("反比例モデル ver2"):
    model_ip2 = lib.ModelIp_mk2(train_x, train_y, target_x, target_y)
    model_ip2.calc_lr()
    plot_y_ip2 = model_ip2.predict(plot_x)
    plt.plot(plot_x, plot_y_ip2, label="反比例モデル ver2")
if st.checkbox("線形飽和モデル"):
    model_branch = lib.ModelBranch(
        train_x, train_y, benchmark, functionName, test_ratio=0
    )
    model_branch.calc_lr()
    plot_y_branch = model_branch.predict(plot_x)
    plt.plot(plot_x, plot_y_branch, label="線形飽和モデル")


# 凡例の表示
plt.legend()
# 軸ラベルの設定
plt.ylabel("関数コール回数")
plt.xlabel("実行コア数")

"""
# 軸の対数軸化
"""

if st.checkbox("X軸の対数化"):
    plt.xscale("log")
if st.checkbox("Y軸の対数化"):
    plt.yscale("log")

st.pyplot(fig)
