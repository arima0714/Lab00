import streamlit as st
import subprocess
import pandas as pd

# DataFrameを生成するcache機能付の関数を定義
@st.cache
def get_df(s):
    return pd.DataFrame({'id':['1','2','3'], 'name':['X', 'Y', s]})

def gen_lib():
    subprocess.run(["jupyter", "nbconvert", "--to", "python", "lib.ipynb"])
    subprocess.run(["mv", "lib.py", "libLab00.py"])
# ライブラリノートを実行可能な形式に変換
gen_lib()
# 変換されたライブラリノートをimport
import libLab00 as lib

# # text formを生成
# s = st.text_input('input s')
# df = get_df(s)
#
# # DataFrameをテーブルで表示
# st.markdown('# Table')
# st.table(df)
#
# # histgramを表示
# st.markdown('# Histgram')
# df['name'].hist()
# st.pyplot()

# コア数と問題サイズのどちらを固定するかを選択
fixedTarget = st.selectbox("コア数と問題サイズのどちらを固定するか？", ["コア数", "問題サイズ"])
st.write(f"{fixedTarget} を選択")

# ベンチマークを選択
benchmark = st.selectbox("ベンチマークを選択", lib.benchmarks)
st.write(f"{benchmark} を選択")


if benchmark == "bt" or benchmark == "sp":
    processes = lib.processes_onlyBTSP
else:
    processes = lib.processes_excludeBTSP
benchmarkClasses = lib.benchmarkClasses

if fixedTarget == "問題サイズ":
    fixed = st.selectbox("どの問題サイズで固定するか？", lib.benchmarkClasses)
else:
    fixed = st.selectbox("どのコア数で固定するか？", processes)
st.write(f"{fixed}を選択")

if fixedTarget == "問題サイズ":
    targetRawDF = lib.returnRawDFperBenchmark(Benchmark=benchmark, fix="Class", Processes=processes, FixedBenchmarkClass=fixed)
else:
    targetRawDF = lib.returnRawDFperBenchmark(Benchmark=benchmark, fix="Process", benchmarkClass=benchmarkClasses)

st.markdown('# targetRawDF')
st.table(targetRawDF)

functionNames = targetRawDF.index.tolist()
functionName = st.selectbox("関数名を選択", functionNames)

targetFunctionDF = targetRawDF.loc[[functionName]]
st.table(targetFunctionDF)

raw_x = targetFunctionDF.columns.tolist()
raw_y = [targetFunctionDF.at[functionName, x] for x in raw_x]
if(fixedTarget == "コア数"):
    raw_x = lib.ConvertBenchmarkClasses(raw_x)

# グラフのプロット
