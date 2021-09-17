import streamlit as st


def app():
    @st.cache
    def gen_lib():
        subprocess.run(["jupyter", "nbconvert", "--to", "python", "../lib/lib.ipynb"])
        subprocess.run(["mv", "../lib/lib.py", "libLab00.py"])

    st.title("データの確認")

    st.header("問題サイズの指定")

    enableA = st.checkbox("問題サイズA", value=True)
    enableB = st.checkbox("問題サイズB")
    enableC = st.checkbox("問題サイズC")
    enableD = st.checkbox("問題サイズD")
    enableE = st.checkbox("問題サイズE")
    enableF = st.checkbox("問題サイズF")

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
    st.write(programsizeList)

    st.header("コア数の指定")

    enable001 = st.checkbox("コア数1", value=True)
    enable002 = st.checkbox("コア数2")
    enable004 = st.checkbox("コア数4")
    enable008 = st.checkbox("コア数8")
    enable016 = st.checkbox("コア数16")
    enable032 = st.checkbox("コア数32")
    enable064 = st.checkbox("コア数64")
    enable128 = st.checkbox("コア数128")
    enable256 = st.checkbox("コア数256")

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
    st.write(numOfCoreList)
