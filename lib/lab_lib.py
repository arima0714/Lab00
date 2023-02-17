#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""ライブラリノート

* 研究で使用する関数や一部の変数を保持したライブラリ

"""


# In[2]:


import copy
import datetime
import glob
import itertools
import math
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pprint
import pytest
import random
import re
import sys
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statistics import median_low
import subprocess
import sympy
from sympy import sympify, pprint, symbols, log
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict
from unittest.mock import MagicMock
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import tree
from IPython.display import Image
from sklearn.datasets import fetch_california_housing
import pydotplus


# In[3]:


# ログ関連処理

from logging import basicConfig, getLogger, DEBUG

basicConfig(level=DEBUG)
logger = getLogger(__name__)

logger.debug("hello")


# In[4]:


class ExceptionInResearchLib(Exception):
    "ライブラリノートでの例外処理用のクラス"


def returnMapeScore(l1, l2):
    """returnMapeScore(l1, l2)
    平均絶対パーセント誤差 (MAPE)(Mean Absolute Percent Error (MAPE))を返す関数

    Args:
        引数として長さの同じ二つのリストをとる
        l1 :list[float] 実測値のリスト
        l2 :list[float] 予測値のリスト
    Returns:
      float : 引数から計算されたMAPEの値（単位％）
    """
    return_num = 0
    if len(l1) != len(l2):
        warnings.warn("引数のリストの長さが異なります")
        return -1
    for i in range(len(l1)):
        l1_num = l1[i]
        l2_num = l2[i]

        return_num += abs((l1_num - l2_num) / l1_num)

    return_num /= len(l1)
    return_num *= 100
    return return_num


def test_returnMapeScore():
    l1 = [1, 2, 3, 4]
    l2 = [4, 3, 2, 1]
    ansByFunc = returnMapeScore(l1, l2)
    ansByHand = (
        (abs(1 - 4) / 1 + abs(2 - 3) / 2 + abs(3 - 2) / 3 + abs(4 - 1) / 4) / 4 * 100
    )
    # 多少の誤差を許容する
    ansByFunc = int(ansByFunc * 100) / 100
    ansByHand = int(ansByHand * 100) / 100

    assert ansByFunc == ansByHand


# In[5]:


# ベンチマークを指定して存在するファイル名のものを返す
def returnExistingFileNames(
    benchmarkNames=[], classes=[], processes=[], csvDirPath="./csv_files"
):
    candidateFileNames = {}
    returnDict = {}
    for benchmarkName in benchmarkNames:
        for benchmarkClass in classes:
            for process in processes:
                candidateFileNames[
                    f"pprof_{benchmarkName}{benchmarkClass}{process}.csv"
                ] = {
                    "benchmarkName": benchmarkName,
                    "benchmarkClass": benchmarkClass,
                    "process": process,
                }
    for candidateFileName in candidateFileNames.keys():
        filePath = os.path.join(csvDirPath, candidateFileName)
        if os.path.exists(filePath) and os.stat(filePath).st_size != 0:
            returnDict[candidateFileName] = candidateFileNames[candidateFileName]
    return returnDict


def test_returnExistingFileNames():
    benchmarkNames = ["test"]
    classes = ["A", "B", "C", "D"]
    processes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    csvDirPath = "../csv_files/"
    returnedList = returnExistingFileNames(
        benchmarkNames=benchmarkNames,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    assert returnedList["pprof_testA128.csv"] == {
        "benchmarkName": "test",
        "benchmarkClass": "A",
        "process": 128,
    }
    assert returnedList["pprof_testB256.csv"] == {
        "benchmarkName": "test",
        "benchmarkClass": "B",
        "process": 256,
    }


# In[6]:


# ベンチマーク名・プロセス数・ベンチマーククラスをリストで渡して、実在するデータが集計されたDFを返す
def returnCollectedExistingData(
    benchmarkNames=[], classes=[], processes=[], csvDirPath="./csv_files/"
):
    fileNames = returnExistingFileNames(
        benchmarkNames=benchmarkNames,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    csvDataList = []
    for fileName in fileNames.keys():
        rawDatum = pd.read_csv(f"{csvDirPath}{fileName}")
        rawDatum["benchmarkName"] = fileNames[fileName]["benchmarkName"]
        rawDatum["benchmarkClass"] = fileNames[fileName]["benchmarkClass"]
        rawDatum["process"] = fileNames[fileName]["process"]
        csvDataList.append(rawDatum)
    returnDF = pd.concat(csvDataList, axis=0)
    returnDF = returnDF.rename(
        columns={"Name": "functionName", "#Call": "functionCallNum"}
    )
    return returnDF


def test_returnCollectedExistingData():
    benchmarkNames = ["test"]
    classes = ["A", "B", "C", "D"]
    processes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    csvDirPath = "../csv_files/"
    returnedData = returnCollectedExistingData(
        benchmarkNames=benchmarkNames,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )

    case01 = {
        "benchmarkName": "test",
        "benchmarkClass": "A",
        "process": 128,
        "functionCalls": {"function00": 99, "function01": 77, "function02": 555},
    }
    case02 = {
        "benchmarkName": "test",
        "benchmarkClass": "B",
        "process": 256,
        "functionCalls": {"function00": 5, "function01": 70, "function02": 900},
    }

    for case in [case01, case02]:
        benchmarkName = case["benchmarkName"]
        benchmarkClass = case["benchmarkClass"]
        process = case["process"]
        for functionName in case["functionCalls"]:
            functionCallNum = case["functionCalls"][functionName]
            targetData = returnedData[
                (returnedData["benchmarkName"] == benchmarkName)
                & (returnedData["benchmarkClass"] == benchmarkClass)
                & (returnedData["process"] == process)
                & (returnedData["functionName"] == functionName)
            ]
            columns = targetData.columns.tolist()
            functionCallNumIndex = columns.index("functionCallNum")
            assert targetData.iloc[0, functionCallNumIndex] == functionCallNum


# In[7]:


# モデルの共通部分となるクラス
# すべての引数はただのリスト。クラスの初期化時に""np.reshape()""を実行する
class ModelBase:
    def __init__(
        self,
        trainX,
        trainY,
        targetX=[],
        targetY=[],
        benchmarkName="benchmarkName",
        functionName="functionName",
    ):
        self.benchmarkName = benchmarkName
        self.functionName = functionName

        self.rawTrainX = trainX
        self.rawTrainY = trainY

        self.trainX = np.reshape(trainX, (-1, 1))
        self.trainY = np.reshape(trainY, (-1, 1))
        self.targetX = np.reshape(targetX, (-1, 1))
        self.targetY = np.reshape(targetY, (-1, 1))

    def returnTargetX(self):
        return self.targetX

    def returnTargetY(self):
        return self.targetY

    def returnTrainX(self):
        return self.trainX

    def returnTrainY(self):
        return self.trainY


# # このクラスを継承したモデルは、いずれも次のように使用する
# _modelLin = ModelLin(trainX=trainX, trainY=trainY, targetX=targetX, targetY=targetY)
# _modelLin.calcLr()
# plotY = _modelLin.predict(plotX)


# In[8]:


# 分岐モデル


class ModelBranch(ModelBase):
    def calcLr(self):
        # t:最大値のインデックス
        self.t = np.ndarray.argmax(self.trainY)
        # tNum:最大値
        self.tNum = self.trainX[self.t]
        # 最大値のインデックスのリストを作成
        tIndice = [i for i, x in enumerate(self.trainY) if x == max(self.trainY)]
        conditionBefore = self.t == 0 or self.t == len(self.trainY) - 1
        conditionAfter = len(tIndice) == 1
        if conditionBefore or conditionAfter:
            self.lr1 = LinearRegression()
            self.lr1.fit(self.trainX, self.trainY)
            self.lr2 = LinearRegression()
            self.lr2.fit(self.trainX, self.trainY)
        else:
            self.trainX1 = self.trainX[: self.t]
            self.trainX2 = self.trainX[self.t :]
            self.trainY1 = self.trainY[: self.t]
            self.trainY2 = self.trainY[self.t :]
            self.lr1 = LinearRegression()
            self.lr1.fit(self.trainX1, self.trainY1)
            self.lr2 = LinearRegression()
            self.lr2.fit(self.trainX2, self.trainY2)

    def predict(self, num):
        num = np.reshape(num, (-1, 1))
        numT = np.ndarray.argmax(num)
        numTMax = num[numT]
        k = np.abs(np.asarray(num) - self.tNum).argmin()
        if len(num) == 1 and numTMax >= self.tNum:
            predicted = self.lr2.predict(num)
            return predicted
        elif numTMax < self.trainX[self.t] or k == 0:
            predicted = self.lr1.predict(num)
            return predicted
        else:
            num1 = num[:k]
            num2 = num[k:]
            predicted1 = self.lr1.predict(num1)
            predicted2 = self.lr2.predict(num2)
            predicted = np.concatenate([predicted1, predicted2])
            return predicted

    def ModelName(self):
        return "ModelBranch"


# 線形飽和モデル
# テスト用モデル式1：
# y = 2x + 3 (x < 10)
#     23     (x >= 10)
# テスト用モデル式2：
# y = 2x + 3


def test_ModelBranch():
    # X軸の値
    plotXForBranch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # Y軸の値
    plotYForBranch = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23, 23]
    #     plt.figure()
    #     plt.plot(plotXForBranch, plotYForBranch, label="y=2*+3(x<10), y=23(x>=10)")
    # モデルの構築
    _modelBranch = ModelBranch(
        trainX=plotXForBranch, trainY=plotYForBranch, targetX=[], targetY=[]
    )
    _modelBranch.calcLr()
    predictedYForBranch = _modelBranch.predict(plotXForBranch)
    #     plt.plot(plotXForBranch, predictedYForBranch, label="線形飽和モデルによるモデル式")
    #     plt.legend()
    mapeScore = returnMapeScore(plotYForBranch, predictedYForBranch)
    assert mapeScore < 1

    # 線形モデルとしても利用可能かのテストケース
    plotX = np.linspace(0.5, 270, 500)
    plotY = 2 * plotX + 3
    _modelBranch2 = ModelBranch(trainX=plotX, trainY=plotY, targetX=[], targetY=[])
    _modelBranch2.calcLr()
    predictedY = _modelBranch2.predict(plotX)
    mapeScore = returnMapeScore(plotY, predictedY)
    assert (
        mapeScore < 1
    ), f"{_modelBranch2.lr1.coef_}, {_modelBranch2.lr1.intercept_}, {_modelBranch2.lr2.coef_}, {_modelBranch2.lr2.intercept_}"


# In[9]:


# # 分岐モデル2


# class ModelBranch2(ModelBase):
#     def calcLr(self):
#         # 分岐点branchIndexを検出する
#         self.branchIndex = returnBranchIndexOfList(
#             inputListX=self.rawTrainX, inputListY=self.rawTrainY
#         )

#         # 分岐点が端点
#         if (
#             self.branchIndex == 0
#             or self.branchIndex == len(self.rawTrainY)
#             or self.branchIndex == -1
#         ):
#             self.lr1 = LinearRegression()
#             self.lr1.fit(self.trainX, self.trainY)
#             self.lr2 = LinearRegression()
#             self.lr2.fit(self.trainX, self.trainY)
#         # 分岐点が端点ではない
#         else:
#             self.trainX1 = self.trainX[: self.branchIndex]
#             self.trainX2 = self.trainX[self.branchIndex :]
#             self.trainY1 = self.trainY[: self.branchIndex]
#             self.trainY2 = self.trainY[self.branchIndex :]
#             self.lr1 = LinearRegression()
#             self.lr1.fit(self.trainX1, self.trainY1)
#             self.lr2 = LinearRegression()
#             self.lr2.fit(self.trainX2, self.trainY2)

#     def predict(self, num):

#         # listの場合
#         if type(num) == list:
#             # 入力値の最大値
#             valueMaxInNum = max(num)
#             # 入力値の最小値
#             valueMinInNum = min(num)
#         # floatとintを想定
#         else:
#             valueMaxInNum = valueMinInNum = num

#         num = np.reshape(num, (-1, 1))

#         branchNumX = self.rawTrainX[self.branchIndex]
#         branchNumY = self.rawTrainY[self.branchIndex]

#         # 全ての値が分岐点未満
#         if valueMaxInNum < branchNumX:
#             predicted = self.lr1.predict(num)
#         # 全ての入力値が分岐点以上
#         elif valueMinInNum >= branchNumX:
#             predicted = self.lr2.predict(num)

#         # 入力値が分岐点にまたがっている
#         else:
#             # 入力値のリストであるnumを分岐点未満のリストと分岐点以上のリストに分ける
#             lessThanBranch = []
#             greaterThanBranch = []
#             for numberInNum in num:
#                 if numberInNum < branchNumX:
#                     lessThanBranch.append(numberInNum)
#                 else:
#                     greaterThanBranch.append(numberInNum)
#             #             lessThanBranch = np.reshape(lessThanBranch, (-1, 1))
#             #             greaterThanBranch = np.reshape(greaterThanBranch, (-1, 1))
#             predicted1 = self.lr1.predict(lessThanBranch)
#             predicted2 = self.lr2.predict(greaterThanBranch)
#             predicted = np.concatenate([predicted1, predicted2])

#         return predicted

#     def ModelName(self):
#         return "ModelBranch2"


# # 線形飽和モデル2
# # テスト用モデル式1：
# # y = 2x + 3 (x < 10)
# #     23     (x >= 10)
# # テスト用モデル式2：
# # y = 2x + 3


# def test_ModelBranch2():
#     # X軸の値
#     plotXForBranch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#     # Y軸の値
#     plotYForBranch = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23, 23]
#     #     plt.figure()
#     #     plt.plot(plotXForBranch, plotYForBranch, label="y=2*+3(x<10), y=23(x>=10)")
#     # モデルの構築
#     _modelBranch = ModelBranch2(
#         trainX=plotXForBranch, trainY=plotYForBranch, targetX=[], targetY=[]
#     )
#     _modelBranch.calcLr()
#     predictedYForBranch = _modelBranch.predict(plotXForBranch)
#     #     plt.plot(plotXForBranch, predictedYForBranch, label="線形飽和モデルによるモデル式")
#     #     plt.legend()
#     mapeScore = returnMapeScore(plotYForBranch, predictedYForBranch)
#     assert mapeScore < 1

#     # 線形モデルとしても利用可能かのテストケース
#     plotX = np.linspace(0.5, 270, 500)
#     plotY = 2 * plotX + 3
#     _modelBranch2 = ModelBranch2(trainX=plotX, trainY=plotY, targetX=[], targetY=[])
#     _modelBranch2.calcLr()
#     predictedY = _modelBranch2.predict(plotX)
#     mapeScore = returnMapeScore(plotY, predictedY)
#     assert (
#         mapeScore < 1
#     ), f"{_modelBranch2.lr1.coef_}, {_modelBranch2.lr1.intercept_}, {_modelBranch2.lr2.coef_}, {_modelBranch2.lr2.intercept_}"


# In[10]:


# 反比例モデル


def ipFunc(x):
    return 1 / x


class ModelIp(ModelBase):
    def calcLr(self):
        self.transformerIp = sp.FunctionTransformer(func=ipFunc, inverse_func=ipFunc)
        trainXIp = self.transformerIp.transform(self.trainX)
        self.lr = LinearRegression()
        self.lr.fit(trainXIp, self.trainY)

    def predict(self, num):
        num = np.reshape(num, (-1, 1))
        numConverted = self.transformerIp.transform(num)
        predicted = self.lr.predict(numConverted)
        return predicted

    def return_coef_(self):
        return self.lr.coef_

    def return_intercept_(self):
        return self.lr.intercept_

    def ModelName(self):
        return "ModelIp"


# 反比例モデル
# テスト用モデル式：
# y = 2/x + 3


def test_ModelIp():
    # X軸の連続値
    plotX = np.linspace(0.5, 270, 500)
    #     plt.figure()
    plotY = 2 / plotX + 3
    #     plt.plot(plotX, plotY, label="y = 2/x + 3")
    # モデルの構築
    _modelIp = ModelIp(trainX=plotX, trainY=plotY, targetX=[], targetY=[])
    _modelIp.calcLr()
    predictedY = _modelIp.predict(plotX)
    #     plt.plot(plotX, predictedY, label="反比例モデルによるモデル式")
    #     plt.legend()
    mapeScore = returnMapeScore(plotY, predictedY)
    assert mapeScore < 1


# In[11]:


# 線形モデル


class ModelLin(ModelBase):
    def calcLr(self):
        self.lr = LinearRegression()
        self.lr.fit(self.trainX, self.trainY)

    def predict(self, num):
        num = np.reshape(num, (-1, 1))
        predicted = self.lr.predict(num)
        return predicted

    def return_coef_(self):
        return self.lr.coef_

    def return_intercept_(self):
        return self.lr.intercept_

    def ModelName(self):
        return "ModelLin"


# 線形モデル
# テスト用モデル式：
# y = 2x + 3


def test_ModelLin():
    # X軸の連続値
    plotX = np.linspace(0.5, 270, 500)
    #     plt.figure()
    plotY = 2 * plotX + 3
    #     plt.plot(plotX, plotY, label="y = 2 * x + 3")
    # モデルの構築
    _modelLin = ModelLin(trainX=plotX, trainY=plotY, targetX=[], targetY=[])
    _modelLin.calcLr()
    predictedY = _modelLin.predict(plotX)
    #     plt.plot(plotX, predictedY, label="線形モデルによるモデル式")
    #     plt.legend()
    mapeScore = returnMapeScore(plotY, predictedY)
    assert mapeScore < 1


# In[12]:


# 対数モデル


def inverterLog10Func(x):
    return 10**x


class ModelLog10(ModelBase):
    def calcLr(self):
        self.transformerLog10 = sp.FunctionTransformer(
            func=np.log10, inverse_func=inverterLog10Func
        )
        trainXLog10 = self.transformerLog10.transform(self.trainX)
        self.lr = LinearRegression()
        self.lr.fit(trainXLog10, self.trainY)

    def predict(self, num):
        num = np.reshape(num, (-1, 1))
        numConverted = self.transformerLog10.transform(num)
        predicted = self.lr.predict(numConverted)
        return predicted

    def return_coef_(self):
        return self.lr.coef_

    def return_intercept_(self):
        return self.lr.intercept_

    def ModelName(self):
        return "ModelLog"


# 対数モデル
# テスト用モデル式：
# y = 2 * log_{10}{x} + 3


def test_ModelLog10():
    # X軸の連続値
    plotX = np.linspace(0.5, 270, 500)
    #     plt.figure()
    plotY = 2 * np.log10(plotX) + 3
    #     plt.plot(plotX, plotY, label="y = 2 * log_{10}{x} + 3")
    # モデルの構築
    _modelLog10 = ModelLog10(trainX=plotX, trainY=plotY, targetX=[], targetY=[])
    _modelLog10.calcLr()
    predictedY = _modelLog10.predict(plotX)
    #     plt.plot(plotX, predictedY, label="対数モデルによるモデル式")
    #     plt.legend()
    mapeScore = returnMapeScore(plotY, predictedY)
    assert mapeScore < 1


# In[13]:


# 引数として渡されたDFに
# functionName, functionCallNum, benchmarkName, benchmarkClass, process
# のカラムがあるかを確認する関数
# あればTrue、なければFalseを返す
def checkRawDFColumns(DF):
    columns = DF.columns.tolist()
    columnNames = [
        "functionName",
        "functionCallNum",
        "benchmarkName",
        "benchmarkClass",
        "process",
    ]
    for columnName in columnNames:
        if (columnName in columns) == False:
            return False
    return True


def test_checkRawDFColumns():
    # Trueケース
    True01DF = pd.DataFrame(
        [["functionName", -1, "benchmarkName", "Z", -1]],
        columns=[
            "functionName",
            "functionCallNum",
            "benchmarkName",
            "benchmarkClass",
            "process",
        ],
    )
    True02DF = pd.DataFrame(
        [["functionName", -1, "benchmarkName", "Z", -1, "addedData0"]],
        columns=[
            "functionName",
            "functionCallNum",
            "benchmarkName",
            "benchmarkClass",
            "process",
            "addedData0",
        ],
    )

    # Falseケース
    False01DF = pd.DataFrame(
        [["functionName", -1, "benchmarkName", "Z"]],
        columns=["functionName", "functionCallNum", "benchmarkName", "benchmarkClass"],
    )
    False02DF = pd.DataFrame(
        [[-1, "benchmarkName", "Z", -1]],
        columns=["functionCallNum", "benchmarkName", "benchmarkClass", "process"],
    )

    assert True == checkRawDFColumns(True01DF)
    assert True == checkRawDFColumns(True02DF)
    assert False == checkRawDFColumns(False01DF)
    assert False == checkRawDFColumns(False02DF)


# In[14]:


# 実験結果を集計するためのデータフレームのカラムの名称のリストを返す関数
def returnNumOfColumns(
    dataType=False, modelNames=["ModelLog", "ModelIp", "ModelBranch", "ModelLin"]
):
    returnList = []
    returnDict = {}
    # ベンチマーク名
    returnList.append("benchmarkName")
    returnDict["benchmarkName"] = str
    # 関数名
    returnList.append("functionName")
    returnDict["functionName"] = str
    # 使用データ(説明変数のリスト)
    returnList.append("usedDataX")
    returnDict["usedDataX"] = object
    # 使用データ(目的変数のリスト)
    returnList.append("usedDataY")
    returnDict["usedDataY"] = object
    # 使用データ数
    returnList.append("numOfData")
    returnDict["numOfData"] = "int16"
    # 固定したもの("Process" or "Class")
    returnList.append("ProcessOrClass")
    returnDict["ProcessOrClass"] = str
    # 固定したもの(プロセス数(数値)or問題サイズ(文字列))
    returnList.append("fixed")
    returnDict["fixed"] = object
    # 予測対象プロセス数
    returnList.append("targetProcess")
    returnDict["targetProcess"] = "int16"
    # 予測対象問題サイズ
    returnList.append("targetProblemSize")
    returnDict["targetProblemSize"] = str
    # 予測対象関数コール回数
    returnList.append("targetNumOfFunctionCall")
    returnDict["targetNumOfFunctionCall"] = "float32"
    # 線形モデル
    if "ModelLin" in modelNames:
        # 線形モデルのオブジェクト
        returnList.append("objectLinModel")
        returnDict["objectLinModel"] = object
        # 線形モデルのMAPE
        returnList.append("MAPEOfLinModel")
        returnDict["MAPEOfLinModel"] = "float32"
    # 反比例モデル
    if "ModelIp" in modelNames:
        # 反比例モデルのオブジェクト
        returnList.append("objectIpModel")
        returnDict["objectIpModel"] = object
        # 反比例モデルのMAPE
        returnList.append("MAPEOfIpModel")
        returnDict["MAPEOfIpModel"] = "float32"
    # 対数モデル
    if "ModelLog" in modelNames:
        # 対数モデルのオブジェクト
        returnList.append("objectLogModel")
        returnDict["objectLogModel"] = object
        # 対数モデルのMAPE
        returnList.append("MAPEOfLogModel")
        returnDict["MAPEOfLogModel"] = "float32"
    # 線形飽和モデル
    if "ModelBranch" in modelNames:
        # 線形飽和モデルのオブジェクト
        returnList.append("objectBranchModel")
        returnDict["objectBranchModel"] = object
        # 線形飽和モデルのMAPE
        returnList.append("MAPEOfBranchModel")
        returnDict["MAPEOfBranchModel"] = "float32"
    if "ModelBranch2" in modelNames:
        # 線形飽和モデルのオブジェクト
        returnList.append("objectBranchModel2")
        returnDict["objectBranchModel2"] = object
        # 線形飽和モデルのMAPE
        returnList.append("MAPEOfBranchModel2")
        returnDict["MAPEOfBranchModel2"] = "float32"
    # 説明変数に対するMAPEが最小のモデル名
    returnList.append("objectBestModelName")
    returnDict["objectBestModelName"] = object
    # 説明変数に対するMAPEが最小のモデルを用いて予測対象の関数コール回数を予測した時の平均絶対相対誤差率[%]
    returnList.append("MAPEOfBestModel")
    returnDict["MAPEOfBestModel"] = "float32"
    # 目標関数コール回数に対する、絶対相対誤差率を保持する
    returnList.append("RelativeErrorRate")
    returnDict["RelativeErrorRate"] = "float32"
    if dataType == True:
        return returnDict
    else:
        return returnList


# 使用例
# columnNames = return_numOfColumns()
# df_sample = pd.DataFrame(columns=columnNames)
# df_sample


def test_returnNumOfColumns():
    lengthOfDictAndList = 21
    modelNamesFull = ["ModelLin", "ModelLog", "ModelBranch", "ModelIp"]

    returnedList = returnNumOfColumns(modelNames=modelNamesFull)
    returnedDict = returnNumOfColumns(dataType=True, modelNames=modelNamesFull)
    # カラム名と辞書のキーが一致しているかを確認
    for key in returnedDict.keys():
        assert key in returnedList
    # カラム名を返す場合にリスト長が想定通りかどうかを確認
    assert len(returnedList) == lengthOfDictAndList
    # カラム名を返す場合に辞書のキー数が想定通りかどうかを確認
    assert len(returnedDict.keys()) == lengthOfDictAndList

    modelNames = ["ModelLin", "ModelLog", "ModelIp"]
    returnedList = returnNumOfColumns(modelNames=modelNames)
    assert len(returnedList) == (lengthOfDictAndList) + 2 * (
        -len(modelNamesFull) + len(modelNames)
    )
    modelNames = ["ModelLin"]
    returnedList = returnNumOfColumns(modelNames=modelNames)
    assert len(returnedList) == (lengthOfDictAndList) + 2 * (
        -len(modelNamesFull) + len(modelNames)
    )


# In[15]:


def returnSpecificDataFromCSV(
    benchmarkName="cg",
    functionName=".TAU_application",
    process="1",
    benchmarkClass="A",
    csvDirPath="./csv_files",
):
    fileName = f"pprof_{benchmarkName}{benchmarkClass}{process}.csv"
    filePath = f"{csvDirPath}/{fileName}"
    rawCSVData = pd.read_csv(filePath)
    rawCSVDataPerFunction = rawCSVData[(rawCSVData["Name"] == functionName)].set_index(
        "Name"
    )
    returnData = rawCSVDataPerFunction.at[functionName, "#Call"]
    return returnData


def test_returnSpecificDataFromCSV():
    case01 = {
        "benchmarkName": "test",
        "benchmarkClass": "A",
        "process": 128,
        "functionName": "function00",
        "functionCallNum": 99,
    }
    case02 = {
        "benchmarkName": "test",
        "benchmarkClass": "B",
        "process": 256,
        "functionName": "function02",
        "functionCallNum": 900,
    }
    for case in [case01, case02]:
        benchmarkName = case["benchmarkName"]
        benchmarkClass = case["benchmarkClass"]
        process = case["process"]
        functionName = case["functionName"]
        functionCallNum = case["functionCallNum"]
        assert functionCallNum == returnSpecificDataFromCSV(
            benchmarkName=benchmarkName,
            functionName=functionName,
            process=process,
            benchmarkClass=benchmarkClass,
            csvDirPath="../csv_files",
        )


# In[16]:


def convertStrToInt_problemSizeInNPB(Alphabet: str):
    if Alphabet == "S":
        return 0.625
    elif Alphabet == "W":
        return 0.125
    elif Alphabet == "A":
        return 1
    elif Alphabet == "B":
        return 4
    elif Alphabet == "C":
        return 16
    elif Alphabet == "D":
        return 256
    elif Alphabet == "E":
        return 4096
    elif Alphabet == "F":
        return 65536
    else:
        return -1


def test_convertStrToInt_problemSizeInNPB():
    case00 = {"input": "A", "output": 1}
    case01 = {"input": "Z", "output": -1}

    for case in [case00, case01]:
        output = convertStrToInt_problemSizeInNPB(case["input"])
        assert output == case["output"]


def convertIntToStr_problemSizeInNPB(number):
    number = int(number)
    if number == 1:
        return "A"
    elif number == 4:
        return "B"
    elif number == 16:
        return "C"
    elif number == 256:
        return "D"
    elif number == 4096:
        return "E"
    elif number == 65536:
        return "F"
    else:
        return "Z"


def test_convertIntToStr_problemSizeInNPB():
    case00 = {"input": 1, "output": "A"}
    case01 = {"input": -1, "output": "Z"}

    for case in [case00, case01]:
        output = convertIntToStr_problemSizeInNPB(case["input"])
        assert output == case["output"]


# 1文字ずつのリストとして渡された問題サイズを数値に変換する関数
# 入力引数inputList：["X1", "X2", ... , "Xn"]
# 返り値：["<X1を数値化した値>", "<X2を数値化した値>", ... , "<Xnを数値化した値>"]
def convertBenchmarkClasses_problemSizeInNPB(inputList=["A", "B", "C", "D"]):
    ReturnList = []
    for content in inputList:
        ReturnList.append(convertStrToInt_problemSizeInNPB(content))
    return ReturnList


def test_convertBenchmarkClasses_problemSizeInNPB():
    case00 = {"input": ["A", "B", "C", "D"], "output": [1, 4, 16, 256]}
    case01 = {"input": ["D", "A"], "output": [256, 1]}
    case02 = {"input": ["A", "X", "Y", "Z"], "output": [1, -1, -1, -1]}

    for case in [case00, case01, case02]:
        returnedList = convertBenchmarkClasses_problemSizeInNPB(inputList=case["input"])
        assert returnedList == case["output"]


# return_numOfColumns()でのカラム名としてのモデル名、モデルのメソッドModelName()が返すモデル名を相互的なキー・バリューとした辞書を返す関数
def returnDictModelNames():
    returnDict = {}
    # カラム名をキー・モデルが返すモデル名をバリュー
    returnDict["objectLinModel"] = "ModelLin"
    returnDict["objectIpModel"] = "ModelIp"
    returnDict["objectLogModel"] = "ModelLog"
    returnDict["objectBranchModel"] = "ModelBranch"
    returnDict["objectBranchModel2"] = "ModelBranch2"
    # モデルが返すモデル名をキー・カラム名をバリュー
    returnDict["ModelLin"] = "objectLinModel"
    returnDict["ModelIp"] = "objectIpModel"
    returnDict["ModelLog"] = "objectLogModel"
    returnDict["ModelBranch"] = "objectBranchModel"
    returnDict["ModelBranch2"] = "objectBranchModel2"

    return returnDict


# In[17]:


# 結果を集計するためのDFに挿入するSeriesを作成する関数
def returnSeriesOfData(
    benchmarkName="benhmarkName",
    functionName="functionName",
    rawX=[1, 2, 3],
    rawY=[1, 2, 3],
    fixProcessOrClass="Class",
    fixed="B",
    targetProcess=256,
    targetBenchmarkClass="B",
    targetFunctionCallNum=-1,
    csvDirPath="./csv_files",
    modelNames=["ModelLin", "ModelIp", "ModelLog", "ModelBranch"],
):
    dataSeries = pd.Series(
        index=returnNumOfColumns(modelNames=modelNames), dtype=object
    )
    dataSeries["benchmarkName"] = benchmarkName
    dataSeries["functionName"] = functionName
    dataSeries["usedDataX"] = rawX
    dataSeries["usedDataY"] = rawY
    dataSeries["numOfData"] = len(rawX)
    dataSeries["ProcessOrClass"] = fixProcessOrClass
    dataSeries["fixed"] = fixed
    dataSeries["targetProcess"] = targetProcess
    dataSeries["targetProblemSize"] = targetBenchmarkClass
    if targetFunctionCallNum < 0:
        dataSeries["targetNumOfFunctionCall"] = returnSpecificDataFromCSV(
            benchmarkName=benchmarkName,
            functionName=functionName,
            process=targetProcess,
            benchmarkClass=targetBenchmarkClass,
            csvDirPath=csvDirPath,
        )
    else:
        dataSeries["targetNumOfFunctionCall"] = targetFunctionCallNum
    #     # MAPE の算出には returnMapeScore()を用いる
    #     # returnMapeScore()の返り値の単位は％

    # 線形モデル
    if "ModelLin" in modelNames:
        modelLin = ModelLin(trainX=rawX, trainY=rawY)
        modelLin.calcLr()
        predictedY = modelLin.predict(rawX)
        dataSeries["objectLinModel"] = modelLin
        dataSeries["MAPEOfLinModel"] = returnMapeScore(predictedY, rawY)
    # 反比例モデル
    if "ModelIp" in modelNames:
        modelIp = ModelIp(trainX=rawX, trainY=rawY)
        modelIp.calcLr()
        predictedY = modelIp.predict(rawX)
        dataSeries["objectIpModel"] = modelIp
        dataSeries["MAPEOfIpModel"] = returnMapeScore(predictedY, rawY)
    # 対数モデル
    if "ModelLog" in modelNames:
        modelLog = ModelLog10(trainX=rawX, trainY=rawY)
        modelLog.calcLr()
        predictedY = modelLog.predict(rawX)
        dataSeries["objectLogModel"] = modelLog
        dataSeries["MAPEOfLogModel"] = returnMapeScore(predictedY, rawY)
    # 分岐モデル
    if "ModelBranch" in modelNames:
        modelBranch = ModelBranch(trainX=rawX, trainY=rawY)
        modelBranch.calcLr()
        predictedY = modelBranch.predict(rawX)
        dataSeries["objectBranchModel"] = modelBranch
        dataSeries["MAPEOfBranchModel"] = returnMapeScore(predictedY, rawY)
    # 分岐モデル2
    # if "ModelBranch2" in modelNames:
    #     modelBranch2 = ModelBranch2(trainX=rawX, trainY=rawY)
    #     modelBranch2.calcLr()
    #     predictedY = modelBranch2.predict(rawX)
    #     dataSeries["objectBranchModel2"] = modelBranch2
    #     dataSeries["MAPEOfBranchModel2"] = returnMapeScore(predictedY, rawY)

    # 最適なモデルのモデルのモデル名・MAPE値の算出
    listToCalcBestModel = {}
    # 線形モデル
    if "ModelLin" in modelNames:
        listToCalcBestModel[dataSeries["objectLinModel"].ModelName()] = dataSeries[
            "MAPEOfLinModel"
        ]
    # 反比例モデル
    if "ModelIp" in modelNames:
        listToCalcBestModel[dataSeries["objectIpModel"].ModelName()] = dataSeries[
            "MAPEOfIpModel"
        ]
    # 対数モデル
    if "ModelLog" in modelNames:
        listToCalcBestModel[dataSeries["objectLogModel"].ModelName()] = dataSeries[
            "MAPEOfLogModel"
        ]
    # 線形飽和モデル
    if "ModelBranch" in modelNames:
        listToCalcBestModel[dataSeries["objectBranchModel"].ModelName()] = dataSeries[
            "MAPEOfBranchModel"
        ]
    # 線形飽和モデル2
    if "ModelBranch2" in modelNames:
        listToCalcBestModel[dataSeries["objectBranchModel2"].ModelName()] = dataSeries[
            "MAPEOfBranchModel2"
        ]

    minMAPE = min(listToCalcBestModel.values())
    dataSeries["MAPEOfBestModel"] = minMAPE
    dataSeries["objectBestModelName"] = [
        k for k, v in listToCalcBestModel.items() if v == minMAPE
    ][0]
    dictOfModelNames = returnDictModelNames()
    bestModelName = dataSeries["objectBestModelName"]
    bestModelColumnName = dictOfModelNames[bestModelName]
    # 目標関数コール回数に対する、絶対相対誤差率
    # 実データ
    realData = targetFunctionCallNum
    # 予測データ
    predictedData = -1
    # 最適モデルで予測を実施
    convertDict = returnDictModelNames()
    bestModelObjct = dataSeries[convertDict[bestModelName]]
    if fixProcessOrClass == "Class":
        targetX = targetProcess
    else:
        targetX = convertStrToInt_problemSizeInNPB(targetBenchmarkClass)

    predictedData = bestModelObjct.predict(targetX)[0][0]

    dataSeries["RelativeErrorRate"] = abs(realData - predictedData) / (realData) * 100

    return dataSeries


@pytest.fixture()
def test_generateCSVFilesForReturnSeriesOfData():
    filePath = "/tmp/pprof_testD256.csv"
    functionName = "testFunctionName"
    with open(filePath, "w") as f:
        f.write("Name,#Call\n")
        # 本来は各モデルごとに最適な関数コール回数とするべきだが、できないので-1を返すようにした
        f.write(f"{functionName},-1\n")


def test_returnSeriesOfData(test_generateCSVFilesForReturnSeriesOfData):
    # 共通部分の設定
    benchmarkName = "test"
    functionName = "testFunctionName"
    targetProcess = 256
    targetBenchmarkClass = "D"
    fixProcessOrClass = "Class"
    fix = targetBenchmarkClass

    csvDirPathForTest = "/tmp"

    # モデル名のリストを作成
    modelNames = ["ModelLin", "ModelIp", "ModelLog", "ModelBranch"]

    explanatoryVariableX = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    responseVariableY = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

    # 分岐モデルが最適となる場合
    # 目的変数を分岐モデルが最適となるように設定する
    responseVariableY = [48, 52, 60, 76, 108, 172, 300, 300, 300]
    branchSeries = returnSeriesOfData(
        benchmarkName=benchmarkName,
        functionName=f"{functionName}",
        rawX=explanatoryVariableX,
        rawY=responseVariableY,
        fixProcessOrClass=fixProcessOrClass,
        fixed=targetBenchmarkClass,
        targetProcess=targetProcess,
        targetBenchmarkClass=targetBenchmarkClass,
        targetFunctionCallNum=responseVariableY[-1],
        csvDirPath=csvDirPathForTest,
        modelNames=modelNames,
    )
    # 最適なモデルが分岐モデルであることを確認
    assert branchSeries["objectBestModelName"] == "ModelBranch"
    # 説明変数に対するMAPEが最小のモデルを用いて予測対象の関数コール回数を予測した時の絶対相対誤差率が非常に小さいことを確認

    # 反比例モデルが最適となる場合
    # 目的変数を反比例モデルが最適となるように設定する
    responseVariableY = 2 / explanatoryVariableX + 3
    ipSeries = returnSeriesOfData(
        benchmarkName=benchmarkName,
        functionName=f"{functionName}",
        rawX=explanatoryVariableX,
        rawY=responseVariableY,
        fixProcessOrClass=fixProcessOrClass,
        fixed=targetBenchmarkClass,
        targetProcess=targetProcess,
        targetBenchmarkClass=targetBenchmarkClass,
        csvDirPath=csvDirPathForTest,
        modelNames=modelNames,
    )
    # 最適なモデルが反比例モデルであることを確認
    assert ipSeries["objectBestModelName"] == "ModelIp"
    # 説明変数に対するMAPEが最小のモデルを用いて予測対象の関数コール回数を予測した時の絶対相対誤差率が非常に小さいことを確認

    # 線形モデルが最適となる場合
    # 目的変数を線形モデルが最適となるように設定する
    responseVariableY = 2 * explanatoryVariableX + 3
    linSeries = returnSeriesOfData(
        benchmarkName=benchmarkName,
        functionName=f"{functionName}",
        rawX=explanatoryVariableX,
        rawY=responseVariableY,
        fixProcessOrClass=fixProcessOrClass,
        fixed=targetBenchmarkClass,
        targetProcess=targetProcess,
        targetBenchmarkClass=targetBenchmarkClass,
        csvDirPath=csvDirPathForTest,
        modelNames=modelNames,
    )
    # 最適なモデルが線形モデルであることを確認
    assert linSeries["objectBestModelName"] == "ModelLin"
    # 説明変数に対するMAPEが最小のモデルを用いて予測対象の関数コール回数を予測した時の絶対相対誤差率が非常に小さいことを確認

    # 対数モデルが最適となる場合
    # 目的変数を対数モデルが最適となるように設定する
    responseVariableY = 2 * np.log10(explanatoryVariableX) + 3
    logSeries = returnSeriesOfData(
        benchmarkName=benchmarkName,
        functionName=f"{functionName}",
        rawX=explanatoryVariableX,
        rawY=responseVariableY,
        fixProcessOrClass=fixProcessOrClass,
        fixed=targetBenchmarkClass,
        targetProcess=targetProcess,
        targetBenchmarkClass=targetBenchmarkClass,
        csvDirPath=csvDirPathForTest,
        modelNames=modelNames,
    )
    # 最適なモデルが対数モデルであることを確認
    assert logSeries["objectBestModelName"] == "ModelLog"
    # 説明変数に対するMAPEが最小のモデルを用いて予測対象の関数コール回数を予測した時の絶対相対誤差率が非常に小さいことを確認


# In[18]:


@pytest.fixture()
def test_generateAllBranchFunctionCSVData():
    benchmarkName = "branch"
    fileNamePrefix = f"pprof_{benchmarkName}"
    classes = ["A", "B", "C", "D"]
    explanatoryVariableX = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    responseVariableY = [48, 52, 60, 76, 108, 172, 300, 300, 300]
    functionNames = []
    for i in range(4):
        functionNames.append(f"{benchmarkName}_0{i}")
    for benchmarkClass in classes:
        for process in explanatoryVariableX:
            fileName = f"{fileNamePrefix}{benchmarkClass}{process}.csv"
            filePath = f"/tmp/{fileName}"
            with open(filePath, "w") as f:
                f.write("Name,#Call\n")
                for functionName in functionNames:
                    functionCallNum = responseVariableY[
                        explanatoryVariableX.index(process)
                    ]
                    f.write(f"{functionName},{functionCallNum}")


# In[19]:


# 論文などに載せる集計結果を作成するために用いるDFを作成するための関数


def returnDFSummarizedData(
    benchmarkNames=["cg", "ep", "ft", "is", "lu", "mg"],
    classes=["A", "B", "C", "D"],
    processes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    targetIndex=-1,
    csvDirPath="./csv_files/",
    modelNames=["ModelLin", "ModelIp", "ModelLog", "ModelBranch"],
):
    listOfSeriesData = []
    for benchmarkName in benchmarkNames:
        dfPerBenchmark = returnCollectedExistingData(
            benchmarkNames=[benchmarkName],
            classes=classes,
            processes=processes,
            csvDirPath=csvDirPath,
        )
        for benchmarkClass in classes:
            dfPerBenchmarkClass = dfPerBenchmark[
                dfPerBenchmark["benchmarkClass"] == benchmarkClass
            ]
            functionNames = sorted(list(set(dfPerBenchmarkClass["functionName"])))
            for functionName in functionNames:
                dfPerFunction = dfPerBenchmarkClass[
                    dfPerBenchmarkClass["functionName"] == functionName
                ]

                # 説明変数と目的変数とをリスト化したものを抽出
                # プロセス数
                rawX = dfPerFunction["process"].tolist()
                # 関数コール回数
                rawY = dfPerFunction["functionCallNum"].tolist()
                # 引数として渡されたプロセス数未満の関数を除外する
                if len(rawX) != len(processes) or len(rawY) != len(processes):
                    continue

                # 説明変数のリストと目的変数のリストをモデル構築用・モデル試験用に分割
                trainX = rawX[:targetIndex]
                trainY = rawY[:targetIndex]
                targetX = rawX[targetIndex:]
                targetY = rawY[targetIndex:]

                # 説明変数のリスト・目的変数のリストが長さ0で渡される場合があり、それによるエラーを回避するための例外処理
                if (
                    len(trainX) == 0
                    or len(trainY) == 0
                    or len(targetX) == 0
                    or len(targetY) == 0
                ):
                    continue
                seriesPerFunction = returnSeriesOfData(
                    benchmarkName=benchmarkName,
                    functionName=functionName,
                    rawX=trainX,
                    rawY=trainY,
                    fixProcessOrClass="Class",
                    fixed=benchmarkClass,
                    targetProcess=targetX[0],
                    targetBenchmarkClass=benchmarkClass,
                    targetFunctionCallNum=targetY[0],
                    csvDirPath=csvDirPath,
                    modelNames=modelNames,
                )
                listOfSeriesData.append(seriesPerFunction)
    returnDF = pd.concat(listOfSeriesData, axis=1).T
    return returnDF


def test_returnDFSummarizedData():
    test_benchmarkNames = ["cg", "ep", "ft", "is", "lu", "mg"]
    test_classes = ["A", "B", "C", "D"]
    test_processes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    test_targetIndex = -1
    # テストデータを作成する時間がないので、利用可能な既存のすべての実データを利用する
    test_csvDirPath = "../csv_files/"
    test_DF = returnDFSummarizedData(
        benchmarkNames=test_benchmarkNames,
        classes=test_classes,
        processes=test_processes,
        targetIndex=test_targetIndex,
        csvDirPath=test_csvDirPath,
    )
    for benchmarkName in test_benchmarkNames:
        test_DFPerBenchmarkName = test_DF[test_DF["benchmarkName"] == benchmarkName]
        for benchmarkClass in test_classes:
            test_DFPerBenchmarkNamePerBenchmarkClass = test_DFPerBenchmarkName[
                test_DFPerBenchmarkName["fixed"] == benchmarkClass
            ]
            # [processesの要素数-1] と モデルの構築に使用されたデータ数が同じ
            for index in test_DFPerBenchmarkNamePerBenchmarkClass.index.tolist():
                assert (
                    test_DFPerBenchmarkNamePerBenchmarkClass.at[index, "usedDataX"]
                    == test_processes[:test_targetIndex]
                )
            # targetProcessとfixedが同じ、もしくは、targetProblemSizeとfixedが同じ
            assert (
                len(
                    test_DFPerBenchmarkNamePerBenchmarkClass[
                        test_DFPerBenchmarkNamePerBenchmarkClass["targetProcess"]
                        == test_DFPerBenchmarkNamePerBenchmarkClass["fixed"]
                    ].index
                )
                == 0
            )


# In[20]:


# 入力：returnSeriesOfDataを結合したDF（含むベンチマークの種類は1つ）
# 出力：各モデルの採用割合が入ったSeries
def returnSeriesOfDatumPerBenchmark(
    inputDF, modelNames=["ModelIp", "ModelLog", "ModelBranch", "ModelLin"]
):
    # 全データが単一のベンチマークによるものかを確認し、そうでなければ警告を出力する
    listOfBenchmarkNameInInputDF = inputDF["benchmarkName"].tolist()
    noDuplicateBenchmarkName = list(set(listOfBenchmarkNameInInputDF))
    if len(noDuplicateBenchmarkName) != 1:
        warnings.warn("入力DFには複数のベンチマークの結果が含まれています")
    benchmarkName = noDuplicateBenchmarkName[0]
    numOfInputDF = len(inputDF)
    _functionNames = inputDF["functionName"].tolist()
    contentList = [f"{benchmarkName.upper()}({len(_functionNames)})"]
    summarizedRateExcludeModelLin = 0
    for modelName in modelNames:
        dfOfModel = inputDF[inputDF["objectBestModelName"] == modelName]
        numOfModel = len(dfOfModel)
        rateOfModel = int(numOfModel / numOfInputDF * 100)

        try:
            maxInDfOfModel = int(dfOfModel["MAPEOfBestModel"].max() * 10) / 10
        except:
            maxInDfOfModel = "-"

        try:
            minInDfOfModel = int(dfOfModel["MAPEOfBestModel"].min() * 10) / 10
        except:
            minInDfOfModel = "-"

        if modelName != "ModelLin":
            summarizedRateExcludeModelLin += rateOfModel
        elif modelName == "ModelLin":
            rateOfModel = 100 - summarizedRateExcludeModelLin

        instanceDatumAboutRateOfModel = DatumAboutRateOfModel(
            modelName=modelName,
            rateOfModel=rateOfModel,
            minMAPE=minInDfOfModel,
            maxMAPE=maxInDfOfModel,
        )
        contentList.append(instanceDatumAboutRateOfModel.returnFormattedStr())
    columnList = ["benchmarkName"] + modelNames
    returnSeries = pd.Series(data=contentList, index=columnList)
    return returnSeries


class DatumAboutRateOfModel:
    def __init__(self, modelName, rateOfModel, minMAPE, maxMAPE):
        # モデル名
        self.modelName = modelName
        # モデルの被採用率
        self.rateOfModel = rateOfModel
        # モデルの最小MAPE・最大MAPE
        self.minMAPE = minMAPE
        self.maxMAPE = maxMAPE

    def returnFormattedStr(self):
        if self.maxMAPE == "-" or self.minMAPE == "-":
            strMinMax = "-"
        else:
            strMinMax = f"{self.minMAPE},{self.maxMAPE}"
        returnStr = f"{self.rateOfModel}({strMinMax})"
        return returnStr


def test_returnSeriesOfDatumPerBenchmark():
    # テストについて
    pass
    # 各モデルがそれぞれカウントされている
    # 線形飽和モデル
    plotXForBranch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    plotYForBranch = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 23, 23, 23, 23]
    seriesBranch = returnSeriesOfData(
        benchmarkName="test",
        functionName="modelBranch",
        rawX=plotXForBranch,
        rawY=plotYForBranch,
        fixProcessOrClass="Class",
        fixed="B",
        targetProcess=256,
        targetBenchmarkClass="B",
        targetFunctionCallNum=plotYForBranch[-1],
        csvDirPath="./csv_files",
    )
    csvDirPath = "../csv_files"
    # 反比例モデル・線形モデル・対数モデルで共通の説明変数
    plotX = np.linspace(0.5, 256, 500)
    # 3モデルで共通な説明変数が256のときのインデックス値
    indexAt256Core = -1
    # 反比例モデル
    plotYForIp = 2 / plotX + 3
    seriesIp = returnSeriesOfData(
        benchmarkName="test",
        functionName="modelIp",
        rawX=plotX,
        rawY=plotYForIp,
        fixProcessOrClass="Class",
        fixed="B",
        targetProcess=256,
        targetBenchmarkClass="B",
        targetFunctionCallNum=plotYForIp[-1],
        csvDirPath=csvDirPath,
    )
    # 線形モデル
    plotYForLin = 2 * plotX + 3
    seriesLin = returnSeriesOfData(
        benchmarkName="test",
        functionName="modelLin",
        rawX=plotX,
        rawY=plotYForLin,
        fixProcessOrClass="Class",
        fixed="B",
        targetProcess=256,
        targetBenchmarkClass="B",
        targetFunctionCallNum=plotYForLin[-1],
        csvDirPath=csvDirPath,
    )
    # 対数モデル
    plotYForLog = 2 * np.log10(plotX) + 3
    seriesLog = returnSeriesOfData(
        benchmarkName="test",
        functionName="modelLog",
        rawX=plotX,
        rawY=plotYForLog,
        fixProcessOrClass="Class",
        fixed="B",
        targetProcess=256,
        targetBenchmarkClass="B",
        targetFunctionCallNum=plotYForLog[-1],
        csvDirPath=csvDirPath,
    )

    # テスト対象となる関数の引数となるDF
    inputDF = pd.concat([seriesBranch, seriesIp, seriesLin, seriesLog], axis=1).T
    # テスト対象となる関数の返り値
    result = returnSeriesOfDatumPerBenchmark(inputDF=inputDF)
    # 4つのモデルが最適となるDFをテストデータとするのでそれぞれ25%であることを確認
    dictResult = result.to_dict()
    for benchmarkName in ["ModelIp", "ModelLog", "ModelBranch", "ModelLin"]:
        assert dictResult[benchmarkName][:2] == "25"


# In[21]:


# 相対誤差率を返す関数
# realNum：真値
# predictedNum：予測値
# decimalPlace：少数第n位までにするか


# 返り値は相対誤差率[%]
def returnRelativeErrorRate(realNum=1, predictedNum=1, decimalPlace=3):
    if realNum == 0:
        warnings.warn("真値が0です")
        return -1
    diff = realNum - predictedNum
    relativeErrorRate = abs(diff / realNum) * 100
    roundedRelativeErrorRate = np.round(relativeErrorRate, decimalPlace)
    return roundedRelativeErrorRate


def test_returnRelativeErrorRate():
    case00 = returnRelativeErrorRate(realNum=1, predictedNum=1, decimalPlace=3)
    assert -0.01 < case00 < 0.01
    case01 = returnRelativeErrorRate(realNum=1, predictedNum=100)
    assert 9890.0 < case01 < 9910.0
    case02 = returnRelativeErrorRate(realNum=3, predictedNum=4, decimalPlace=2)
    assert 32.90 < case02 < 33.34
    with pytest.warns(None):
        case03 = returnRelativeErrorRate(realNum=0, predictedNum=0)
    assert case03 == -1


# In[22]:


# Multiple regression analysis （重回帰分析）


# class baseModelForMultipleRegression
# 重回帰分析用のモデルの共通部分となるクラス
# 引数名とその説明
# inputDF：入力データの全てを保持したDF（説明変数・目的変数・ベンチマーク名・関数名を最低限保持している）
# explanatoryVariableColumnNames：inputDFの列名の中で、説明変数として用いるカラム名のリスト
# responseVariableColumnNames：inputDFの列名の中で、説明変数として用いるカラム名のリスト
# conditionDictForTest："カラム名":"要素"でテスト用データを指定する
# targetDF：inputDFとデータ構成は同じだが、予測対象のデータがセットされている
class ModelBaseForMultipleRegression:
    """ModelBaseForMultipleRegression

    複数の説明変数を用いた予測を行うにあたってベースとなるクラス

    Attributes:
        explanatoryVariableColumnNames :list[str] 説明変数のカラム名のリスト
        responseVariableColumnNames :list[str]    目的変数のカラム名のリスト
        rawExplanaoryVariable :pd.DataFrame       説明変数のデータフレーム
        rawResponseVariable :pd.DataFrame         目的変数のデータフレーム
    Note:
    """

    def __init__(
        self,
        inputDF: pd.DataFrame,
        explanatoryVariableColumnNames: list[str],
        responseVariableColumnNames: list[str],
        conditionDictForTest: Dict[str, str] = {},
        targetDF: pd.DataFrame = None,
    ):
        # 関数名が複数種類ある場合は警告
        functionName = set(inputDF["functionName"].tolist())
        if len(functionName) != 1:
            warnings.warn(f"関数が複数種類存在します\nfunctionName={functionName}")

        # 各種カラム名を保持
        self.explanatoryVariableColumnNames = explanatoryVariableColumnNames
        self.responseVariableColumnNames = responseVariableColumnNames

        # テスト用とモデル構築用にデータを分割する
        # テスト用
        dfForTestingModel = inputDF
        # モデル構築用DF
        dfForBuildingModel = inputDF
        if len(conditionDictForTest) != 0:
            for keys in conditionDictForTest.keys():
                dfForTestingModel = dfForTestingModel[
                    dfForTestingModel[keys] == conditionDictForTest[keys]
                ]
                dfForBuildingModel = dfForBuildingModel[
                    dfForBuildingModel[keys] != conditionDictForTest[keys]
                ]

        # self.rawExplanatoryVariableをセット
        self.rawExplanaoryVariable = dfForBuildingModel[explanatoryVariableColumnNames]
        # self.rawResponseVariableをセット
        self.rawResponseVariable = dfForBuildingModel[responseVariableColumnNames]
        # self.rawExplanatoryVariableForTestをセット
        self.rawExplanaoryVariableForTest = dfForTestingModel[
            explanatoryVariableColumnNames
        ]
        # self.rawResponseVariableForTestをセット
        self.rawResponseVariableForTest = dfForTestingModel[responseVariableColumnNames]


class ModelLinForMultipleRegression(ModelBaseForMultipleRegression):
    # 線形モデル（重回帰分析）

    def transformDataForModel(self, inputDF):
        # inputDFで与えられたデータをモデルに適した形に変形する
        return inputDF

    def setUpDataBeforeCalcLr(self):
        # 説明変数・目的変数を変換する関数
        # モデル構築用データ
        self.dataXForPredict = self.transformDataForModel(self.rawExplanaoryVariable)
        self.dataTForPredict = self.transformDataForModel(self.rawResponseVariable)
        # テスト用データ
        self.dataXForTest = self.transformDataForModel(
            self.rawExplanaoryVariableForTest
        )
        self.dataTForTest = self.transformDataForModel(self.rawResponseVariableForTest)

    def calcLr(self):
        # 実際にモデルを構築する
        self.lr = LinearRegression()
        self.lr.fit(self.dataXForPredict, self.dataTForPredict)

    def predict(self, inputDF):
        # inputDFのデータから構築されたモデルを使って予測を行う

        # inputDFから説明変数データのみを取得
        inputDFOnlyExplanatoryVariableColumn = inputDF[
            self.explanatoryVariableColumnNames
        ]
        # 予測を実行
        result = self.lr.predict(inputDFOnlyExplanatoryVariableColumn)

        return result


def test_ModelLinForMultipleRegression():
    # 説明変数
    plotX = np.linspace(0, 20, 10)
    plotY = np.linspace(20, 40, 10)
    plotZ = np.linspace(40, 60, 10)
    # 目的変数
    plotT = plotX + 2 * plotY + 3 * plotZ + 4

    # DFを作成する
    # カラム名のリスト
    columnNames = ["plotX", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = columnNames[:-1]
    # 説明変数のカラム名のリスト
    columnNamesForRes = columnNames[-1:]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = ModelLinForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの生成の準備
    objectModel.setUpDataBeforeCalcLr()
    # モデルの生成
    objectModel.calcLr()
    # モデルによる予測
    # 入力データDFを作成
    inputDFForPredict = pd.DataFrame(inputDFForTest.tail(1))
    predictedNum = objectModel.predict(inputDFForPredict)

    # 相対誤差率でテスト対象のデータが想定通りに動作しているかを判断する
    # 相対誤差率を計算するために実データを取得する
    realNum = plotT[-1]
    relativeErrorRate = returnRelativeErrorRate(
        realNum=realNum, predictedNum=predictedNum
    )

    assert relativeErrorRate < 1


# In[23]:


class ModelIpForMultipleRegression(ModelBaseForMultipleRegression):
    # 反比例モデル（重回帰分析）

    def ipFunc(x):
        return 1 / x

    def transformDataForModel(self, inputDF):
        # inputDFで与えられたデータをモデルに適した形に変形する
        returnDF = self.transformerIp.transform(inputDF)
        return returnDF

    def setUpDataBeforeCalcLr(self):
        # 説明変数・目的変数を変換する関数
        self.transformerIp = sp.FunctionTransformer(func=ipFunc, inverse_func=ipFunc)
        # モデル構築用データ
        self.dataXForPredict = self.transformDataForModel(self.rawExplanaoryVariable)
        #         self.dataTForPredict = self.transformDataForModel(
        #             self.rawResponseVariable)
        self.dataTForPredict = self.rawResponseVariable
        # テスト用データ
        self.dataXForTest = self.transformDataForModel(
            self.rawExplanaoryVariableForTest
        )
        #         self.dataTForTest = self.transformDataForModel(
        #             self.rawResponseVariableForTest)
        self.dataTForTest = self.rawResponseVariableForTest

    def calcLr(self):
        # 実際にモデルを構築する
        self.lr = LinearRegression()
        self.lr.fit(self.dataXForPredict, self.dataTForPredict)

    def predict(self, inputDF):
        # inputDFのデータから構築されたモデルを使って予測を行う

        # inputDFから説明変数データのみを取得
        inputDFOnlyExplanatoryVariableColumn = inputDF[
            self.explanatoryVariableColumnNames
        ]
        # inputDFで与えられたデータをモデルに適した形に変形する
        transformedInputDF = self.transformDataForModel(
            inputDFOnlyExplanatoryVariableColumn
        )
        # 予測を実行
        result = self.lr.predict(transformedInputDF)

        return result

    pass


def test_ModelIpForMultipleRegression():
    # 説明変数
    plotX = np.linspace(1, 21, 10)
    plotY = np.linspace(20, 40, 10)
    plotZ = np.linspace(40, 60, 10)
    # 目的変数
    plotT = 1 / plotX + 2 / plotY + 3 / plotZ + 4

    # DFを作成する
    # カラム名のリスト
    columnNames = ["plotX", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = columnNames[:-1]
    # 説明変数のカラム名のリスト
    columnNamesForRes = columnNames[-1:]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = ModelIpForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの生成の準備
    objectModel.setUpDataBeforeCalcLr()
    # モデルの生成
    objectModel.calcLr()
    # モデルによる予測
    # 入力データDFを作成
    inputDFForPredict = pd.DataFrame(inputDFForTest.tail(1))
    predictedNum = objectModel.predict(inputDFForPredict)

    # 相対誤差率でテスト対象のデータが想定通りに動作しているかを判断する
    # 相対誤差率を計算するために実データを取得する
    realNum = plotT[-1]
    relativeErrorRate = returnRelativeErrorRate(
        realNum=realNum, predictedNum=predictedNum
    )

    assert relativeErrorRate < 1


class ModelLogForMultipleRegression(ModelBaseForMultipleRegression):
    # 対数モデル（重回帰分析）

    def inverterLog10Func(x):
        return 10**x

    def transformDataForModel(self, inputDF):
        # inputDFで与えられたデータをモデルに適した形に変形する
        returnDF = self.transformerLog10.transform(inputDF)
        return returnDF

    def setUpDataBeforeCalcLr(self):
        # 説明変数・目的変数を変換する関数
        self.transformerLog10 = sp.FunctionTransformer(
            inverse_func=self.inverterLog10Func, func=np.log10
        )
        # モデル構築用データ
        self.dataXForPredict = self.transformDataForModel(self.rawExplanaoryVariable)
        self.dataTForPredict = self.rawResponseVariable
        #         self.dataTForPredict = self.transformDataForModel(
        #             self.rawResponseVariable)
        # テスト用データ
        self.dataXForTest = self.transformDataForModel(
            self.rawExplanaoryVariableForTest
        )
        self.dataTForTest = self.rawResponseVariableForTest

    #         self.dataTForTest = self.transformDataForModel(
    #             self.rawResponseVariableForTest)

    def calcLr(self):
        # 実際にモデルを構築する
        self.lr = LinearRegression()
        self.lr.fit(self.dataXForPredict, self.dataTForPredict)

    def predict(self, inputDF):
        # inputDFのデータから構築されたモデルを使って予測を行う

        # inputDFから説明変数データのみを取得
        inputDFOnlyExplanatoryVariableColumn = inputDF[
            self.explanatoryVariableColumnNames
        ]
        # inputDFで与えられたデータをモデルに適した形に変形する
        transformedInputDF = self.transformDataForModel(
            inputDFOnlyExplanatoryVariableColumn
        )
        # 予測を実行
        result = self.lr.predict(transformedInputDF)

        return result

    pass


def test_ModelLogForMultipleRegression():
    # 説明変数
    plotX = np.linspace(1, 21, 10)
    plotY = np.linspace(20, 40, 10)
    plotZ = np.linspace(40, 60, 10)
    # 目的変数
    plotT = 1 * np.log10(plotX) + 2 * np.log10(plotY) + 3 * np.log10(plotZ) + 4

    # DFを作成する
    # カラム名のリスト
    columnNames = ["plotX", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = columnNames[:-1]
    # 説明変数のカラム名のリスト
    columnNamesForRes = columnNames[-1:]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = ModelLogForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの生成の準備
    objectModel.setUpDataBeforeCalcLr()
    # モデルの生成
    objectModel.calcLr()
    # モデルによる予測
    # 入力データDFを作成
    inputDFForPredict = pd.DataFrame(inputDFForTest.tail(1))
    predictedNum = objectModel.predict(inputDFForPredict)

    # 相対誤差率でテスト対象のデータが想定通りに動作しているかを判断する
    # 相対誤差率を計算するために実データを取得する
    realNum = plotT[-1]
    relativeErrorRate = returnRelativeErrorRate(
        realNum=realNum, predictedNum=predictedNum
    )
    assert relativeErrorRate < 1


# In[24]:


def returnDFwithFunctionsExecUnderAllConditions(
    inputDF, classes=["A", "B", "C"], processes=[2, 4, 8, 16, 32, 64, 128]
):
    # DFを渡されて processes, classes の全ての条件で実行された関数のみ保持されたDFを返す関数
    # inputDF：入力DF。returnCollectedExistingData()の返り値を想定している
    # classes：条件１（ベンチマーククラス）のリスト
    # processes：条件２（プロセス数）のリスト
    # 重複のない関数名のリスト
    functionNames = list(set(inputDF["functionName"]))
    # データ数が全てのパターンにある関数のリスト
    functionNamesAllPattern = []

    # 返り値となるDF
    returnDF = pd.DataFrame()

    for functionName in functionNames:
        DFPerFunction = inputDF[inputDF["functionName"] == functionName]
        if len(classes) * len(processes) == len(DFPerFunction):
            returnDF = pd.concat([returnDF, DFPerFunction], axis=0)
            functionNamesAllPattern.append(functionName)

    if len(functionNamesAllPattern) == 0:
        warnings.warn("対象となる関数がありません")

    return returnDF


@pytest.fixture()
def test_generateCSVDataForReturnDFwithFunctionsExecUnderAllConditions():
    benchmarkName = "benchmarkName"
    fileNamePrefix = f"pprof_{benchmarkName}"
    classes = ["A", "B", "C", "D"]
    processes = [2, 4, 8, 16]

    functionNames = [
        "functionValid00",
        "functionValid01",
        "functionInvalid00",
        "functionInvalid01",
    ]

    for benchmarkClass in classes:
        for process in processes:
            fileName = f"{fileNamePrefix}{benchmarkClass}{process}.csv"
            filePath = f"/tmp/{fileName}"
            with open(filePath, "w") as f:
                f.write("Name,#Call\n")
                for functionName in functionNames:
                    functionCallNum = 1
                    if ("Invalid" in functionName) and (process > 4):
                        pass
                    else:
                        f.write(f"{functionName},{functionCallNum}\n")


def test_returnDFwithFunctionsExecUnderAllConditions(
    test_generateCSVDataForReturnDFwithFunctionsExecUnderAllConditions,
):
    classes = ["A", "B", "C", "D"]
    processes = [2, 4, 8, 16]

    benchmarkName = "benchmarkName"

    rawDF = returnCollectedExistingData(
        benchmarkNames=[benchmarkName],
        classes=classes,
        processes=processes,
        csvDirPath="/tmp/",
    )
    returnedDF = returnDFwithFunctionsExecUnderAllConditions(
        inputDF=rawDF, classes=classes, processes=processes
    )

    functionNamesInDF = list(set(returnedDF["functionName"].tolist()))

    assert ("functionValid00" in functionNamesInDF) == True
    assert ("functionValid01" in functionNamesInDF) == True
    assert ("functionInvalid00" in functionNamesInDF) == False
    assert ("functionInvalid01" in functionNamesInDF) == False


# In[25]:


# 最終的な集計に必要な情報を保持したDFのカラム名のリストもしくは各カラムのデータタイプを返す関数
def returnListAboutInformation(dataType=False):
    returnListColumnDataType = []
    returnListColumnName = []

    # 関数名
    returnListColumnDataType.append("functionName")
    returnListColumnName.append(str)

    # ベンチマーク名
    returnListColumnDataType.append("BenchmarkName")
    returnListColumnName.append(str)

    # 線形モデル
    returnListColumnDataType.append("modelLin")
    returnListColumnName.append(object)

    # 反比例モデル
    returnListColumnDataType.append("modelLin")
    returnListColumnName.append(object)

    # 対数モデル
    returnListColumnDataType.append("modelLin")
    returnListColumnName.append(object)

    if dataType:
        return returnListColumnDataType
    else:
        return returnListColumnName


def test_returnListAboutInformation():
    NumOfColumns = 5

    dataType = returnListAboutInformation(dataType=True)
    name = returnListAboutInformation(dataType=False)

    assert len(dataType) == NumOfColumns and len(name) == NumOfColumns


# In[ ]:


# In[26]:


# 引数として渡されたDFから、関数ごとに「関数名 | ベンチマーク名 | 説明変数 | 目的変数 | 集計結果」を保持したDFを作成する関数
# 引数として渡されたDFにはベンチマーク・関数はそれぞれ１種類のデータが格納されている
# 単回帰分析のため、説明変数は1種類のみ
def returnDFtoMakeSummary(
    inputDF,
    benchmarkName="benchmarkName",
    validFunctionName="validFunctionName",
    targetClass="D",
    targetProcess=256,
    expVarColNames=[],
    resVarColNames=[],
):
    if len(expVarColNames) == 0:
        warnings.warn("説明変数のカラム名を保持したリストが空です")
    if len(resVarColNames) == 0:
        warnings.warn("目的変数のカラム名を保持したリストが空です")
    # モデルを一括で作成
    targetDF = inputDF[
        (inputDF["benchmarkClass"] == targetClass)
        & (inputDF["process"] == targetProcess)
    ]
    dropIndex = inputDF.index[
        (inputDF["benchmarkClass"] == targetClass)
        | (inputDF["process"] == targetProcess)
    ]
    droppedInputDF = inputDF.drop(dropIndex)
    models = Models(
        inputDF=droppedInputDF,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
        targetDF=targetDF,
    )
    # 学習
    models.setUpDataBeforeCalcLr()
    models.calcLr()
    # MAPE・相対誤差率を算出
    models.calcMAPE()
    models.calcRelativeErrorRate()
    # 結果の格納
    dictAggregateResult = {
        "MAPE": models.returnCalculatedMAPE(),
        "relativeErrorRate": models.returnRelativeErrorRateDict(),
    }
    expVarDict = models.returnExpVarDatumDF().to_dict(orient="list")
    resVarDict = models.returnResVarDatumDF().to_dict(orient="list")
    modelsName = models.returnModelsName()
    dictDatumForDF = {
        "functionName": validFunctionName,
        "benchmarkName": benchmarkName,
        "expVarDatumDict": expVarDict,
        "resVarDatumDict": resVarDict,
        "modelsName": modelsName,
        "dictAggregateResult": dictAggregateResult,
    }
    listDatumKeysForDF = dictDatumForDF.keys()
    listDatumValuesForDF = dictDatumForDF.values()
    returnDF = pd.DataFrame(index=listDatumKeysForDF, data=listDatumValuesForDF).T
    return returnDF


def test_returnDFtoMakeSummary():
    # 入力用DF、inputDFを作成する
    plotX = np.linspace(1, 20, 10)
    plotY = np.linspace(21, 40, 10)
    # functionCallNum
    functionCallNumLin = plotX + 2 * plotY + 4
    functionCallNumIp = 1 / plotX + 2 / plotY + 4
    functionCallNumLog = np.log10(plotX) + 2 * np.log10(plotY) + 4
    # processes
    process = np.linspace(1, 20, 10)
    # benchmarkClassInNum
    benchmarkClassInNum = np.linspace(21, 40, 10)
    # functionName
    functionNameLin = "functionNameLin"
    functionNameIp = "functionNameIp"
    functionNameLog = "functionNameLog"
    # benchmarkName
    benchmarkNameLin = "benchmarkNameLin"
    benchmarkNameIp = "benchmarkNameIp"
    benchmarkNameLog = "benchmarkNameLog"
    # benchmarkClass
    benchmarkClass = ["Z"] * len(benchmarkClassInNum)
    benchmarkClass[-1] = "X"

    dictForDFatLin = {
        "functionCallNum": functionCallNumLin,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatLin = pd.DataFrame(dictForDFatLin)
    inputDFatLin["functionName"] = functionNameLin
    inputDFatLin["benchmarkName"] = benchmarkNameLin

    dictForDFatIp = {
        "functionCallNum": functionCallNumIp,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatIp = pd.DataFrame(dictForDFatIp)
    inputDFatIp["functionName"] = functionNameIp
    inputDFatIp["benchmarkName"] = benchmarkNameIp

    dictForDFatLog = {
        "functionCallNum": functionCallNumLog,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatLog = pd.DataFrame(dictForDFatLog)
    inputDFatLog["functionName"] = functionNameLog
    inputDFatLog["benchmarkName"] = benchmarkNameLog

    # 関数の実行に必要な引数を作成する
    targetClass = benchmarkClass[-1]
    targetProcess = process[-1]
    expVarColNames = ["process", "benchmarkClassInNum"]
    resVarColNames = ["functionCallNum"]

    # returnDFtoMakeSummary()の実行
    resultAtLin = returnDFtoMakeSummary(
        inputDF=inputDFatLin,
        benchmarkName=benchmarkNameLin,
        validFunctionName=functionNameLin,
        targetClass=targetClass,
        targetProcess=targetProcess,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )
    resultAtIp = returnDFtoMakeSummary(
        inputDF=inputDFatIp,
        benchmarkName=benchmarkNameIp,
        validFunctionName=functionNameIp,
        targetClass=targetClass,
        targetProcess=targetProcess,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )
    resultAtLog = returnDFtoMakeSummary(
        inputDF=inputDFatLog,
        benchmarkName=benchmarkNameLog,
        validFunctionName=functionNameLog,
        targetClass=targetClass,
        targetProcess=targetProcess,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )

    # linについて
    assert len(resultAtLin) == 1
    # functionName
    functionNameAtLin = resultAtLin["functionName"].tolist()
    assert functionNameAtLin[0] == functionNameLin
    # benchmarkName
    benchmarkNameAtLinResult = resultAtLin["benchmarkName"].tolist()
    assert benchmarkNameAtLinResult[0] == benchmarkNameLin
    # expVarDatumDict
    expVarDictLinResult = resultAtLin.at[resultAtLin.index[0], "expVarDatumDict"]
    processLinResult = expVarDictLinResult["process"]
    benchmarkClassInNumLinResult = expVarDictLinResult["benchmarkClassInNum"]
    assert list(np.around(processLinResult)) == list(np.around(process[:-1]))
    assert list(np.round(benchmarkClassInNumLinResult)) == list(
        np.round(benchmarkClassInNum[:-1])
    )
    # resVarDatumDict
    resVarDictLinResult = resultAtLin.at[resultAtLin.index[0], "resVarDatumDict"]
    functionCallNumLinResult = resVarDictLinResult["functionCallNum"]
    assert list(np.round(functionCallNumLinResult)) == list(
        np.round(functionCallNumLin[:-1])
    )
    # modelsName
    modelsNameLinResult = resultAtLin.at[resultAtLin.index[0], "modelsName"]
    assert "modelLin" in modelsNameLinResult
    assert "modelIp" in modelsNameLinResult
    assert "modelLog" in modelsNameLinResult
    # dictAggregateResult
    dictAggregateResult = resultAtLin.at[resultAtLin.index[0], "dictAggregateResult"]
    MAPELinResult = dictAggregateResult["MAPE"]
    assert MAPELinResult["modelLin"] < 1.0
    relativeErrorRateLinResult = dictAggregateResult["MAPE"]
    assert relativeErrorRateLinResult["modelLin"] < 1.0

    # Ipについて
    assert len(resultAtIp) == 1
    # functionName
    functionNameAtIp = resultAtIp["functionName"].tolist()
    assert functionNameAtIp[0] == functionNameIp
    # benchmarkName
    benchmarkNameAtIpResult = resultAtIp["benchmarkName"].tolist()
    assert benchmarkNameAtIpResult[0] == benchmarkNameIp
    # expVarDatumDict
    expVarDictIpResult = resultAtIp.at[resultAtIp.index[0], "expVarDatumDict"]
    processIpResult = expVarDictIpResult["process"]
    benchmarkClassInNumIpResult = expVarDictIpResult["benchmarkClassInNum"]
    assert list(np.around(processIpResult)) == list(np.around(process[:-1]))
    assert list(np.round(benchmarkClassInNumIpResult)) == list(
        np.round(benchmarkClassInNum[:-1])
    )
    # resVarDatumDict
    resVarDictIpResult = resultAtIp.at[resultAtIp.index[0], "resVarDatumDict"]
    functionCallNumIpResult = resVarDictIpResult["functionCallNum"]
    assert list(np.round(functionCallNumIpResult)) == list(
        np.round(functionCallNumIp[:-1])
    )
    # modelsName
    modelsNameIpResult = resultAtIp.at[resultAtIp.index[0], "modelsName"]
    assert "modelLin" in modelsNameIpResult
    assert "modelIp" in modelsNameIpResult
    assert "modelLog" in modelsNameIpResult
    # dictAggregateResult
    dictAggregateResult = resultAtIp.at[resultAtIp.index[0], "dictAggregateResult"]
    MAPEIpResult = dictAggregateResult["MAPE"]
    assert MAPEIpResult["modelIp"] < 1.0
    relativeErrorRateIpResult = dictAggregateResult["MAPE"]
    assert relativeErrorRateIpResult["modelIp"] < 1.0

    # Logについて
    assert len(resultAtLog) == 1
    # functionName
    functionNameAtLog = resultAtLog["functionName"].tolist()
    assert functionNameAtLog[0] == functionNameLog
    # benchmarkName
    benchmarkNameAtLogResult = resultAtLog["benchmarkName"].tolist()
    assert benchmarkNameAtLogResult[0] == benchmarkNameLog
    # expVarDatumDict
    expVarDictLogResult = resultAtLog.at[resultAtLog.index[0], "expVarDatumDict"]
    processLogResult = expVarDictLogResult["process"]
    benchmarkClassInNumLogResult = expVarDictLogResult["benchmarkClassInNum"]
    assert list(np.around(processLogResult)) == list(np.around(process[:-1]))
    assert list(np.round(benchmarkClassInNumLogResult)) == list(
        np.round(benchmarkClassInNum[:-1])
    )
    # resVarDatumDict
    resVarDictLogResult = resultAtLog.at[resultAtLog.index[0], "resVarDatumDict"]
    functionCallNumLogResult = resVarDictLogResult["functionCallNum"]
    assert list(np.round(functionCallNumLogResult)) == list(
        np.round(functionCallNumLog[:-1])
    )
    # modelsName
    modelsNameLogResult = resultAtLog.at[resultAtLog.index[0], "modelsName"]
    assert "modelLin" in modelsNameLogResult
    assert "modelIp" in modelsNameLogResult
    assert "modelLog" in modelsNameLogResult
    # dictAggregateResult
    dictAggregateResult = resultAtLog.at[resultAtLog.index[0], "dictAggregateResult"]
    MAPELogResult = dictAggregateResult["MAPE"]
    assert MAPELogResult["modelLog"] < 1.0
    relativeErrorRateLogResult = dictAggregateResult["MAPE"]
    assert relativeErrorRateLogResult["modelLog"] < 1.0


# In[27]:


# 必要な変数などを事前に宣言するfixture
@pytest.fixture()
def returnTemporaryRawDFAtLin():
    # 入力用DF、inputDFを作成する
    plotX = np.linspace(1, 20, 10)
    plotY = np.linspace(21, 40, 10)
    # functionCallNum
    functionCallNumLin = plotX + 2 * plotY + 4
    # processes
    process = np.linspace(1, 20, 10)
    # benchmarkClassInNum
    benchmarkClassInNum = np.linspace(21, 40, 10)
    # functionName
    functionNameLin = "functionNameLin"
    # benchmarkName
    benchmarkNameLin = "benchmarkNameLin"
    # benchmarkClass
    benchmarkClass = ["Z"] * len(benchmarkClassInNum)
    benchmarkClass[-1] = "X"

    dictForDFatLin = {
        "functionCallNum": functionCallNumLin,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatLin = pd.DataFrame(dictForDFatLin)
    inputDFatLin["functionName"] = functionNameLin
    inputDFatLin["benchmarkName"] = benchmarkNameLin

    #     # 関数の実行に必要な引数を作成する
    #     targetClass = benchmarkClass[-1]
    #     targetProcess = process[-1]
    #     expVarColNames = ["process", "benchmarkClassInNum"]
    #     resVarColNames = ["functionCallNum"]

    return inputDFatLin


@pytest.fixture()
def returnTemporaryRawDFAtIp():
    # 入力用DF、inputDFを作成する
    plotX = np.linspace(1, 20, 10)
    plotY = np.linspace(21, 40, 10)
    # functionCallNum
    functionCallNumIp = 1 / plotX + 2 / plotY + 4
    # processes
    process = np.linspace(1, 20, 10)
    # benchmarkClassInNum
    benchmarkClassInNum = np.linspace(21, 40, 10)
    # functionName
    functionNameIp = "functionNameIp"
    # benchmarkName
    benchmarkNameIp = "benchmarkNameIp"
    # benchmarkClass
    benchmarkClass = ["Z"] * len(benchmarkClassInNum)
    benchmarkClass[-1] = "X"

    dictForDFatIp = {
        "functionCallNum": functionCallNumIp,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatIp = pd.DataFrame(dictForDFatIp)
    inputDFatIp["functionName"] = functionNameIp
    inputDFatIp["benchmarkName"] = benchmarkNameIp

    #     # 関数の実行に必要な引数を作成する
    #     targetClass = benchmarkClass[-1]
    #     targetProcess = process[-1]
    #     expVarColNames = ["process", "benchmarkClassInNum"]
    #     resVarColNames = ["functionCallNum"]

    return inputDFatIp


@pytest.fixture()
def returnTemporaryRawDFAtLog():
    # 入力用DF、inputDFを作成する
    plotX = np.linspace(1, 20, 10)
    plotY = np.linspace(21, 40, 10)
    # functionCallNum
    functionCallNumLog = np.log10(plotX) + 2 * np.log10(plotY) + 4
    # processes
    process = np.linspace(1, 20, 10)
    # benchmarkClassInNum
    benchmarkClassInNum = np.linspace(21, 40, 10)
    # functionName
    functionNameLog = "functionNameLog"
    # benchmarkName
    benchmarkNameLog = "benchmarkNameLog"
    # benchmarkClass
    benchmarkClass = ["Z"] * len(benchmarkClassInNum)
    benchmarkClass[-1] = "X"

    dictForDFatLog = {
        "functionCallNum": functionCallNumLog,
        "process": process,
        "benchmarkClassInNum": benchmarkClassInNum,
        "benchmarkClass": benchmarkClass,
    }
    inputDFatLog = pd.DataFrame(dictForDFatLog)
    inputDFatLog["functionName"] = functionNameLog
    inputDFatLog["benchmarkName"] = benchmarkNameLog

    #     # 関数の実行に必要な引数を作成する
    #     targetClass = benchmarkClass[-1]
    #     targetProcess = process[-1]
    #     expVarColNames = ["process", "benchmarkClassInNum"]
    #     resVarColNames = ["functionCallNum"]

    return inputDFatLog


@pytest.fixture()
def returnDFSummarizeTheResultsOfTheFunctionReturnDFtoMakeSummary(
    returnTemporaryRawDFAtLin, returnTemporaryRawDFAtIp, returnTemporaryRawDFAtLog
):
    functionNames = {
        "Lin": "functionNameLin",
        "Ip": "functionNameIp",
        "Log": "functionNameLog",
    }
    benchmarkNames = {
        "Lin": "benchmarkNameLin",
        "Ip": "benchmarkNameIp",
        "Log": "benchmarkNameLog",
    }

    expVarColNames = ["process", "benchmarkClassInNum"]
    resVarColNames = ["functionCallNum"]

    targetClasses = {}
    targetProcesses = {}
    for key, valueDF in zip(
        ["Lin", "Ip", "Log"],
        [
            returnTemporaryRawDFAtLin,
            returnTemporaryRawDFAtIp,
            returnTemporaryRawDFAtLog,
        ],
    ):
        targetClasses[key] = valueDF["benchmarkClass"].tolist()[-1]
        targetProcesses[key] = valueDF["process"].tolist()[-1]

    DFatLin = returnDFtoMakeSummary(
        returnTemporaryRawDFAtLin,
        benchmarkName=benchmarkNames["Lin"],
        validFunctionName=functionNames["Lin"],
        targetClass=targetClasses["Lin"],
        targetProcess=targetProcesses["Lin"],
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )
    DFatIp = returnDFtoMakeSummary(
        returnTemporaryRawDFAtIp,
        benchmarkName=benchmarkNames["Ip"],
        validFunctionName=functionNames["Ip"],
        targetClass=targetClasses["Ip"],
        targetProcess=targetProcesses["Ip"],
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )
    DFatLog = returnDFtoMakeSummary(
        returnTemporaryRawDFAtLog,
        benchmarkName=benchmarkNames["Log"],
        validFunctionName=functionNames["Log"],
        targetClass=targetClasses["Log"],
        targetProcess=targetProcesses["Log"],
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNames,
    )
    allDF = pd.concat([DFatLin, DFatIp, DFatLog]).reset_index(drop=True)
    allDF["benchmarkName"] = "benchmarkName"
    return allDF


# In[28]:


# returnDFtoMakeSummary()の返り値を縦に結合したDFを引数として受け、変換したデータをDFとして出力する関数
# ベンチマークの種類は1つ
def convertDictToMakeSummary(
    inputDF, modelAdoptionRate=False, averageRelativeError=False
):
    returnDict = {}

    benchmarkNames = list(set(inputDF["benchmarkName"].tolist()))
    if len(benchmarkNames) != 1:
        warnings.warn("ベンチマークの種類が複数あります。空のDFを返しました。")
        return returnDict
    functionNames = list(set(inputDF["functionName"].tolist()))
    modelNames = inputDF["modelsName"].tolist()[0]

    if modelAdoptionRate:
        # [ベンチマーク名|モデル0の採用率(最小MAPE, 最大MAPE)| ... |モデルnの採用率(最小MAPE, 最大MAPE)]

        # inputDF["dictAggregateResult"]の"MAPE"をそれぞれ比較
        # MAPE["modelName":["count":int, "min":float, "max":float]]
        aboutMAPE = {}
        for modelName in modelNames:
            aboutMAPE[modelName] = {"count": 0, "min": float("inf"), "max": 0.0}

        for index, row in inputDF.iterrows():
            rowMAPEData = row["dictAggregateResult"]["MAPE"]
            lowestMAPE = min(list(rowMAPEData.values()))
            lowestMAPEindex = list(rowMAPEData.values()).index(lowestMAPE)
            lowestMAPEModelName = list(rowMAPEData.keys())[lowestMAPEindex]

            aboutMAPE[lowestMAPEModelName]["count"] += 1
            if aboutMAPE[lowestMAPEModelName]["min"] > rowMAPEData[lowestMAPEModelName]:
                aboutMAPE[lowestMAPEModelName]["min"] = rowMAPEData[lowestMAPEModelName]
            if aboutMAPE[lowestMAPEModelName]["max"] < rowMAPEData[lowestMAPEModelName]:
                aboutMAPE[lowestMAPEModelName]["max"] = rowMAPEData[lowestMAPEModelName]

        returnDict["modelAdoptionRate"] = aboutMAPE

    if averageRelativeError:
        # [ベンチマーク名|対象環境での関数コール回数の相対誤差率 の平均]

        # inputDF["dictAggregateResult"]の"relativeErrorRate"をそれぞれ比較
        aboutRelativeErrorRate = {}
        for index, row in inputDF.iterrows():
            rowMAPEData = row["dictAggregateResult"]
            aboutRelativeErrorRate[row["functionName"]] = min(
                list(rowMAPEData["relativeErrorRate"].values())
            )

        returnDict["averageRelativeError"] = sum(
            list(aboutRelativeErrorRate.values())
        ) / len(aboutRelativeErrorRate.keys())

    return returnDict


def test_convertDictToMakeSummary(
    returnDFSummarizeTheResultsOfTheFunctionReturnDFtoMakeSummary,
):
    inputDF = returnDFSummarizeTheResultsOfTheFunctionReturnDFtoMakeSummary

    functionNames = list(set(inputDF["functionName"].tolist()))
    benchmarkNames = list(set(inputDF["benchmarkName"].tolist()))

    resultFF = convertDictToMakeSummary(
        inputDF=inputDF, modelAdoptionRate=False, averageRelativeError=False
    )
    resultTT = convertDictToMakeSummary(
        inputDF=inputDF, modelAdoptionRate=True, averageRelativeError=True
    )

    assert len(resultFF) == 0
    assert len(resultTT) == 2

    aboutMAPE = resultTT["modelAdoptionRate"]
    aboutReletiveError = resultTT["averageRelativeError"]

    # Lin
    assert aboutMAPE["modelLin"]["count"] == 1
    assert aboutMAPE["modelLin"]["min"] < 0.1
    # Ip
    assert aboutMAPE["modelIp"]["count"] == 1
    assert aboutMAPE["modelIp"]["min"] < 0.1
    # Log
    assert aboutMAPE["modelLog"]["count"] == 1
    assert aboutMAPE["modelLog"]["min"] < 0.1

    pass


# In[29]:


# 目的変数を構築するための関数
def returnListForBranchModel(
    inputList=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], branchIndex=5, a=1, b=1
):
    def _funcBranch(numList, a, b):
        return [a * num + b for num in numList]

    returnedList = [
        num if inputList.index(num) < branchIndex else inputList[branchIndex]
        for num in inputList
    ]
    returnedList = _funcBranch(returnedList, a, b)
    return returnedList


def test_returnListForBranchModel():
    branchIndex = 5

    inputList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # a=1, b=1
    returnedList = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
    result = returnListForBranchModel(
        inputList=inputList, branchIndex=branchIndex, a=1, b=0
    )
    assert returnedList == result

    inputList = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # a=2, b=3
    returnedList = [25, 27, 29, 31, 33, 35, 35, 35, 35, 35]
    result = returnListForBranchModel(
        inputList=inputList, branchIndex=branchIndex, a=2, b=3
    )
    assert returnedList == result

    inputList = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # a=4, b=5
    returnedList = [89, 93, 97, 101, 105, 109, 109, 109, 109, 109]
    result = returnListForBranchModel(
        inputList=inputList, branchIndex=branchIndex, a=4, b=5
    )
    assert returnedList == result


# In[30]:


# 分岐点を見つける関数
# 引数は分岐点を探るためのリスト
# 分岐点を発見出来たら、そのインデックス値を返す。失敗もしくは存在しなければ-1を返す。
def returnBranchIndexOfList(inputListX=[], inputListY=[]):
    # データを分割するための関数
    def returnProcessedList(inputList, elementQuantity=3):
        if len(inputList) < elementQuantity:
            returnList = [inputList]
            return returnList

        returnList = []
        for i in range(len(inputList) - elementQuantity):
            processedList = inputList[i : i + elementQuantity]
            returnList.append(processedList)
        return returnList

    if len(inputListX) == 0 or len(inputListY) == 0:
        warnings.warn("引数のリストの要素数が0です。")
        return -1

    processedX = returnProcessedList(inputList=inputListX, elementQuantity=3)
    processedY = returnProcessedList(inputList=inputListY, elementQuantity=3)

    coefList = []
    # 1.  for で splittedList の要素を回す
    for elementX, elementY in zip(processedX, processedY):
        # 1-1. 線形回帰をして傾きをリストに入れる
        lr = LinearRegression()
        reshapedX = np.array(elementX).reshape((-1, 1))
        reshapedY = np.array(elementY).reshape((-1, 1))
        lr.fit(reshapedX, reshapedY)

        if len(lr.coef_) != 1:
            warnings.warn("傾きが複数存在します")
        # 算出された生の傾きを取得
        coef = lr.coef_[0][0]
        # 取得した傾きの桁数を短縮する
        coef = int(coef * 100) / 100
        coefList.append(coef)
    # 2. 傾きが保持されたリストを集合(set)にする
    coefSetList = list(set(coefList))
    # 保持された傾きが3種類でない場合は、本関数で検出できる範疇を超えるため、return(-1)する。
    if len(coefSetList) != 3:
        return -1
    # 3. 集合の中の中央値を取得
    medInCoefList = median_low(coefSetList)
    # 4. 中央値から傾きが保持されたリストのインデクスを取得する
    medIndexInCoefList = coefList.index(medInCoefList)
    # 5. 傾きが保持されたリストのインデクスから入力リストの実要素のインデックスを取得する
    oneDataSetInProcessedX = processedX[medIndexInCoefList]
    medInOneDataSetInProcessedX = median_low(oneDataSetInProcessedX)
    medIndexInOneDataSetInProcessedX = inputListX.index(medInOneDataSetInProcessedX)
    return medIndexInOneDataSetInProcessedX


def test_returnBranchIndexOfList():
    # データを用意
    # データ１：分岐点以降はデータの変化なし
    branchIndex = 11
    dataBase1 = list(range(branchIndex))
    dataBase2 = [branchIndex] * (21 - 11)
    dataList1 = [3 * x + 4 for x in dataBase1]
    dataList2 = [3 * x + 4 for x in dataBase2]
    dataBase2 = list(range(11, 21))
    dataListY = dataList1 + dataList2
    dataListX = dataBase1 + dataBase2
    assert len(dataListY) == len(dataListX)
    result = returnBranchIndexOfList(inputListX=dataListX, inputListY=dataListY)
    assert result == branchIndex

    # データ２：分岐点で異なる線形モデルに変化
    branchIndex = 10
    dataBase1 = list(range(branchIndex))
    dataBase2 = list(range(branchIndex, 20))
    dataList1 = [x + 1 for x in dataBase1]
    dataList2 = [0.5 * x + 6 for x in dataBase2]
    dataListY = dataList1 + dataList2
    dataListX = dataBase1 + dataBase2
    result = returnBranchIndexOfList(inputListX=dataListX, inputListY=dataListY)
    assert result == branchIndex

    # データ３：分岐点無し
    branchIndex = 22
    dataBase1 = list(range(branchIndex))
    dataListY = [5 * x + 6 for x in dataBase1]
    dataListX = dataBase1
    result = returnBranchIndexOfList(inputListX=dataListX, inputListY=dataListY)
    assert result == -1


# In[31]:


# Latex形式の表で出力する直前のDFを返す関数
# 1変数での予測における、各モデルの採用割合と最小MAPE, 最大MAPEが要素
def returnDFtoLatexWithMAPE(
    benchmarkNames=["cg", "ep", "ft", "is", "lu", "mg"],
    # classes = ["A", "B", "C", "D"]
    classes=["B"],
    processes=[2, 4, 8, 16, 32, 64, 128, 256],
    targetIndex=-1,
    csvDirPath="./csv_files/",
    modelNames=["ModelLin", "ModelIp", "ModelLog", "ModelBranch"],
):
    dfByDatum = returnDFSummarizedData(
        benchmarkNames=benchmarkNames,
        classes=classes,
        processes=processes,
        targetIndex=targetIndex,
        csvDirPath=csvDirPath,
        modelNames=modelNames,
    )
    # dfByDatum

    dictForLatexTable = {}
    numOfData = 0
    for benchmarkName in benchmarkNames:
        dictForLatexTable[benchmarkName] = dfByDatum[
            dfByDatum["benchmarkName"] == benchmarkName
        ]
        numOfData += len(dfByDatum[dfByDatum["benchmarkName"] == benchmarkName])

    numOfData

    listForDF = []

    for benchmarkName in benchmarkNames:
        listForDF.append(
            returnSeriesOfDatumPerBenchmark(inputDF=dictForLatexTable[benchmarkName])
        )
    DF = pd.DataFrame(listForDF)
    return DF


# In[32]:


# 入力に該当する初期化データを返す関数
# benchmarkName:ベンチマーク名（文字列）
# programSize:問題サイズ（アルファベット1文字）
# 返り値:辞書


def returnInitVars(benchmarkName="", programSize=""):
    if benchmarkName == "":
        warnings.warn(UserWarning("no benchmarkName in returnInitVars()"))
        return 0

    if programSize == "":
        warnings.warn(UserWarning("no benchmarkName in returnInitVars()"))
        return 0

    programSizes = ["S", "W", "A", "B", "C", "D", "E", "F"]

    if benchmarkName == "cg":
        na = -1
        nonzer = -1
        niter = -1
        shift = -1
        if programSize == "S":
            na = 1400
            nonzer = 7
            niter = 15
            shift = "10.d0"
        elif programSize == "W":
            na = 7000
            nonzer = 8
            niter = 15
            shift = "12.d0"
        elif programSize == "A":
            na = 14000
            nonzer = 11
            niter = 15
            shift = "20.d0"
        elif programSize == "B":
            na = 75000
            nonzer = 13
            niter = 75
            shift = "60.d0"
        elif programSize == "C":
            na = 150000
            nonzer = 15
            niter = 75
            shift = "110.d0"
        elif programSize == "D":
            na = 150000
            nonzer = 21
            niter = 100
            shift = "500.d0"
        elif programSize == "E":
            na = 9000000
            nonzer = 26
            niter = 100
            shift = "1.5d3"
        elif programSize == "F":
            na = 54000000
            nonzer = 31
            niter = 100
            shift = "5.0d3"

        else:
            warnings.warn("not correct programSize")

        retDict = {"na": na, "nonzer": nonzer, "niter": niter, "shift": shift}

    elif benchmarkName == "ep":
        if programSize in programSizes:
            retDict = {"programSize": programSize}
        else:
            warnings.warn("not correct programSize")
        pass
    elif benchmarkName == "ft":
        if programSize == "S":
            d1 = 64
            d2 = 64
            d3 = 64
            nt = 6
        elif programSize == "W":
            d1 = 128
            d2 = 128
            d3 = 32
            nt = 6
        elif programSize == "A":
            d1 = 256
            d2 = 256
            d3 = 128
            nt = 6
        elif programSize == "B":
            d1 = 512
            d2 = 256
            d3 = 256
            nt = 20
        elif programSize == "C":
            d1 = 512
            d2 = 512
            d3 = 512
            nt = 20
        elif programSize == "D":
            d1 = 2048
            d2 = 1024
            d3 = 1024
            nt = 25
        elif programSize == "E":
            d1 = 4096
            d2 = 2048
            d3 = 2048
            nt = 25
        elif programSize == "F":
            d1 = 8192
            d2 = 4096
            d3 = 4096
            nt = 25

        else:
            warnings.warn("not correct programSize")
        retDict = {"d1": d1, "d2": d2, "d3": d3, "nt": nt}

    elif benchmarkName == "is":
        CLASS = "S"
        NUM_PROCS = 1
        MIN_PROCS = 1
        ONE = 1
        if programSize == "S":
            TOTAL_KEYS_LOG_2 = 16
            MAX_KEY_LOG_2 = 11
            NUM_BUCKETS_LOG_2 = 9
        elif programSize == "W":
            TOTAL_KEYS_LOG_2 = 20
            MAX_KEY_LOG_2 = 16
            NUM_BUCKETS_LOG_2 = 10
        elif programSize == "A":
            TOTAL_KEYS_LOG_2 = 23
            MAX_KEY_LOG_2 = 19
            NUM_BUCKETS_LOG_2 = 10
        elif programSize == "B":
            TOTAL_KEYS_LOG_2 = 25
            MAX_KEY_LOG_2 = 21
            NUM_BUCKETS_LOG_2 = 10
        elif programSize == "C":
            TOTAL_KEYS_LOG_2 = 27
            MAX_KEY_LOG_2 = 23
            NUM_BUCKETS_LOG_2 = 10
        elif programSize == "D":
            TOTAL_KEYS_LOG_2 = 29
            MAX_KEY_LOG_2 = 27
            NUM_BUCKETS_LOG_2 = 10
            MIN_PROCS = 4
        elif programSize == "E":
            TOTAL_KEYS_LOG_2 = 29
            MAX_KEY_LOG_2 = 31
            NUM_BUCKETS_LOG_2 = 10
            MIN_PROCS = 64
            ONE = "1L"
        else:
            warnings.warn("not correct programSize")
        retDict = {
            "TOTAL_KEYS_LOG_2": TOTAL_KEYS_LOG_2,
            "MAX_KEY_LOG_2": MAX_KEY_LOG_2,
            "NUM_BUCKETS_LOG_2": NUM_BUCKETS_LOG_2,
            "MIN_PROCS": MIN_PROCS,
            "ONE": ONE,
        }

    elif benchmarkName == "lu":
        if programSize in programSizes:
            retDict = {"programSize": programSize}
        else:
            warnings.warn("not correct programSize")
        pass
    elif benchmarkName == "mg":
        if programSize in programSizes:
            retDict = {"programSize": programSize}
        else:
            warnings.warn("not correct programSize")
        pass
    else:
        warnings.warn("not correct benchmarkName")

    return retDict


def test_returnInitVars00():
    benchmarkNames = ["cg", "ep", "ft", "is", "lu", "mg"]
    programSizes = ["A", "B", "C", "D"]

    randomIndexToBenchmarkName = random.randint(0, len(benchmarkNames) - 1)
    randomIndexToProgramSize = random.randint(0, len(programSizes) - 1)

    benchmarkName = benchmarkNames[randomIndexToBenchmarkName]
    programSize = programSizes[randomIndexToProgramSize]

    ret = returnInitVars(benchmarkName=benchmarkName, programSize=programSize)
    assert type(ret) == dict


@pytest.mark.filterwarnings("ignore:no benchmarkName in ")
def test_returnInitVars01():
    ret = returnInitVars(benchmarkName="cg")
    assert ret == 0
    ret = returnInitVars(programSize="C")
    assert ret == 0


# In[33]:


# ベンチマーク名・問題サイズを受け取り、条件に合った変数群を辞書形式で返す関数, 引数のbenchmarkClassが''のときは負の値がバリューとなった辞書を返す関数
# 引数：ベンチマーク名、問題サイズ
# 返り値：辞書（形式ー＞{"<変数名1>":<値1>, "<変数名2>":<値2>}）


def retDictAboutInitVars(benchmarkName, benchmarkClass):
    retDict = {}
    if benchmarkName == "cg":
        if benchmarkClass == "S":
            retDict["na"] = 1400
            retDict["nonzer"] = 7
            retDict["niter"] = 15
            retDict["shift"] = 24
        elif benchmarkClass == "W":
            retDict["na"] = 7000
            retDict["nonzer"] = 8
            retDict["niter"] = 15
            retDict["shift"] = 25
        elif benchmarkClass == "A":
            retDict["na"] = 14000
            retDict["nonzer"] = 11
            retDict["niter"] = 15
            retDict["shift"] = 20
        elif benchmarkClass == "B":
            retDict["na"] = 75000
            retDict["nonzer"] = 13
            retDict["niter"] = 75
            retDict["shift"] = 60
        elif benchmarkClass == "C":
            retDict["na"] = 150000
            retDict["nonzer"] = 15
            retDict["niter"] = 75
            retDict["shift"] = 110
        elif benchmarkClass == "D":
            retDict["na"] = 1500000
            retDict["nonzer"] = 21
            retDict["niter"] = 100
            retDict["shift"] = 500
        elif benchmarkClass == "E":
            retDict["na"] = 9000000
            retDict["nonzer"] = 26
            retDict["niter"] = 100
            retDict["shift"] = 1500
        elif benchmarkClass == "F":
            retDict["na"] = 54000000
            retDict["nonzer"] = 26
            retDict["niter"] = 100
            retDict["shift"] = 1500
        else:
            retDict["na"] = -1
            retDict["nonzer"] = -1
            retDict["niter"] = -1
            retDict["shift"] = -1

    elif benchmarkName == "ep":
        if benchmarkClass == "S":
            retDict["m"] = 24
        elif benchmarkClass == "W":
            retDict["m"] = 25
        elif benchmarkClass == "A":
            retDict["m"] = 28
        elif benchmarkClass == "B":
            retDict["m"] = 30
        elif benchmarkClass == "C":
            retDict["m"] = 32
        elif benchmarkClass == "D":
            retDict["m"] = 36
        elif benchmarkClass == "E":
            retDict["m"] = 40
        elif benchmarkClass == "F":
            retDict["m"] = 44
        else:
            retDict["m"] = -1
    elif benchmarkName == "ft":
        if benchmarkClass == "S":
            retDict["nx"] = 64
            retDict["ny"] = 64
            retDict["nz"] = 64
            retDict["niter_default"] = 6
        elif benchmarkClass == "W":
            retDict["nx"] = 128
            retDict["ny"] = 128
            retDict["nz"] = 32
            retDict["niter_default"] = 6
        elif benchmarkClass == "A":
            retDict["nx"] = 256
            retDict["ny"] = 256
            retDict["nz"] = 128
            retDict["niter_default"] = 6
        elif benchmarkClass == "B":
            retDict["nx"] = 512
            retDict["ny"] = 512
            retDict["nz"] = 256
            retDict["niter_default"] = 20
        elif benchmarkClass == "C":
            retDict["nx"] = 512
            retDict["ny"] = 512
            retDict["nz"] = 512
            retDict["niter_default"] = 20
        elif benchmarkClass == "D":
            retDict["nx"] = 2048
            retDict["ny"] = 1024
            retDict["nz"] = 1024
            retDict["niter_default"] = 25
        elif benchmarkClass == "E":
            retDict["nx"] = 4096
            retDict["ny"] = 2048
            retDict["nz"] = 2048
            retDict["niter_default"] = 25
        elif benchmarkClass == "F":
            retDict["nx"] = 8192
            retDict["ny"] = 4096
            retDict["nz"] = 4096
            retDict["niter_default"] = 25
        else:
            retDict["nx"] = -1
            retDict["ny"] = -1
            retDict["nz"] = -1
            retDict["niter_default"] = -1
    elif benchmarkName == "is":
        if benchmarkClass == "S":
            retDict["TOTAL_KEY_LOG_2"] = 2**16
            retDict["MAX_KEY_LOG_2"] = 2**11
        elif benchmarkClass == "W":
            retDict["TOTAL_KEY_LOG_2"] = 2**20
            retDict["MAX_KEY_LOG_2"] = 2**16
        elif benchmarkClass == "A":
            retDict["TOTAL_KEY_LOG_2"] = 2**23
            retDict["MAX_KEY_LOG_2"] = 2**19
        elif benchmarkClass == "B":
            retDict["TOTAL_KEY_LOG_2"] = 2**25
            retDict["MAX_KEY_LOG_2"] = 2**21
        elif benchmarkClass == "C":
            retDict["TOTAL_KEY_LOG_2"] = 2**27
            retDict["MAX_KEY_LOG_2"] = 2**23
        elif benchmarkClass == "D":
            retDict["TOTAL_KEY_LOG_2"] = 2**31
            retDict["MAX_KEY_LOG_2"] = 2**27
        elif benchmarkClass == "E":
            retDict["TOTAL_KEY_LOG_2"] = 2**35
            retDict["MAX_KEY_LOG_2"] = 2**31
        else:
            retDict["TOTAL_KEY_LOG_2"] = -1
            retDict["MAX_KEY_LOG_2"] = -1
    elif benchmarkName == "mg":
        if benchmarkClass == "S":
            retDict["nx_default"] = 32
            retDict["ny_default"] = 32
            retDict["nz_default"] = 32
            retDict["nit_default"] = 4
        elif benchmarkClass == "W":
            retDict["nx_default"] = 128
            retDict["ny_default"] = 128
            retDict["nz_default"] = 128
            retDict["nit_default"] = 4
        elif benchmarkClass == "A":
            retDict["nx_default"] = 256
            retDict["ny_default"] = 256
            retDict["nz_default"] = 256
            retDict["nit_default"] = 4
        elif benchmarkClass == "B":
            retDict["nx_default"] = 256
            retDict["ny_default"] = 256
            retDict["nz_default"] = 256
            retDict["nit_default"] = 20
        elif benchmarkClass == "C":
            retDict["nx_default"] = 512
            retDict["ny_default"] = 512
            retDict["nz_default"] = 512
            retDict["nit_default"] = 20
        elif benchmarkClass == "D":
            retDict["nx_default"] = 2048
            retDict["ny_default"] = 1024
            retDict["nz_default"] = 1024
            retDict["nit_default"] = 50
        elif benchmarkClass == "E":
            retDict["nx_default"] = 2048
            retDict["ny_default"] = 2048
            retDict["nz_default"] = 2048
            retDict["nit_default"] = 50
        elif benchmarkClass == "F":
            retDict["nx_default"] = 4096
            retDict["ny_default"] = 4096
            retDict["nz_default"] = 4096
            retDict["nit_default"] = 50
        else:
            retDict["nx_default"] = -1
            retDict["ny_default"] = -1
            retDict["nz_default"] = -1
            retDict["nit_default"] = -1
    elif benchmarkName == "lu":
        if benchmarkClass == "S":
            retDict["isiz01"] = 12
            retDict["isiz02"] = 12
            retDict["isiz03"] = 12
            retDict["itmax_default"] = 50
            retDict["dt_default"] = 0.5
        elif benchmarkClass == "W":
            retDict["isiz01"] = 33
            retDict["isiz02"] = 33
            retDict["isiz03"] = 33
            retDict["itmax_default"] = 300
            retDict["dt_default"] = 0.0015
        elif benchmarkClass == "A":
            retDict["isiz01"] = 64
            retDict["isiz02"] = 64
            retDict["isiz03"] = 64
            retDict["itmax_default"] = 250
            retDict["dt_default"] = 2
        elif benchmarkClass == "B":
            retDict["isiz01"] = 102
            retDict["isiz02"] = 102
            retDict["isiz03"] = 102
            retDict["itmax_default"] = 250
            retDict["dt_default"] = 2
        elif benchmarkClass == "C":
            retDict["isiz01"] = 162
            retDict["isiz02"] = 162
            retDict["isiz03"] = 162
            retDict["itmax_default"] = 250
            retDict["dt_default"] = 2
        elif benchmarkClass == "D":
            retDict["isiz01"] = 408
            retDict["isiz02"] = 408
            retDict["isiz03"] = 408
            retDict["itmax_default"] = 300
            retDict["dt_default"] = 1
        elif benchmarkClass == "E":
            retDict["isiz01"] = 1020
            retDict["isiz02"] = 1020
            retDict["isiz03"] = 1020
            retDict["itmax_default"] = 300
            retDict["dt_default"] = 0.5
        elif benchmarkClass == "F":
            retDict["isiz01"] = 2560
            retDict["isiz02"] = 2560
            retDict["isiz03"] = 2560
            retDict["itmax_default"] = 300
            retDict["dt_default"] = 0.2
        else:
            retDict["isiz01"] = -1
            retDict["isiz02"] = -1
            retDict["isiz03"] = -1
            retDict["itmax_default"] = -1
            retDict["dt_default"] = -1
    elif benchmarkName == "sp":
        if benchmarkClass == "S":
            retDict["problem_size"] = 12
            retDict["niter_default"] = 100
            retDict["dt_default"] = 0.015
        elif benchmarkClass == "W":
            retDict["problem_size"] = 36
            retDict["niter_default"] = 400
            retDict["dt_default"] = 0.0015
        elif benchmarkClass == "A":
            retDict["problem_size"] = 64
            retDict["niter_default"] = 400
            retDict["dt_default"] = 0.0015
        elif benchmarkClass == "B":
            retDict["problem_size"] = 102
            retDict["niter_default"] = 400
            retDict["dt_default"] = 0.001
        elif benchmarkClass == "C":
            retDict["problem_size"] = 162
            retDict["niter_default"] = 400
            retDict["dt_default"] = 0.00067
        elif benchmarkClass == "D":
            retDict["problem_size"] = 408
            retDict["niter_default"] = 500
            retDict["dt_default"] = 0.0003
        elif benchmarkClass == "E":
            retDict["problem_size"] = 1020
            retDict["niter_default"] = 500
            retDict["dt_default"] = 0.0001
        elif benchmarkClass == "F":
            retDict["problem_size"] = 2560
            retDict["niter_default"] = 500
            retDict["dt_default"] = 0.000015
        else:
            retDict["problem_size"] = -1
            retDict["niter_default"] = -1
            retDict["dt_default"] = -1
    return retDict


# 生データを引数として受け取り、そのデータの問題サイズの値に合わせた追加の初期値を追加し、引数として渡されたDFにデータが付与されたDFを返す関数
# 引数：生データDF
# 返り値：生データDFにベンチマークごとの初期値が付与されたDF


def addInitDataToRawDF(rawDF):
    # ベンチマーク名を引数から取得
    benchmarkNames = sorted(list(set(rawDF["benchmarkName"])))
    # ベンチマーク名が複数含まれている場合はreturn -1する
    if len(benchmarkNames) != 1:
        warnings.warn("ベンチマーク名が複数もしくは1つも渡されていません")
        return -1
    # 問題サイズrawDFから取得
    benchmarkClasses = sorted(list(set(rawDF["benchmarkClass"])))
    # 追加する列を作成
    columnDict = retDictAboutInitVars(
        benchmarkName=benchmarkNames[0], benchmarkClass=""
    )
    for dictKey in columnDict.keys():
        rawDF[dictKey] = columnDict[dictKey]
    # ベンチマーク名・問題サイズに合わせた値を格納
    for benchmarkName in benchmarkNames:
        for benchmarkClass in benchmarkClasses:
            # ベンチマーク名と問題サイズを満たす行を抽出
            extractedBools = (rawDF["benchmarkName"] == benchmarkName) & (
                rawDF["benchmarkClass"] == benchmarkClass
            )
            # 抽出された行に値を格納
            if len(rawDF[extractedBools]) > 0:
                dictAboutInitVars = retDictAboutInitVars(
                    benchmarkName=benchmarkName, benchmarkClass=benchmarkClass
                )
                for columnName in dictAboutInitVars.keys():
                    rawDF.loc[extractedBools, columnName] = dictAboutInitVars[
                        columnName
                    ]
    return rawDF


def test_addInitDataToRawDF():
    # rawDF の作成（cg）
    functionName = [
        "functionAtCG",
    ]
    functionCallNum = [1]
    benchmarkName = ["cg"]
    benchmarkClass = ["A"]
    process = [2]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["na"] = 14000
    rawDF["nonzer"] = 11
    rawDF["niter"] = 15
    rawDF["shift"] = 20
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)
    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    # rawDF の作成（ep）
    functionName = [
        "functionAtEP",
    ]
    functionCallNum = [2]
    benchmarkName = ["ep"]
    benchmarkClass = ["B"]
    process = [4]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["m"] = 30
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)
    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    # rawDF の作成（ft）
    functionName = [
        "functionAtFT",
    ]
    functionCallNum = [4]
    benchmarkName = ["ft"]
    benchmarkClass = ["C"]
    process = [8]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["nx"] = 512
    rawDF["ny"] = 512
    rawDF["nz"] = 512
    rawDF["niter_default"] = 20
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)
    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    # rawDF の作成（is）
    functionName = [
        "functionAtIS",
    ]
    functionCallNum = [8]
    benchmarkName = ["is"]
    benchmarkClass = ["D"]
    process = [16]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["TOTAL_KEY_LOG_2"] = 2**31
    rawDF["MAX_KEY_LOG_2"] = 2**27
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)
    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    # rawDF の作成（lu）
    functionName = [
        "functionAtLU",
    ]
    functionCallNum = [16]
    benchmarkName = ["lu"]
    benchmarkClass = ["E"]
    process = [32]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["isiz01"] = 1020
    rawDF["isiz02"] = 1020
    rawDF["isiz03"] = 1020
    rawDF["itmax_default"] = 300
    rawDF["dt_default"] = 0.5
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)
    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    # rawDF の作成（mg）
    functionName = [
        "functionAtMG",
    ]
    functionCallNum = [32]
    benchmarkName = ["mg"]
    benchmarkClass = ["F"]
    process = [64]
    rawDict = {
        "functionName": functionName,
        "functionCallNum": functionCallNum,
        "benchmarkName": benchmarkName,
        "benchmarkClass": benchmarkClass,
        "process": process,
    }
    rawDF = pd.DataFrame(rawDict)
    originalRawDF = rawDF.copy()
    # rawDF に手動で関数のやっていることを実施した DF を作成
    rawDF["nx_default"] = 4096
    rawDF["ny_default"] = 4096
    rawDF["nz_default"] = 4096
    rawDF["nit_default"] = 50
    # 比較して正しいことを確認
    retDF = addInitDataToRawDF(originalRawDF)

    pd.testing.assert_frame_equal(retDF, rawDF, check_dtype=False)

    pass
    return rawDF


# In[34]:


# 辞書:<キー1：バリュー1, キー2：バリュー2, ... , キーn：バリューn>のときに最低値のバリューのキーを返す関数
# 引数：辞書
# 返り値：最低値のバリューのキーを返す関数


def retMinValsKey(inputDict):
    # 最低値のバリューを取得
    minItem = min(list(inputDict.values()))
    # 最低値のバリューを保持しているキーを取得
    retKeyList = [k for k, v in inputDict.items() if v == minItem]
    retKey = retKeyList[0]
    return retKey


def test_retMinValsKey():
    ans = "ans"
    inputDict = {"ans": -1, "ans2": -1, "ans3": -1}
    retAns = retMinValsKey(inputDict=inputDict)
    assert ans == retAns

    inputDict = {"ans2": 2, "ans3": 2, "ans": -1}
    retAns = retMinValsKey(inputDict=inputDict)
    assert ans == retAns

    inputDict = {"ans2": 3, "ans": 1, "ans3": 2}
    retAns = retMinValsKey(inputDict=inputDict)
    assert ans == retAns


test_retMinValsKey()


# In[35]:


# 引数に該当するデータから説明変数として使用する列名のリストを返す関数
# benchmarkName:ベンチマーク名
# classes:ベンチマーククラスのリスト
# processes:コア数のリスト
# csvDirPath:CSVの保存されているディレクトリへのパス
# baseExpVar:真偽値、Trueなら
# initExpVar:真偽値、Trueなら
# ["説明変数名1", ... , "説明変数名N"]となっているリスト


def returnExplanatoryVariablesList(
    benchmarkName="",
    classes=[],
    processes=[],
    csvDirPath="../csv_files/",
    baseExpVar=True,
    initExpVar=True,
):
    # 返り値として返すリストの初期化
    retList = []
    # 引数に問題がないかを確認
    if benchmarkName == "":
        warnings.warn("ベンチマーク名が入力されていません")
        return retList
    if len(classes) == 0:
        warnings.warn("問題サイズのリストに何も入っていません")
        return retList
    if len(processes) == 0:
        warnings.warn("コア数のリストに何も入っていません")
        return retList
    # 条件に当てはまるデータを取得
    rawDF = returnCollectedExistingData(
        benchmarkNames=[benchmarkName],
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    # 取得したデータに対して初期値を付与する処理
    rawDF = addInitDataToRawDF(rawDF)
    # baseExpVarに対する処理
    baseExpVarList = ["process", "intBenchmarkClass"]
    if baseExpVar == True:
        retList = retList + baseExpVarList
    # initExpVarに対する処理
    initExpVarList = rawDF.columns.to_list()
    for baseExpVarElement in baseExpVarList:
        if baseExpVarElement in initExpVarList:
            initExpVarList.remove(baseExpVarElement)
    for removeElement in [
        "functionName",
        "functionCallNum",
        "benchmarkName",
        "benchmarkClass",
    ]:
        initExpVarList.remove(removeElement)
    if initExpVar == True:
        retList = retList + initExpVarList
    return retList


def test_returnExplanatoryVariablesList(csvDirPath="../csv_files/"):
    # データを取得
    benchmarkName = "cg"
    classes = ["S", "W", "A", "B", "C", "D", "E", "F"]
    processes = [128]

    rawDF = returnCollectedExistingData(
        benchmarkNames=[benchmarkName],
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    rawDF = addInitDataToRawDF(rawDF)

    # テストケース１：基本列名（コア数・数値化された問題サイズ）
    shouldbeResult = ["process", "intBenchmarkClass"]
    retResult = returnExplanatoryVariablesList(
        benchmarkName=benchmarkName,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
        baseExpVar=True,
        initExpVar=False,
    )
    assert shouldbeResult == retResult
    # テストケース２：初期化変数の列名
    shouldbeResult = ["na", "nonzer", "niter", "shift"]
    retResult = returnExplanatoryVariablesList(
        benchmarkName=benchmarkName,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
        baseExpVar=False,
        initExpVar=True,
    )
    assert shouldbeResult == retResult
    # テストケース３：基本列名（コア数・数値化された問題サイズ）＆初期化変数の列名
    shouldbeResult = ["process", "intBenchmarkClass", "na", "nonzer", "niter", "shift"]
    retResult = returnExplanatoryVariablesList(
        benchmarkName=benchmarkName,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
        baseExpVar=True,
        initExpVar=True,
    )
    assert shouldbeResult == retResult

    return 0


# In[36]:


def addLowestMAPEColumn(
    inputDF: pd.DataFrame, model_name_list: list[str] = [], version: int = 1
) -> pd.DataFrame:
    """addLowestMAPEColumn(inputDF, model_name_list=[], version=1)

    引数として渡されたDFに最低MAPEの列を追加する関数

    Args:
        引数の名前 (引数の型): 引数の説明
        引数の名前 (:obj:`引数の型`, optional): 引数の説明.
        inputDF (pd.DataFrame): 列構成が次を満たすDF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n]
        model_name_list (list[str]): モデル名が要素のリスト
        version (int): バージョン。1がオリジナル。

    Returns:
        pd.DataFrame: 列構成が次を満たすDF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n, 最低MAPE]
    Note:
        なし

    """

    if len(model_name_list) == 0 or version == 1:
        funcNames: list[str] = inputDF.index.to_list()
        modelNames: list[str] = inputDF.columns.to_list()

        inputDF["最低値"] = math.inf

        for funcName in funcNames:
            lowestInFunc: float = math.inf
            seriesInFunc: pd.Series = inputDF.loc[funcName]
            for modelName in modelNames:
                elem: float = seriesInFunc[modelName]
                if elem < lowestInFunc:
                    lowestInFunc = elem
                inputDF.at[funcName, "最低値"] = lowestInFunc

        return inputDF
    elif version == 2:
        inputDF["最低値"] = math.inf

        func_names: list[str] = inputDF.index.to_list()

        for func_name in func_names:
            lowestInFunc: float = math.inf
            seriesInFunc: pd.Series = inputDF.loc[func_name]
            for model_name in model_name_list:
                elem: float = seriesInFunc[model_name]
                if elem < lowestInFunc:
                    lowestInFunc = elem
                inputDF.at[func_name, "最低値"] = lowestInFunc
        return inputDF


def test_addLowestMAPEColumn():
    # version = 1 でのテスト
    # 入力を作成
    base_dict = {
        "functionName": [
            "functionName0",
            "functionName1",
            "functionName2",
            "functionName3",
            "functionName4",
            "functionName5",
        ],
        "lin": [3, 3, 1, 3, 1, 1],
        "ip": [2, 1, 3, 1, 3, 1],
        "log": [1, 2, 2, 1, 1, 3],
    }
    input_DF = pd.DataFrame.from_dict(data=base_dict)
    input_DF = input_DF.set_index("functionName")
    # テスト結果を手動で作成
    result_DF_sample = input_DF.copy(deep=True)
    result_DF_sample["最低値"] = [1, 1, 1, 1, 1, 1]
    # テスト対象のテストを実行
    result_DF = addLowestMAPEColumn(input_DF)
    # 結果を比較
    assert (result_DF_sample == result_DF).all().all()

    # version = 2 でのテスト
    # 入力を作成
    base_dict = {
        "functionName": [
            "functionName0",
            "functionName1",
            "functionName2",
            "functionName3",
            "functionName4",
            "functionName5",
        ],
        "lin": [3, 3, 1, 3, 1, 1],
        "ip": [2, 1, 3, 1, 3, 1],
        "log": [1, 2, 2, 1, 1, 3],
    }
    model_name_list = ["lin", "ip", "log"]
    version = 2
    input_DF = pd.DataFrame.from_dict(data=base_dict)
    input_DF = input_DF.set_index("functionName")
    # テスト結果を手動で作成
    result_DF_sample = input_DF.copy(deep=True)
    result_DF_sample["最低値"] = [1, 1, 1, 1, 1, 1]
    # テスト対象のテストを実行
    result_DF = addLowestMAPEColumn(
        inputDF=input_DF, model_name_list=model_name_list, version=version
    )
    # 結果を比較
    assert result_DF_sample["最低値"].to_list() == result_DF["最低値"].to_list()


# In[37]:


# 引数として渡された辞書からDFを返す関数
# 引数：辞書{ベンチマーク名:DF[関数名, モデル名1, ... , モデル名n, 最低MAPE]}
# 返値：DF[ベンチマーク名, 最低MAPEの平均]


def returnDFBenchmarkNameAndAverageLowestMAPE(inputDict):
    benchmarkNames = []
    lowestMAPEAverage = []

    for benchmarkName in inputDict.keys():
        averageSeries = inputDict[benchmarkName].mean()
        benchmarkNames.append(benchmarkName)
        lowestMAPEAverage.append(int(averageSeries["最低値"] * 10) / 10)

    resultDF = pd.DataFrame(
        index=benchmarkNames, data=lowestMAPEAverage, columns=["平均絶対誤差"]
    )
    return resultDF


def test_returnDFBenchmarkNameAndAverageLowestMAPE():
    function_names = ["func1", "func1", "func1"]
    lin_results = [1, 2, 2]
    ip_results = [2, 1, 3]
    log_results = [3, 3, 1]
    lowest_results = [1, 1, 1]
    input_DF_1 = pd.DataFrame(
        data={
            "関数名": function_names,
            "線形モデル": lin_results,
            "反比例モデル": ip_results,
            "対数モデル": log_results,
            "最低値": lowest_results,
        }
    )

    function_names = ["func2", "func2", "func2", "func2"]
    lin_results = [1, 2, 1, 2]
    ip_results = [1, 2, 1, 2]
    log_results = [1, 2, 1, 3]
    lowest_results = [1, 2, 1, 2]
    input_DF_2 = pd.DataFrame(
        data={
            "関数名": function_names,
            "線形モデル": lin_results,
            "反比例モデル": ip_results,
            "対数モデル": log_results,
            "最低値": lowest_results,
        }
    )

    input_dict = {"benchmark1": input_DF_1, "benchmark2": input_DF_2}
    returned_result = returnDFBenchmarkNameAndAverageLowestMAPE(input_dict)
    result_sample_DF = pd.DataFrame(
        index=["benchmark1", "benchmark2"], data=[1, 1.5], columns=["平均絶対誤差"]
    )

    pd.testing.assert_frame_equal(returned_result, result_sample_DF)


# In[38]:


# 引数に該当する生データを取得する関数
# 引数
# benchmark_names：ベンチマーク名のリスト
# classes：問題サイズのリスト
# processes：コア数のリスト
# csv_dir_path：CSVファイルを格納しているディレクトリのパス


def return_rawDF_with_init_param(
    benchmark_name="", classes=[], processes=[], csv_dir_path="./csv_files/"
):
    if benchmark_name == "":
        raise ExceptionInResearchLib(
            "return_rawDF_with_init_param()の引数benchmark_nameが空文字列です"
        )
    if classes == []:
        raise ExceptionInResearchLib("return_rawDF_with_init_param()の引数classesが空リストです")
    if processes == []:
        raise ExceptionInResearchLib(
            "return_rawDF_with_init_param()の引数processesが空リストです"
        )
    if os.path.exists(csv_dir_path) == False:
        raise ExceptionInResearchLib(
            "return_rawDF_with_init_param()の引数csv_dir_pathに該当するディレクトリが存在しません"
        )

    rawDF = returnCollectedExistingData(
        benchmarkNames=[benchmark_name],
        classes=classes,
        processes=processes,
        csvDirPath=csv_dir_path,
    )

    # 説明変数用に問題サイズを数値化した列を追加
    strListProblemSize = rawDF["benchmarkClass"].tolist()
    intListProblemSize = convertBenchmarkClasses_problemSizeInNPB(
        inputList=strListProblemSize
    )
    rawDF["intBenchmarkClass"] = intListProblemSize
    # 説明変数用に問題サイズ由来のほかの数値を保持する列を追加
    rawDF = addInitDataToRawDF(rawDF)

    return rawDF


def test_return_rawDF_with_init_param():
    classes = ["A", "B", "C", "D"]
    processes = [2, 4, 6, 8, 16, 32]
    csv_dir_path = "../csv_files/"
    # ベンチマークCGで実施
    benchmark_name = "cg"
    # ベンチマークCGの生データをテスト関数を用いて取得
    rawDF = return_rawDF_with_init_param(
        benchmark_name=benchmark_name,
        classes=classes,
        processes=processes,
        csv_dir_path=csv_dir_path,
    )
    # 初期変数として存在すべき値をリスト化
    init_param_names = ["na", "nonzer", "niter", "shift"]
    # 取得したDFに初期化変数があることを確認
    column_names_from_DF = rawDF.columns.tolist()
    for init_param_name in init_param_names:
        assert init_param_name in column_names_from_DF

    # ベンチマークFTで実施
    benchmark_name = "ft"
    # ベンチマークFTの生データをテスト関数を用いて取得
    rawDF = return_rawDF_with_init_param(
        benchmark_name=benchmark_name,
        classes=classes,
        processes=processes,
        csv_dir_path=csv_dir_path,
    )
    # 初期変数として存在すべき値をリスト化
    init_param_names = [
        "nx",
        "ny",
        "nz",
        "niter_default",
    ]
    # 取得したDFに初期化変数があることを確認
    column_names_from_DF = rawDF.columns.tolist()
    for init_param_name in init_param_names:
        assert init_param_name in column_names_from_DF


# In[39]:


# 引数として渡されたDFに最低MAPEのモデル名の列を追加する関数
# 引数 inputDF：DF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n], model_name_list
# 引数 model_name_list：モデル名が要素のリスト
# 引数 version：バージョン。1がオリジナル。
# 返値：DF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n, 最低MAPEのモデル名]


def addLowestMAPEsModelNameColumn(inputDF, model_name_list=[], version=1):
    """addLowestMAPEsModelNameColumn()の説明

    引数として渡されたDFに最低MAPEのモデル名の列を追加する関数

    Args:
        inputDF (pandas.DataFrame): DF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n]
        model_name_list (list): モデル名が要素のリスト
        version(int): バージョン。1がオリジナル。

    Returns:
        pandas.DataFrame: DF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n, 最低MAPEのモデル名]

    Raises:
        ExceptionInResearchLib: 引数に矛盾が生じている場合に発生

    Note:
        version=1の時のドキュメント消失。使うな。

    """

    if len(model_name_list) == 0 or version == 1:
        logger.warning(
            "addLowestMAPEsModelNameColumn()：version=1はすでに古い実装となっています。version=2の利用を検討してください。"
        )
        funcNames = inputDF.index.to_list()
        modelNames = inputDF.columns.to_list()

        inputDF["最適モデル"] = "lin"

        for funcName in funcNames:
            lowestInFunc_modelName = ""
            lowestInFunc = math.inf
            seriesInFunc = inputDF.loc[funcName]
            for modelName in modelNames:
                elem = seriesInFunc[modelName]
                if type(float(elem)) != type(float(lowestInFunc)):
                    logger.debug(
                        f"type(elem)={type(elem)}, type(lowestInFunc)={type(lowestInFunc)}"
                    )
                    logger.debug(f"elem={elem}, lowestInFunc={lowestInFunc}")
                if elem < lowestInFunc:
                    lowestInFunc = elem
                    lowestInFunc_modelName = modelName
                inputDF.at[funcName, "最適モデル"] = lowestInFunc_modelName

        return inputDF
    elif model_name_list != 0 and version == 2:
        func_names = inputDF.index.to_list()

        inputDF["最適モデル"] = model_name_list[0]

        for func_name in func_names:
            lowestInFunc_modelName = ""
            lowestInFunc = math.inf
            seriesInFunc = inputDF.loc[func_name]
            for model_name in model_name_list:
                elem = seriesInFunc[model_name]
                if type(float(elem)) != type(float(lowestInFunc)):
                    logger.debug(
                        f"type(elem)={type(elem)}, type(lowestInFunc)={type(lowestInFunc)}"
                    )
                    logger.debug(f"elem={elem}, lowestInFunc={lowestInFunc}")
                if elem < lowestInFunc:
                    lowestInFunc = elem
                    lowestInFunc_modelName = model_name
                inputDF.at[func_name, "最適モデル"] = lowestInFunc_modelName
        return inputDF
    else:
        ExceptionInResearchLib("addLowestMAPEsModelNameColumn()に渡された引数に矛盾が生じています")


def test_addLowestMAPEsModelNameColumn():
    # 入力を作成
    base_dict = {
        "lin": [1, 2, 2, 1],
        "ip": [2, 1, 3, 1],
        "log": [3, 3, 1, 1],
    }
    input_DF = pd.DataFrame.from_dict(data=base_dict)
    # テスト結果を手動で作成
    result_DF_sample = input_DF.copy(deep=True)
    result_DF_sample["最適モデル"] = ["lin", "ip", "log", "lin"]
    # テスト対象の関数を実行
    result_DF = addLowestMAPEsModelNameColumn(input_DF)
    # 結果を比較
    assert (result_DF_sample == result_DF).all().all()

    # 入力を作成
    base_dict = {
        "lin": [1, 2, 2, 1],
        "ip": [2, 1, 3, 1],
        "log": [3, 3, 1, 1],
    }
    input_DF = pd.DataFrame.from_dict(data=base_dict)
    # テスト結果を手動で作成
    result_DF_sample = input_DF.copy(deep=True)
    result_DF_sample["最適モデル"] = ["lin", "ip", "log", "lin"]
    # テスト対象の関数を実行
    result_DF = addLowestMAPEsModelNameColumn(
        input_DF, model_name_list=["lin", "ip", "log", "lin"], version=2
    )
    # 結果の確認
    assert input_DF["最適モデル"].to_list() == result_DF["最適モデル"].to_list()


# In[40]:


# 引数に該当するデータからMAPE（学習データに対する一致度）を各モデルごとにまとめたデータフレームを返す関数
# benchmarkName:ベンチマーク名
# classes:ベンチマーククラスのリスト
# targetClass:不要だがベンチマーククラスを指定できる
# processes:コア数のリスト
# targetProcess:不要だがコア数を指定できる
# expVar:学習に使用する列名もリスト
# csvDirPath:CSVの保存されているディレクトリへのパス
# 返り値：列名が["関数名", "モデル1", ... , "モデルN"]となっているDF


def returnDictAboutMAPETable(
    benchmarkName,
    classes,
    targetClass,
    processes,
    targetProcess,
    expVar,
    csvDirPath,
    modelNames=["modelLin", "modelIp", "modelLog"],
):
    """returnDictAboutMAPETable()の説明

    引数に該当するデータからMAPE（学習データに対する一致度）を各モデルごとにまとめたデータフレームを返す関数

    Args:
        benchmarkName (string): ベンチマーク名
        classes (list): ベンチマーククラスのリスト
        targetClass (list): 不要だがベンチマーククラスを指定できる
        processes (list): コア数のリスト
        targetProcess (int): 不要だがコア数を指定できる
        expVar (list): 学習に使用する列名もリスト
        csvDirPath (string): CSVの保存されているディレクトリへのパス
        modelNames (list): モデル名のリスト

    Returns:
        pandas.DataFrame: 列名が["関数名", "モデル1", ... , "モデルN"]となっているDF

    """

    # データを取得
    rawDF = returnCollectedExistingData(
        benchmarkNames=[benchmarkName],
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    rawDF = addInitDataToRawDF(rawDF)
    # 結果を格納するためのDFを作成
    # resultDF = pd.DataFrame(columns=["functionName"] + modelNames)
    result_series_list = []
    # モデルを用いた処理を実施
    functionNames = sorted(list(set(rawDF["functionName"])))
    for functionName in functionNames:
        ## 関数ごとに生データを取得
        DFExtractedByFunction = rawDF[rawDF["functionName"] == functionName].copy()
        ## 説明変数用に問題サイズ列を数値化した列を追加する
        strListProblemSize = DFExtractedByFunction["benchmarkClass"].tolist()
        intListProblemSize = convertBenchmarkClasses_problemSizeInNPB(
            inputList=strListProblemSize
        )
        DFExtractedByFunction["intBenchmarkClass"] = intListProblemSize
        ## 3モデルでMAPEを出力
        ### 目的変数のリストを作成
        resVar = ["functionCallNum"]
        ### 回帰を行う
        #### 3モデルを同時に作成
        models = Models(
            inputDF=DFExtractedByFunction,
            expVarColNames=expVar,
            resVarColNames=resVar,
            targetDF=None,
            modelNames=modelNames,
        )
        models.setUpDataBeforeCalcLr()
        models.calcLr()
        models.calcMAPE()
        dictCalcedMAPE = models.returnCalculatedMAPE()
        #### 計算されたMAPEの数値を小数第一位までにする
        for key in dictCalcedMAPE.keys():
            dictCalcedMAPE[key] = int(dictCalcedMAPE[key] * 10) / 10
        #### 関数ごとの結果に格納
        dict_for_series = {
            "functionName": functionName,
        }
        for key in dictCalcedMAPE.keys():
            dict_for_series[key] = dictCalcedMAPE[key]
        series = pd.Series(dict_for_series)
        result_series_list.append(series)
        # resultDF = resultDF.append(series, ignore_index=True)
    resultDF = pd.DataFrame(result_series_list)
    return resultDF


def test_returnDictAboutMAPETable(csvDirPath="../csv_files/"):
    ####
    # 予測を行う。一つの関数・変数（コア数・各種ベンチマーク由来の初期化変数）
    benchmarkNames = ["cg"]
    benchmarkName = "cg"
    classes = ["S", "W", "A", "B", "C", "D", "E", "F"]
    targetClass = "F"
    processes = [128]
    targetProcess = 256

    # データを取得
    rawDF = returnCollectedExistingData(
        benchmarkNames=benchmarkNames,
        classes=classes,
        processes=processes,
        csvDirPath=csvDirPath,
    )
    rawDF = addInitDataToRawDF(rawDF)

    # 結果(初期変数含む)を格納するためのDF
    # resultDictMulti = pd.DataFrame(
    #     columns=["functionName", "modelLin", "modelIp", "modelLog"]
    # )
    result_series_list_multi = []
    # 結果(初期変数含まない)を格納するためのDF
    # resultDictSingle = pd.DataFrame(
    #     columns=["functionName", "modelLin", "modelIp", "modelLog"]
    # )
    result_series_list_single = []

    # モデルを用いた処理を実施
    functionNames = sorted(list(set(rawDF["functionName"])))
    for functionName in functionNames:
        ##関数ごとにrawDFを抽出
        DFExtractedByFunction = rawDF[rawDF["functionName"] == functionName].copy()
        ## 説明変数用に問題サイズ列を数値化した列を追加する
        strListProblemSize = DFExtractedByFunction["benchmarkClass"].tolist()
        intListProblemSize = convertBenchmarkClasses_problemSizeInNPB(
            inputList=strListProblemSize
        )
        DFExtractedByFunction["intBenchmarkClass"] = intListProblemSize
        ## 3モデルでMAPEを出力
        ### 目的変数のリストを作成
        resVar = ["functionCallNum"]
        ### 重回帰分析（初期変数を含めた）
        #### 列名のリストをDFから取得
        expVarMulti = DFExtractedByFunction.columns.tolist()
        #### 除外する列名を除く
        for removeElement in [
            "functionName",
            "functionCallNum",
            "benchmarkName",
            "benchmarkClass",
        ]:
            expVarMulti.remove(removeElement)
        #### 3モデル（線形・反比例・対数）を同時に作成
        threeModelsByMulti = Models(
            inputDF=DFExtractedByFunction,
            expVarColNames=expVarMulti,
            resVarColNames=resVar,
            targetDF=None,
            modelNames=["modelLin", "modelIp", "modelLog"],
        )
        threeModelsByMulti.setUpDataBeforeCalcLr()
        threeModelsByMulti.calcLr()
        threeModelsByMulti.calcMAPE()
        dictCalcedMAPE = threeModelsByMulti.returnCalculatedMAPE()
        # dictCalcedMAPEの数値を小数第一位までにする
        for key in dictCalcedMAPE.keys():
            dictCalcedMAPE[key] = int(dictCalcedMAPE[key] * 10) / 10
        #### 関数ごとの結果に格納
        seriesMulti = pd.Series(
            {
                "functionName": functionName,
                "modelLin": dictCalcedMAPE["modelLin"],
                "modelIp": dictCalcedMAPE["modelIp"],
                "modelLog": dictCalcedMAPE["modelLog"],
            }
        )
        # resultDictMulti = resultDictMulti.append(seriesMulti, ignore_index=True)
        result_series_list_multi.append(seriesMulti)

        ### 単回帰分析（初期変数を含めない）
        expVarSingle = ["process", "intBenchmarkClass"]
        threeModelsBySingle = Models(
            inputDF=DFExtractedByFunction,
            expVarColNames=expVarSingle,
            resVarColNames=resVar,
            targetDF=None,
            modelNames=["modelLin", "modelIp", "modelLog"],
        )
        threeModelsBySingle.setUpDataBeforeCalcLr()
        threeModelsBySingle.calcLr()
        threeModelsBySingle.calcMAPE()
        dictCalcedMAPE = threeModelsBySingle.returnCalculatedMAPE()
        # dictCalcedMAPEの数値を小数第一位までにする
        for key in dictCalcedMAPE.keys():
            dictCalcedMAPE[key] = int(dictCalcedMAPE[key] * 10) / 10
        #### 関数ごとの結果に格納
        seriesSingle = pd.Series(
            {
                "functionName": functionName,
                "modelLin": dictCalcedMAPE["modelLin"],
                "modelIp": dictCalcedMAPE["modelIp"],
                "modelLog": dictCalcedMAPE["modelLog"],
            }
        )
        # resultDictSingle = resultDictSingle.append(seriesSingle, ignore_index=True)
        result_series_list_single.append(seriesSingle)

    resultDictMulti = pd.DataFrame(result_series_list_multi)
    resultDictSingle = pd.DataFrame(result_series_list_single)

    resultDictMultiFromFunc = returnDictAboutMAPETable(
        benchmarkName=benchmarkName,
        classes=classes,
        targetClass=targetClass,
        processes=processes,
        targetProcess=targetProcess,
        expVar=expVarMulti,
        csvDirPath=csvDirPath,
    )
    pd.testing.assert_frame_equal(
        resultDictMulti, resultDictMultiFromFunc, check_dtype=False
    )
    resultDictSingleFromFunc = returnDictAboutMAPETable(
        benchmarkName=benchmarkName,
        classes=classes,
        targetClass=targetClass,
        processes=processes,
        targetProcess=targetProcess,
        expVar=expVarSingle,
        csvDirPath=csvDirPath,
    )
    pd.testing.assert_frame_equal(
        resultDictSingle, resultDictSingleFromFunc, check_dtype=False
    )

    return 0


# In[41]:


class ModelMultipleEquationForMultipleRegression(ModelBaseForMultipleRegression):
    """組み合わせモデル（重回帰分析）

    組み合わせモデルを実現するためのクラス

    Attributes:
        equationDict (dict): キー・バリューが列名・変形モデル（線形、反比例、対数など）
        lr : モデルのオブジェクト
        dataXForPredict : 説明変数のDF
        dataTForPredict : 目的変数のDF

    """

    def __init__(
        self,
        inputDF,
        explanatoryVariableColumnNames,
        responseVariableColumnNames,
        equationDict,
        conditionDictForTest={},
        targetDF=None,
    ):
        super().__init__(
            inputDF,
            explanatoryVariableColumnNames,
            responseVariableColumnNames,
            conditionDictForTest={},
            targetDF=None,
        )
        self.equationDict = equationDict

    def inverterLog10Func(x):
        return 10**x

    def ipFunc(x):
        return 1 / x

    def transformDataForModel(self, inputDF):
        """transformDataForModel(self, inputDF)

        inputDFで与えられたデータをモデルに適した形に変形する

        Args:
            self : none
            inputDF (pandas.DataFrame): 変形されるDF

        Returns:
            pandas.DataFrame: inputDFをself.equationDictに沿って変形したDF
        """
        returnDF = inputDF.copy(deep=True)
        # equationDictのキーをループで回す
        for key in self.equationDict.keys():
            ## equationDictのバリューに合った形で変形を実施
            ### 線形モデルの場合
            if self.equationDict[key] == "lin":
                returnDF[key] = inputDF[key]
            ### 反比例モデルの場合
            elif self.equationDict[key] == "ip":
                returnDF[key] = 1 / inputDF[key]
            ### 対数モデルの場合
            elif self.equationDict[key] == "log":
                returnDF[key] = np.log10(inputDF[key])
            else:
                logger.warning(
                    f"not lin, ip, log what it is?(self.equationDict[key]={self.equationDict[key]})"
                )

        return returnDF

    def setUpDataBeforeCalcLr(self):
        """setUpDataBeforeCalcLr(self)

        transformDataForModel()を使って学習用データを変換

        Args:
            self : none
        """

        # モデル構築用データ
        self.dataXForPredict = self.transformDataForModel(self.rawExplanaoryVariable)
        self.dataTForPredict = self.rawResponseVariable
        #         self.dataTForPredict = self.transformDataForModel(
        #             self.rawResponseVariable)
        # テスト用データ
        self.dataXForTest = self.transformDataForModel(
            self.rawExplanaoryVariableForTest
        )
        self.dataTForTest = self.rawResponseVariableForTest

    #         self.dataTForTest = self.transformDataForModel(
    #             self.rawResponseVariableForTest)

    def calcLr(self):
        """calcLr(self)

        実際にモデルを構築する

        Args:
            self : none
            inputDF (pandas.DataFrame): 変形されるDF
        """
        self.lr = LinearRegression()
        self.lr.fit(self.dataXForPredict, self.dataTForPredict)

    def predict(self, inputDF):
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame): 構築されたモデルを用いて予測に使うDF

        Returns:
            pandas.DataFrame: 構築されたモデルから予測された値。型に確証なし
        """

        # inputDFから説明変数データのみを取得
        inputDFOnlyExplanatoryVariableColumn = inputDF[
            self.explanatoryVariableColumnNames
        ]
        # inputDFで与えられたデータをモデルに適した形に変形する
        transformedInputDF = self.transformDataForModel(
            inputDFOnlyExplanatoryVariableColumn
        )
        # 予測を実行
        result = self.lr.predict(transformedInputDF)

        return result

    def returnPredictedFromDataXForPredict(self):
        """returnPredictedFromDataXForPredict(self)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none

        Returns:
            pandas.DataFrame: モデルの構築に用いたデータから予測された値。型に確証なし
        """
        returnDatum = self.lr.predict(self.dataXForPredict)
        return returnDatum

    pass


def test_ModelMultipleEquationForMultipleRegression():
    """test_ModelMultipleEquationForMultipleRegression()

    ModelMultipleEquationForMultipleRegressionのテスト
    """

    # 説明変数
    plotX = np.linspace(10, 20, 10)
    plotY = np.linspace(10, 20, 10)
    plotZ = np.linspace(10, 20, 10)
    # 目的変数
    plotT = 10 * plotX + 15 / plotY + 20 * np.log10(plotZ) + 30

    # DFを作成する
    # カラム名のリスト
    columnNames = ["plotX", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = ["plotX", "plotY", "plotZ"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]
    # 各説明変数に対する式のリスト
    equationDict = {"plotX": "lin", "plotY": "ip", "plotZ": "log"}

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = ModelMultipleEquationForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
        equationDict=equationDict,
    )
    # モデルの生成の準備
    objectModel.setUpDataBeforeCalcLr()
    # モデルの生成
    objectModel.calcLr()
    # モデルによる予測
    # 入力データDFを作成
    inputDFForPredict = pd.DataFrame(inputDFForTest.head(1))
    predictedNum = objectModel.predict(inputDFForPredict)

    # 相対誤差率でテスト対象のデータが想定通りに動作しているかを判断する
    # 相対誤差率を計算するために実データを取得する
    realNum = plotT[0]
    relativeErrorRate = returnRelativeErrorRate(
        realNum=realNum, predictedNum=predictedNum
    )
    assert relativeErrorRate < 1


# In[42]:


# class Models
# 各モデルのオブジェクトデータを保持している。
# 学習用データ・予測対象データを保持している
# 引数名とその説明
# inputDF：入力データの全てを保持したDF（説明変数・目的変数・ベンチマーク名・関数名を最低限保持している）
# expVarColNames：inputDFの列名の中で、説明変数として用いるカラム名のリスト
# resVarColNames：inputDFの列名の中で、説明変数として用いるカラム名のリスト
# targetDF：inputDFとデータ構成は同じだが、予測対象のデータがセットされている
# modelNames：実施するモデル名を指定できる(["modelLin", "modelIp", "modelLog"])


class Models:
    """Models

    複数のモデルを管理して結果を出すためのクラス

    Attributes:
        inputDF (pandas.DataFrame) : 入力された生データ
        expVarColNames : 入力された生データにおける説明変数の列名
        resVarColNames : 入力された生データにおける目的変数の列名
    """

    def __init__(
        self,
        inputDF,
        expVarColNames,
        resVarColNames,
        targetDF=None,
        modelNames=["modelLin", "modelIp", "modelLog"],
    ):
        """__init__(self, inputDF, expVarColNames, resVarColNames, targetDF, modelNames)

        初期化関数

        Args:
            self : none
            inputDF (pandas.DataFrame) : 説明変数の値と目的変数の値を保持したDF
            expVarColNames (list) : inputDFにおいて説明変数の列名を保持したリスト
            resVarColNames (list) : inputDFにおいて目的変数の列名を保持したリスト
            targetDF (pandas.DataFrame) : DF。基本的に使用していない
            modelNames (list) : モデル名（modelLin, modelIp, modelLog）などを保持したリスト
        """
        self.inputDF = inputDF
        self.expVarColNames = expVarColNames
        if len(resVarColNames) > 1:
            warnings.warn("目的変数が複数個存在しています")
        self.resVarColNames = resVarColNames
        self.targetDF = targetDF
        self.functionName = ""
        self.benchmarkName = ""
        self.modelNames = modelNames

        if "modelLin" in self.modelNames:
            self.objectModelLin = ModelLinForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelIp" in self.modelNames:
            self.objectModelIp = ModelIpForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelLog" in self.modelNames:
            self.objectModelLog = ModelLogForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelProcessDividedByProblemSize" in self.modelNames:
            self.objectModelProcessDividedByProblemSize = (
                Model_ProcessesDevidedByProblemSize_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelProblemSizeDividedByProcess" in self.modelNames:
            self.objectModelProblemSizeDividedByProcess = (
                Model_ProblemSizeDevidedByProcesses_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelBasicTree" in self.modelNames:
            self.objectModelBasicTree = Model_BasicTree(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelInfiniteProductOfProblemSizeDividedByProcesses" in self.modelNames:
            self.objectModelInfiniteProductOfProblemSizeDividedByProcesses = Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelInfiniteProductOfProblemSizeMultipliedByProcesses" in self.modelNames:
            self.objectModelInfiniteProductOfProblemSizeMultipliedByProcesses = Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
            )
        if "modelLinAndIp" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "lin"
                else:
                    equation_dict[expVarElement] = "ip"
            # logger.debug(f"modelLinAndIp:equation_dict={equation_dict}")
            self.objectModelLinAndIp = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelLinAndLog" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "lin"
                else:
                    equation_dict[expVarElement] = "log"
            # logger.debug(f"modelLinAndLog:equation_dict={equation_dict}")
            self.objectModelLinAndLog = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelIpAndLin" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "ip"
                else:
                    equation_dict[expVarElement] = "lin"
            # logger.debug(f"modelIpAndLin:equation_dict={equation_dict}")
            self.objectModelIpAndLin = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelIpAndLog" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "ip"
                else:
                    equation_dict[expVarElement] = "log"
            # logger.debug(f"modelIpAndLog:equation_dict={equation_dict}")
            self.objectModelIpAndLog = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelLogAndLin" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "log"
                else:
                    equation_dict[expVarElement] = "lin"
            # logger.debug(f"modelLogAndLin:equation_dict={equation_dict}")
            self.objectModelLogAndLin = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelLogAndIp" in self.modelNames:
            equation_dict = {}
            for expVarElement in expVarColNames:
                if expVarElement == "process":
                    equation_dict[expVarElement] = "log"
                else:
                    equation_dict[expVarElement] = "ip"
            # logger.debug(f"modelLogAndIp:equation_dict={equation_dict}")
            self.objectModelLogAndIp = ModelMultipleEquationForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                equationDict=equation_dict,
            )
        if "modelLinearSumOf2elementCombination" in self.modelNames:
            self.objectModelLinearSumOf2elementCombination = (
                Model_LinearSumOf2elementCombination_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelLinearSumOfElementCombinations" in self.modelNames:
            self.objectModelLinearSumOfElementCombinations = (
                Model_LinearSumOfElementCombinations_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelLinearSumOf2elementCombinationWithCubed" in self.modelNames:
            self.objectModelLinearSumOf2elementCombinationWithCubed = (
                Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                    exponent=3,
                )
            )
        if "modelLinearSumOf2elementCombinationWithSquared" in self.modelNames:
            self.objectModelLinearSumOf2elementCombinationWithSquared = (
                Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                    exponent=2,
                )
            )
        if (
            "modelLinearSumOf2elementCombinationWithCubedWithoutProcess"
            in self.modelNames
        ):
            self.objectModelLinearSumOf2elementCombinationWithCubedWithoutProcess = Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                exponent=3,
            )
        if (
            "modelLinearSumOf2elementCombinationWithSquaredWithoutProcess"
            in self.modelNames
        ):
            self.objectModelLinearSumOf2elementCombinationWithSquaredWithoutProcess = Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression(
                inputDF,
                explanatoryVariableColumnNames=expVarColNames,
                responseVariableColumnNames=resVarColNames,
                targetDF=targetDF,
                exponent=2,
            )
        if "modelSquareRootOfProcess" in self.modelNames:
            self.objectModelSquareRootOfProcess = (
                Model_squareRootOfProcess_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelSquareRootTimesOtherElems" in self.modelNames:
            self.objectModelSquareRootTimesOtherElems = (
                Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )
        if "modelObeyOneParameter" in self.modelNames:
            self.objectModelObeyOneParameter = (
                Model_obeyOneParameter_ForMultipleRegression(
                    inputDF,
                    explanatoryVariableColumnNames=expVarColNames,
                    responseVariableColumnNames=resVarColNames,
                    targetDF=targetDF,
                )
            )

    def setUpDataBeforeCalcLr(self):
        """setUpDataBeforeCalcLr(self)

        各モデルを構築する前処理をする関数

        Args:
            self : none
        """
        if "modelLin" in self.modelNames:
            self.objectModelLin.setUpDataBeforeCalcLr()
        if "modelIp" in self.modelNames:
            self.objectModelIp.setUpDataBeforeCalcLr()
        if "modelLog" in self.modelNames:
            self.objectModelLog.setUpDataBeforeCalcLr()
        if "modelLinAndIp" in self.modelNames:
            self.objectModelLinAndIp.setUpDataBeforeCalcLr()
        if "modelLinAndLog" in self.modelNames:
            self.objectModelLinAndLog.setUpDataBeforeCalcLr()
        if "modelIpAndLin" in self.modelNames:
            self.objectModelIpAndLin.setUpDataBeforeCalcLr()
        if "modelIpAndLog" in self.modelNames:
            self.objectModelIpAndLog.setUpDataBeforeCalcLr()
        if "modelLogAndLin" in self.modelNames:
            self.objectModelLogAndLin.setUpDataBeforeCalcLr()
        if "modelLogAndIp" in self.modelNames:
            self.objectModelLogAndIp.setUpDataBeforeCalcLr()

    def calcLr(self):
        """calcLr(self)

        各モデルを構築する関数

        Args:
            self : none
        """
        if "modelLin" in self.modelNames:
            self.objectModelLin.calcLr()
        if "modelIp" in self.modelNames:
            self.objectModelIp.calcLr()
        if "modelLog" in self.modelNames:
            self.objectModelLog.calcLr()
        if "modelProcessDividedByProblemSize" in self.modelNames:
            self.objectModelProcessDividedByProblemSize.build_model()
        if "modelProblemSizeDividedByProcess" in self.modelNames:
            self.objectModelProblemSizeDividedByProcess.build_model()
        if "modelLinAndIp" in self.modelNames:
            self.objectModelLinAndIp.calcLr()
        if "modelLinAndLog" in self.modelNames:
            self.objectModelLinAndLog.calcLr()
        if "modelIpAndLin" in self.modelNames:
            self.objectModelIpAndLin.calcLr()
        if "modelIpAndLog" in self.modelNames:
            self.objectModelIpAndLog.calcLr()
        if "modelLogAndLin" in self.modelNames:
            self.objectModelLogAndLin.calcLr()
        if "modelLogAndIp" in self.modelNames:
            self.objectModelLogAndIp.calcLr()
        if "modelBasicTree" in self.modelNames:
            self.objectModelBasicTree.build_model()
        if "modelInfiniteProductOfProblemSizeMultipliedByProcesses" in self.modelNames:
            self.objectModelInfiniteProductOfProblemSizeMultipliedByProcesses.build_model()
        if "modelInfiniteProductOfProblemSizeDividedByProcesses" in self.modelNames:
            self.objectModelInfiniteProductOfProblemSizeDividedByProcesses.build_model()
        if "modelLinearSumOf2elementCombination" in self.modelNames:
            self.objectModelLinearSumOf2elementCombination.build_model()
        if "modelLinearSumOfElementCombinations" in self.modelNames:
            self.objectModelLinearSumOfElementCombinations.build_model()
        if "modelLinearSumOf2elementCombinationWithSquared" in self.modelNames:
            self.objectModelLinearSumOf2elementCombinationWithSquared.build_model()
        if "modelLinearSumOf2elementCombinationWithCubed" in self.modelNames:
            self.objectModelLinearSumOf2elementCombinationWithCubed.build_model()
        if (
            "modelLinearSumOf2elementCombinationWithCubedWithoutProcess"
            in self.modelNames
        ):
            self.objectModelLinearSumOf2elementCombinationWithCubedWithoutProcess.build_model()
        if (
            "modelLinearSumOf2elementCombinationWithSquaredWithoutProcess"
            in self.modelNames
        ):
            self.objectModelLinearSumOf2elementCombinationWithSquaredWithoutProcess.build_model()
        if "modelSquareRootOfProcess" in self.modelNames:
            self.objectModelSquareRootOfProcess.build_model()
        if "modelSquareRootTimesOtherElems" in self.modelNames:
            self.objectModelSquareRootTimesOtherElems.build_model()
        if "modelObeyOneParameter" in self.modelNames:
            self.objectModelObeyOneParameter.build_model()

    # inputDF：__init__()でのinputDFとDF構成は同じ
    def predict(self, inputDF):
        pass

    def calcMAPE(self):
        """calcMAPE(self)
        学習用データへの適合度（MAPE[%]）を計算する

        Args:
            self : none
        """
        # MAPEatTrain:辞書
        # キーはmodelNamesの要素、バリューは学習データの適合度としてのMAPE
        MAPEatTrain = {}
        if len(self.resVarColNames) > 1:
            warnings.warn("目的変数が複数カラムに及んでいるため、正常な動作を期待できません")
        realData = self.inputDF[self.resVarColNames[0]].tolist()

        if "modelLin" in self.modelNames:
            predictedDataAtLin = self.objectModelLin.predict(
                self.inputDF[self.expVarColNames]
            )

            modelLinMAPEatTrain = returnMapeScore(realData, predictedDataAtLin)
            MAPEatTrain["modelLin"] = modelLinMAPEatTrain
        if "modelIp" in self.modelNames:
            predictedDataAtIp = self.objectModelIp.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpMAPEatTrain = returnMapeScore(realData, predictedDataAtIp)
            MAPEatTrain["modelIp"] = modelIpMAPEatTrain
        if "modelLog" in self.modelNames:
            predictedDataAtLog = self.objectModelLog.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogMAPEatTrain = returnMapeScore(realData, predictedDataAtLog)
            MAPEatTrain["modelLog"] = modelLogMAPEatTrain
        if "modelProcessDividedByProblemSize" in self.modelNames:
            MAPEatTrain[
                "modelProcessDividedByProblemSize"
            ] = self.objectModelProcessDividedByProblemSize.returnMAPE()
        if "modelProblemSizeDividedByProcess" in self.modelNames:
            MAPEatTrain[
                "modelProblemSizeDividedByProcess"
            ] = self.objectModelProblemSizeDividedByProcess.returnMAPE()
        if "modelBasicTree" in self.modelNames:
            MAPEatTrain["modelBasicTree"] = self.objectModelBasicTree.returnMAPE()
        if "modelInfiniteProductOfProblemSizeDividedByProcesses" in self.modelNames:
            MAPEatTrain[
                "modelInfiniteProductOfProblemSizeDividedByProcesses"
            ] = (
                self.objectModelInfiniteProductOfProblemSizeDividedByProcesses.returnMAPE()
            )
        if "modelInfiniteProductOfProblemSizeMultipliedByProcesses" in self.modelNames:
            MAPEatTrain[
                "modelInfiniteProductOfProblemSizeMultipliedByProcesses"
            ] = (
                self.objectModelInfiniteProductOfProblemSizeMultipliedByProcesses.returnMAPE()
            )
        if "modelLinAndIp" in self.modelNames:
            predictedDataAtLinAndIp = self.objectModelLinAndIp.predict(
                self.inputDF[self.expVarColNames]
            )

            modelLinAndIpMAPEatTrain = returnMapeScore(
                realData, predictedDataAtLinAndIp
            )
            MAPEatTrain["modelLinAndIp"] = modelLinAndIpMAPEatTrain
        if "modelLinAndLog" in self.modelNames:
            predictedFromBuildDatum = (
                self.objectModelLinAndLog.returnPredictedFromDataXForPredict()
            )

            modelLinAndLogMAPEatTrain = returnMapeScore(
                realData, predictedFromBuildDatum
            )

            MAPEatTrain["modelLinAndLog"] = modelLinAndLogMAPEatTrain
        if "modelIpAndLin" in self.modelNames:
            predictedDataAtIpAndLin = self.objectModelIpAndLin.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpAndLinMAPEatTrain = returnMapeScore(
                realData, predictedDataAtIpAndLin
            )
            MAPEatTrain["modelIpAndLin"] = modelIpAndLinMAPEatTrain
        if "modelIpAndLog" in self.modelNames:
            predictedDataAtIpAndLog = self.objectModelIpAndLog.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpAndLogMAPEatTrain = returnMapeScore(
                realData, predictedDataAtIpAndLog
            )
            MAPEatTrain["modelIpAndLog"] = modelIpAndLogMAPEatTrain
        if "modelLogAndLin" in self.modelNames:
            predictedDataAtLogAndLin = self.objectModelLogAndLin.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogAndLinMAPEatTrain = returnMapeScore(
                realData, predictedDataAtLogAndLin
            )
            MAPEatTrain["modelLogAndLin"] = modelLogAndLinMAPEatTrain
        if "modelLogAndIp" in self.modelNames:
            predictedDataAtLogAndIp = self.objectModelLogAndIp.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogAndIpMAPEatTrain = returnMapeScore(
                realData, predictedDataAtLogAndIp
            )
            MAPEatTrain["modelLogAndIp"] = modelLogAndIpMAPEatTrain
        if "modelLinearSumOf2elementCombination" in self.modelNames:
            MAPEatTrain[
                "modelLinearSumOf2elementCombination"
            ] = self.objectModelLinearSumOf2elementCombination.returnMAPE()
        if "modelLinearSumOfElementCombinations" in self.modelNames:
            MAPEatTrain[
                "modelLinearSumOfElementCombinations"
            ] = self.objectModelLinearSumOfElementCombinations.returnMAPE()
        if "modelLinearSumOf2elementCombinationWithSquared" in self.modelNames:
            MAPEatTrain[
                "modelLinearSumOf2elementCombinationWithSquared"
            ] = self.objectModelLinearSumOf2elementCombinationWithSquared.returnMAPE()
        if "modelLinearSumOf2elementCombinationWithCubed" in self.modelNames:
            MAPEatTrain[
                "modelLinearSumOf2elementCombinationWithCubed"
            ] = self.objectModelLinearSumOf2elementCombinationWithCubed.returnMAPE()
        if (
            "modelLinearSumOf2elementCombinationWithCubedWithoutProcess"
            in self.modelNames
        ):
            MAPEatTrain[
                "modelLinearSumOf2elementCombinationWithCubedWithoutProcess"
            ] = (
                self.objectModelLinearSumOf2elementCombinationWithCubedWithoutProcess.returnMAPE()
            )
        if (
            "modelLinearSumOf2elementCombinationWithSquaredWithoutProcess"
            in self.modelNames
        ):
            MAPEatTrain[
                "modelLinearSumOf2elementCombinationWithSquaredWithoutProcess"
            ] = (
                self.objectModelLinearSumOf2elementCombinationWithSquaredWithoutProcess.returnMAPE()
            )
        if "modelSquareRootOfProcess" in self.modelNames:
            MAPEatTrain[
                "modelSquareRootOfProcess"
            ] = self.objectModelSquareRootOfProcess.returnMAPE()
        if "modelSquareRootTimesOtherElems" in self.modelNames:
            MAPEatTrain[
                "modelSquareRootTimesOtherElems"
            ] = self.objectModelSquareRootTimesOtherElems.returnMAPE()
        if "modelObeyOneParameter" in self.modelNames:
            MAPEatTrain[
                "modelObeyOneParameter"
            ] = self.objectModelObeyOneParameter.returnMAPE()

        self.MAPEatTrain = MAPEatTrain

    def calcWeightedMAPE(self):
        """calcWeightedMAPE(self)

        学習用データへの重み付きMAPEを計算する

        Args:
            self
        """

        WeightedMAPEatTrain: dict[float] = {}

        if len(self.resVarColNames) > 1:
            warnings.warn("目的変数が複数カラムに及んでいるため、正常な動作を期待できません。")
        realData: list[float] = self.inputDF[self.resVarColNames[0]].tolist()

        if "modelLin" in self.modelNames:
            predictedDataAtLin: list[float] = self.objectModelLin.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLinWeightedMAPEatTrain: float = returnWeightedMapeScore(
                real=realData, predicted=predictedDataAtLin
            )
            WeightedMAPEatTrain["modelLin"] = modelLinWeightedMAPEatTrain

        if "modelIp" in self.modelNames:
            predictedDataAtIp: list[float] = self.objectModelIp.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpWeightedMAPEatTrain: float = returnWeightedMapeScore(
                real=realData, predicted=predictedDataAtIp
            )
            WeightedMAPEatTrain["modelIp"] = modelIpWeightedMAPEatTrain

        if "modelLog" in self.modelNames:
            predictedDataAtLog: list[float] = self.objectModelLog.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogWeightedMAPEatTrain = returnWeightedMapeScore(
                real=realData, predicted=predictedDataAtLog
            )
            WeightedMAPEatTrain["modelLog"] = modelLogWeightedMAPEatTrain

        if "modelProcessDividedByProblemSize" in self.modelNames:
            WeightedMAPEatTrain[
                "modelProcessDividedByProblemSize"
            ] = self.objectModelProcessDividedByProblemSize.returnWeightedMAPE()
        if "modelProblemSizeDividedByProcess" in self.modelNames:
            WeightedMAPEatTrain[
                "modelProblemSizeDividedByProcess"
            ] = self.objectModelProblemSizeDividedByProcess.returnWeightedMAPE()

        if "modelLinearSumOf2elementCombination" in self.modelNames:
            WeightedMAPEatTrain[
                "modelLinearSumOf2elementCombination"
            ] = self.objectModelLinearSumOf2elementCombination.returnWeightedMAPE()
        if "modelLinearSumOfElementCombinations" in self.modelNames:
            WeightedMAPEatTrain[
                "modelLinearSumOfElementCombinations"
            ] = self.objectModelLinearSumOfElementCombinations.returnWeightedMAPE()

        if "modelLinAndIp" in self.modelNames:
            predictedDataAtLinAndIp = self.objectModelLinAndIp.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLinAndIpMAPEatTrain = returnWeightedMapeScore(
                realData, predictedDataAtLinAndIp
            )
            WeightedMAPEatTrain["modelLinAndIp"] = modelLinAndIpMAPEatTrain
        if "modelLinAndLog" in self.modelNames:
            predictedFromBuildDatum = (
                self.objectModelLinAndLog.returnPredictedFromDataXForPredict()
            )
            modelLinAndLogMAPEatTrain = returnWeightedMapeScore(
                realData, predictedFromBuildDatum
            )
            WeightedMAPEatTrain["modelLinAndLog"] = modelLinAndLogMAPEatTrain
        if "modelIpAndLin" in self.modelNames:
            predictedDataAtIpAndLin = self.objectModelIpAndLin.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpAndLinMAPEatTrain = returnWeightedMapeScore(
                realData, predictedDataAtIpAndLin
            )
            WeightedMAPEatTrain["modelIpAndLin"] = modelIpAndLinMAPEatTrain
        if "modelIpAndLog" in self.modelNames:
            predictedDataAtIpAndLog = self.objectModelIpAndLog.predict(
                self.inputDF[self.expVarColNames]
            )
            modelIpAndLogMAPEatTrain = returnWeightedMapeScore(
                realData, predictedDataAtIpAndLog
            )
            WeightedMAPEatTrain["modelIpAndLog"] = modelIpAndLogMAPEatTrain
        if "modelLogAndLin" in self.modelNames:
            predictedDataAtLogAndLin = self.objectModelLogAndLin.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogAndLinMAPEatTrain = returnWeightedMapeScore(
                realData, predictedDataAtLogAndLin
            )
            WeightedMAPEatTrain["modelLogAndLin"] = modelLogAndLinMAPEatTrain
        if "modelLogAndIp" in self.modelNames:
            predictedDataAtLogAndIp = self.objectModelLogAndIp.predict(
                self.inputDF[self.expVarColNames]
            )
            modelLogAndIpMAPEatTrain = returnWeightedMapeScore(
                realData, predictedDataAtLogAndIp
            )
            WeightedMAPEatTrain["modelLogAndIp"] = modelLogAndIpMAPEatTrain

        self.WeightedMAPEatTrain = WeightedMAPEatTrain

    def returnCalculatedMAPE(self):
        """returnCalculatedMAPE(self)

        Args:
            self : none

        Returns:
            Dict: calcMAPEで計算した辞書を返す関数。返す辞書がない場合は空の辞書を返す
        """
        if self.MAPEatTrain is None:
            return {}
        else:
            return self.MAPEatTrain

    def returnCalculatedWeightedMAPE(self):
        """returnCalculatedWeightedMAPE(self)

        Args:
            self : none

        Returns:
            Dict: calcMAPEで計算した辞書を返す関数。返す辞書がない場合は空の辞書を返す
        """
        if self.WeightedMAPEatTrain is None:
            return {}
        else:
            return self.WeightedMAPEatTrain

    # 引数 targetDF:本オブジェクト構築時に必要になるinputDFをデータ構造が同じDF
    def calcRelativeErrorRate(self, targetDF=None):
        """calcRelativeErrorRate(self, targetDF=None)

        予測対象データとの相対誤差率を計算する

        Args:
            self : none
        """
        # relativeErrorRateDict:辞書
        # キーはmodelNamesの要素、バリューは絶対相対誤差率
        relativeErrorRateDict = {}
        # （すでに予測対象の説明変数データがある or targetDF is not None）なら問題ない。
        if (self.targetDF is None) and (targetDF is None):
            warnings.warn("相対誤差率を計算するための真値が与えられていません。")
            return -1

        if len(self.resVarColNames) == 0:
            warnings.warn("説明変数のカラム名が複数設定されています")
        # targetDFがNoneの場合
        if targetDF is None:
            if len(self.targetDF) > 1:
                warnings.warn("ターゲットとなるDFに要素が2つ以上含まれています。")
            realData = self.targetDF[self.resVarColNames[0]]
            _targetDF = self.targetDF
        # self.targetDFがNoneの場合
        else:
            if len(targetDF) > 1:
                warnings.warn("ターゲットとなるDFに要素が2つ以上含まれています。")
            realData = targetDF[self.resVarColNames[0]]
            _targetDF = targetDF

        # realData は DataFrame なので、それをリスト化して、最初の要素のみ保持する
        realData = realData.tolist()[0]

        if "modelLin" in self.modelNames:
            predictedData = self.objectModelLin.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLin"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelIp" in self.modelNames:
            predictedData = self.objectModelIp.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelIp"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelLog" in self.modelNames:
            predictedData = self.objectModelLog.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLog"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelLinAndIp" in self.modelNames:
            predictedData = self.objectModelLinAndIp.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLinAndIp"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelLinAndLog" in self.modelNames:
            predictedData = self.objectModelLinAndLog.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLinAndLog"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelIpAndLin" in self.modelNames:
            predictedData = self.objectModelIpAndLin.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelIpAndLin"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelIpAndLog" in self.modelNames:
            predictedData = self.objectModelIpAndLog.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelIpAndLog"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelLogAndLin" in self.modelNames:
            predictedData = self.objectModelLogAndLin.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLogAndLin"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        if "modelLogAndIp" in self.modelNames:
            predictedData = self.objectModelLogAndIp.predict(
                _targetDF[self.expVarColNames]
            ).tolist()[0][0]
            relativeErrorRateDict["modelLogAndIp"] = returnRelativeErrorRate(
                realNum=realData, predictedNum=predictedData
            )
        self.relativeErrorRateDict = relativeErrorRateDict

    def returnObject(self, modelName: str):
        """returnObject(self, modelName :str)

        構築したモデルオブジェクトを返す関数

        Args:
            self : none
            modelName (str) : モデル名
        """
        if "modelLin" == modelName:
            return self.objectModelLin
        if "modelIp" == modelName:
            return self.objectModelIp
        if "modelLog" == modelName:
            return self.objectModelLog
        if "modelProcessDividedByProblemSize" == modelName:
            return self.objectModelProcessDividedByProblemSize
        if "modelProblemSizeDividedByProcess" == modelName:
            return self.objectModelProblemSizeDividedByProcess
        if "modelLinAndIp" == modelName:
            return self.objectModelLinAndIp
        if "modelLinAndLog" == modelName:
            return self.objectModelLinAndLog
        if "modelIpAndLin" == modelName:
            return self.objectModelIpAndLin
        if "modelIpAndLog" == modelName:
            return self.objectModelIpAndLog
        if "modelLogAndLin" == modelName:
            return self.objectModelLogAndLin
        if "modelLogAndIp" == modelName:
            return self.objectModelLogAndIp
        if "modelBasicTree" == modelName:
            return self.objectModelBasicTree
        if "modelInfiniteProductOfProblemSizeMultipliedByProcesses" == modelName:
            return self.objectModelInfiniteProductOfProblemSizeMultipliedByProcesses
        if "modelInfiniteProductOfProblemSizeDividedByProcesses" == modelName:
            return self.objectModelInfiniteProductOfProblemSizeDividedByProcesses
        if "modelLinearSumOf2elementCombination" == modelName:
            return self.objectModelLinearSumOf2elementCombination
        if "modelLinearSumOfElementCombinations" == modelName:
            return self.objectModelLinearSumOfElementCombinations
        if "modelLinearSumOf2elementCombinationWithSquared" == modelName:
            return self.objectModelLinearSumOf2elementCombinationWithSquared
        if "modelLinearSumOf2elementCombinationWithCubed" == modelName:
            return self.objectModelLinearSumOf2elementCombinationWithCubed
        if "modelLinearSumOf2elementCombinationWithCubedWithoutProcess" == modelName:
            return self.objectModelLinearSumOf2elementCombinationWithCubedWithoutProcess
        if "modelLinearSumOf2elementCombinationWithSquaredWithoutProcess" == modelName:
            return (
                self.objectModelLinearSumOf2elementCombinationWithSquaredWithoutProcess
            )
        if "modelSquareRootOfProcess" == modelName:
            return self.objectModelSquareRootOfProcess
        if "modelSquareRootTimesOtherElems" == modelName:
            return self.objectModelSquareRootTimesOtherElems
        if "modelObeyOneParameter" == modelName:
            return self.objectModelObeyOneParameter

    def returnRelativeErrorRateDict(self):
        """returnRelativeErrorRateDict(self)

        calcRelativeErrorRate()で計算した辞書を返す関数

        Args:
            引数の名前 (引数の型): 引数の説明
            引数の名前 (:obj:`引数の型`, optional): 引数の説明.

        Returns:
            Dict : 返す辞書がない場合は空の辞書を返す
        """

        if self.relativeErrorRateDict is None:
            return {}
        else:
            return self.relativeErrorRateDict

    def updateFunctionAndBenchmarkName(self, functionName=None, benchmarkName=None):
        """updateFunctionAndBenchmarkName(self, functionName=None, benchmarkName=None)

        関数名・ベンチマーク名を更新する

        Args:
            self : none
            functionName (str) : 変更後の関数名
            benchmarkName (str) : 変更後のベンチマーク名
        """
        # 各種引数が空の場合は更新しない
        if functionName is not None:
            self.functionName = functionName
        if benchmarkName is not None:
            self.benchmarkName = benchmarkName

    def returnFunctionName(self):
        return self.functionName

    def returnBenchmarkName(self):
        return self.benchmarkName

    def returnExpVarDatumDF(self):
        return self.inputDF[self.expVarColNames]

    def returnResVarDatumDF(self):
        return self.inputDF[self.resVarColNames]

    def returnModelsName(self):
        return self.modelNames


#         self.inputDF = inputDF
#         self.expVarColNames = expVarColNames
#         if len(resVarColNames) > 1:
#             warnings.warn("説明変数が複数個存在しています")
#         self.resVarColNames = resVarColNames
#         self.targetDF = targetDF
#         self.functionName = ""
#         self.benchmarkName = ""
#         self.modelNames = modelNames


def test_Models():
    """test_Models()

    クラスModelsを更新する
    """
    # inputDFを準備
    # 説明変数
    plotX = np.linspace(1, 20, 10)
    plotY = np.linspace(21, 40, 10)
    plotZ = np.linspace(41, 60, 10)
    # 目的変数
    plotTforLin = plotX + 2 * plotY + 3 * plotZ + 4
    plotTforIp = 1 / plotX + 2 / plotY + 3 / plotZ + 4
    plotTforLog = np.log10(plotX) + 2 * np.log10(plotY) + 3 * np.log10(plotZ) + 4
    inputDF = pd.DataFrame(
        {
            "plotX": plotX,
            "plotY": plotY,
            "plotZ": plotZ,
            "plotTforLin": plotTforLin,
            "plotTforIp": plotTforIp,
            "plotTforLog": plotTforLog,
        }
    )

    # functionNameを準備
    functionName = "functionName"
    # benchmarkNameを準備
    benchmarkName = "benchmarkName"

    inputDF[functionName] = functionName
    inputDF[benchmarkName] = benchmarkName
    # targetDFを準備
    targetDF = inputDF.tail(1)

    columnNames = inputDF.columns.tolist()
    # expVarColNamesを準備
    expVarColNames = columnNames[:3]
    # resVarColNamesを準備
    resVarColNames = columnNames[-3:]
    resVarColNamesForLin = ["plotTforLin"]
    resVarColNamesForIp = ["plotTforIp"]
    resVarColNamesForLog = ["plotTforLog"]

    # インスタンスを作成
    modelsLin = Models(
        inputDF=inputDF,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNamesForLin,
        targetDF=targetDF,
    )
    modelsIp = Models(
        inputDF=inputDF,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNamesForIp,
        targetDF=targetDF,
    )
    modelsLog = Models(
        inputDF=inputDF,
        expVarColNames=expVarColNames,
        resVarColNames=resVarColNamesForLog,
        targetDF=targetDF,
    )
    # 予測に必要な初期化作業を開始
    modelsLin.setUpDataBeforeCalcLr()
    modelsIp.setUpDataBeforeCalcLr()
    modelsLog.setUpDataBeforeCalcLr()
    # モデル構築を実施
    modelsLin.calcLr()
    modelsIp.calcLr()
    modelsLog.calcLr()

    # 予測をして学習データに対するMAPEを計算し、その値がそれぞれ小さいことを確認
    modelsLin.calcMAPE()
    dictCalcedMAPEatLin = modelsLin.returnCalculatedMAPE()
    modelsIp.calcMAPE()
    dictCalcedMAPEatIp = modelsIp.returnCalculatedMAPE()
    modelsLog.calcMAPE()
    dictCalcedMAPEatLog = modelsLog.returnCalculatedMAPE()
    assert dictCalcedMAPEatLin["modelLin"] < 0.001
    assert dictCalcedMAPEatIp["modelIp"] < 0.001
    assert dictCalcedMAPEatLog["modelLog"] < 0.001

    # 相対誤差率を計算し、それが小さいことを確認
    modelsLin.calcRelativeErrorRate(targetDF=targetDF)
    modelsIp.calcRelativeErrorRate(targetDF=targetDF)
    modelsLog.calcRelativeErrorRate(targetDF=targetDF)
    relativeErrorRateDictAtLin = modelsLin.returnRelativeErrorRateDict()
    relativeErrorRateDictAtIp = modelsIp.returnRelativeErrorRateDict()
    relativeErrorRateDictAtLog = modelsLog.returnRelativeErrorRateDict()
    assert relativeErrorRateDictAtLin["modelLin"] < 0.0001
    assert relativeErrorRateDictAtIp["modelIp"] < 0.0001
    assert relativeErrorRateDictAtLog["modelLog"] < 0.0001

    # 関数名・ベンチマーク名を更新する関数のテスト
    modelsLin.updateFunctionAndBenchmarkName(functionName=functionName)
    modelsIp.updateFunctionAndBenchmarkName(functionName=functionName)
    modelsLog.updateFunctionAndBenchmarkName(functionName=functionName)
    assert (
        modelsLin.returnFunctionName() == functionName
        and modelsIp.returnFunctionName() == functionName
        and modelsLog.returnFunctionName() == functionName
    )
    assert (
        modelsLin.returnBenchmarkName() == ""
        and modelsIp.returnBenchmarkName() == ""
        and modelsLog.returnBenchmarkName() == ""
    )
    modelsLin.updateFunctionAndBenchmarkName(benchmarkName=benchmarkName)
    modelsIp.updateFunctionAndBenchmarkName(benchmarkName=benchmarkName)
    modelsLog.updateFunctionAndBenchmarkName(benchmarkName=benchmarkName)
    assert (
        modelsLin.returnFunctionName() == functionName
        and modelsIp.returnFunctionName() == functionName
        and modelsLog.returnFunctionName() == functionName
    )
    assert (
        modelsLin.returnBenchmarkName() == benchmarkName
        and modelsIp.returnBenchmarkName() == benchmarkName
        and modelsLog.returnBenchmarkName() == benchmarkName
    )


# In[43]:


def return_MAPE_Table_DF_from_rawDF(
    rawDF, exp_var_list=[], res_var_list=[], model_name_list=[]
):
    """return_MAPE_Table_DF_from_rawDF()の説明

    引数として渡された生データ入りDFから各モデルでのMAPEを記録したDFを返す関数

    Args:
        rawDF (pandas.DataFrame): DF["functionName<必須列>", "データ列名0<全ての要素は数値>" ... ,"データ列名N<全ての要素は数値>"]
        model_name_list (list): モデル名が要素のリスト
        version(int): バージョン。1がオリジナル。

    Returns:
        pandas.DataFrame: DF[関数名(列名ではなくインデックス), モデル名1, ... , モデル名n]

    Raises:
        ExceptionInResearchLib: 引数のリストが空の場合, 渡された生DFに列名「"functionName"」が存在しない場合に発生

    """

    if exp_var_list == []:
        raise ExceptionInResearchLib("説明変数として利用するカラム名のリストが空です")
    if res_var_list == []:
        raise ExceptionInResearchLib("目的変数として利用するカラム名のリストが空です")
    if model_name_list == []:
        raise ExceptionInResearchLib("構築するモデル名のリストが空です")

    function_names = list(set(rawDF["functionName"].tolist()))
    if len(function_names) == 0:
        raise ExceptionInResearchLib("与えられた生データのDFに関数名がありません")

    # 結果を格納するためのDFを作成
    # resultDF = pd.DataFrame(columns=["functionName"] + model_name_list)
    result_series_list = []

    for function_name in function_names:
        # 関数ごとの生データ
        rawDF_per_function = rawDF[rawDF["functionName"] == function_name]
        # モデルの構築
        models = Models(
            inputDF=rawDF_per_function,
            expVarColNames=exp_var_list,
            resVarColNames=res_var_list,
            targetDF=None,
            modelNames=model_name_list,
        )
        models.setUpDataBeforeCalcLr()
        models.calcLr()
        models.calcMAPE()

        # MAPEの算出
        dictCalcedMAPE = models.returnCalculatedMAPE()

        # 算出されたMAPEの数値をfloatにする
        for key in dictCalcedMAPE.keys():
            dictCalcedMAPE[key] = float(dictCalcedMAPE[key])

        # 関数ごとの結果に格納
        dict_for_series = copy.deepcopy(dictCalcedMAPE)
        dict_for_series["functionName"] = function_name

        series = pd.Series(dict_for_series)
        # TODO:下記のappend()をconcat()にする
        # resultDF = resultDF.append(series, ignore_index=True)
        result_series_list.append(series)
    resultDF = pd.DataFrame(result_series_list)
    return resultDF


def test_return_MAPE_Table_DF_from_rawDF():
    # 構築するモデルのリスト
    model_name_list = ["modelLin", "modelIp", "modelLog"]
    # モデル構築用のDFを作成
    plotX = np.linspace(10, 20, 10)
    plotY = np.linspace(20, 30, 10)
    plotZ = np.linspace(30, 40, 10)
    function_name_list = ["function_name"] * 10
    benchmark_name_list = ["benchmark_name"] * 10
    exp_var_list = ["plotX", "plotY", "plotZ"]
    res_var_list = ["plotT"]
    plotT = 100 * plotX - 500 * plotY + 0.1 * plotZ + 40
    rawDF = pd.DataFrame(
        data={
            "benchmarkName": benchmark_name_list,
            "functionName": function_name_list,
            "plotT": plotT,
            "plotX": plotX,
            "plotY": plotY,
            "plotZ": plotZ,
        }
    )
    # テスト対象の関数からDFを取得
    returnedDF = return_MAPE_Table_DF_from_rawDF(
        rawDF=rawDF,
        exp_var_list=exp_var_list,
        res_var_list=res_var_list,
        model_name_list=model_name_list,
    )
    # modelLinが最小であることを確認
    assert returnedDF.at[0, "modelLin"] < returnedDF.at[0, "modelIp"]
    assert returnedDF.at[0, "modelLin"] < returnedDF.at[0, "modelLog"]

    # 組み合わせモデル(線形+反比例)
    # 構築するモデルのリスト
    model_name_list = [
        "modelLin",
        "modelIp",
        "modelLog",
        "modelLinAndLog",
        "modelLinAndIp",
    ]
    # モデル構築用のDFを作成
    plotX = np.linspace(10, 20, 10)
    plotY = np.linspace(20, 30, 10)
    plotZ = np.linspace(30, 40, 10)
    function_name_list = ["function_name"] * 10
    benchmark_name_list = ["benchmark_name"] * 10
    exp_var_list = ["process", "plotY", "plotZ"]
    res_var_list = ["plotT"]
    plotT = 100 * plotX + -100 * plotY + 40 / plotZ + 30
    rawDF = pd.DataFrame(
        data={
            "benchmarkName": benchmark_name_list,
            "functionName": function_name_list,
            "plotT": plotT,
            "process": plotX,
            "plotY": plotY,
            "plotZ": plotZ,
        }
    )
    # テスト対象の関数からDFを取得
    returnedDF = return_MAPE_Table_DF_from_rawDF(
        rawDF=rawDF,
        exp_var_list=exp_var_list,
        res_var_list=res_var_list,
        model_name_list=model_name_list,
    )
    # modelLinAndIpが最小であることを確認
    for model_name in model_name_list:
        if model_name == "modelLinAndIp":
            pass
        else:
            assert returnedDF.at[0, "modelLinAndIp"] < returnedDF.at[0, model_name]

    return


# In[44]:


class Model_ProcessesDevidedByProblemSize_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """プロセス数を問題サイズの変数群の線形和で割ったモデル

    組み合わせモデルを実現するためのクラス

    Attributes:
        equationDict (dict): キー・バリューが列名・変形モデル（線形、反比例、対数など）
        lr : モデルのオブジェクト
        dataXForPredict : 説明変数のDF
        dataTForPredict : 目的変数のDF
    Note:
        モデルの式は次の通り
        関数コール回数 = (プロセス数)/(a * 問題サイズ1 + b * 問題サイズ2 + ... + n * 問題サイズn + c)
    """

    def build_model(self) -> bool:
        """build_model(self)

        inputDFのデータからモデル構築する。

        Args:
            self : none

        Returns: boolean。成功ならTrue,失敗ならFalse

        Note:
            必ず、説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。
        """

        # 説明変数を格納したDFに"process"列名がない場合は警告を出力
        if ("process" in self.rawExplanaoryVariable.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return False

        # 説明変数を格納したDFの一列目の列名が"process"でない場合は、一列目を"process"とする
        if "process" == self.rawExplanaoryVariable.columns.to_list()[0]:
            process_column = self.rawExplanaoryVariable.pop("process")
            self.rawExplanaoryVariable.insert(0, "process", process_column)

        # list_exp:説明変数のリスト
        # list_expをlist_inputとして利用
        list_exp: np.ndarray = self.rawExplanaoryVariable.to_numpy().T
        # list_res:目的変数のリスト
        list_res: np.ndarray = self.rawResponseVariable.to_numpy()
        # list_resが複数の列を持っていると予測ができるか不確かなため、警告を出す
        if list_res.shape[1] != 1:
            warnings.warn("目的変数の個数が想定と異なります")
            return False
        list_res = list_res.ravel()

        # モデルの構築
        self.popt: np.ndarray
        self.pcov: np.ndarray

        self.dataXForPredict: np.ndarray = list_exp
        self.dataTForPredict: np.ndarray = list_res

        # 説明変数の数だけp0用の1埋め配列を作成する
        list_p0: list[float] = [1] * (len(list_exp) + 1)

        self.popt, self.pcov = curve_fit(
            processesDevidedByProblemSize, list_exp, list_res, list_p0, maxfev=1000000
        )

        return True

    def predict(self, inputDF) -> np.ndarray:
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame): 構築されたモデルを用いて予測に使うDF

        Returns:
            pandas.DataFrame: 構築されたモデルから予測された値。型に確証なし

        Note:
            必ず、説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。
        """

        # inputDFとモデルの構築に用いた説明変数のDFの列の順番が同じことを確認
        if inputDF.columns.to_list() != self.rawExplanaoryVariable.columns.to_list():
            warnings.warn(
                f"inputDFとモデルの構築に用いた説明変数のDF列の順番が異なります。\ninputDF.columns.to_list()[{inputDF.columns.to_list()}] != self.rawExplanaoryVariable.to_list()[{self.rawExplanaoryVariable.to_list()}]"
            )
            return -1

        # inputDFに"process"列名がない場合は警告を出力
        if ("process" in inputDF.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return False

        # inputDFの一列目の列名が"process"でない場合は、一列目を"process"とする
        if "process" == inputDF.columns.to_list()[0]:
            process_column = inputDF.pop("process")
            inputDF.insert(0, "process", process_column)

        # inputDFから引数list_inputとして使われる変数ndarray_inputDFを作成
        ndarray_inputDF: np.ndarray = inputDF.to_numpy().T

        predicted_result: np.ndarray = processesDevidedByProblemSize(
            ndarray_inputDF, *self.popt
        )
        return predicted_result

    def returnMAPE(self) -> float:
        """calcMAPE(self)

        モデルの構築に使用されたデータからMAPEを算出する

        Args:
            self : none

        Returns:
            float: MAPE
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray[float] = self.rawResponseVariable.to_numpy().ravel()
        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray 長さ[{len(predicted_result)}]と実際の値の ndarray の長さ[{len(real_data)}]が異なります"
            )
        mape: float = returnMapeScore(l1=predicted_result, l2=real_data)
        return mape

    def returnWeightedMAPE(self) -> float:
        """returnWeightedMAPE(self) -> float

        モデルの構築に使用されたデータからweightedMAPEを算出する

        Args:
            self: none

        Returns:
            float: weightedMAPE
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray[float] = self.rawResponseVariable.to_numpy().ravel()
        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray 長さ[{len(predicted_result)}]と実際の値の ndarray の長さ[{len(real_data)}]が異なります"
            )
        weightedMape: float = returnWeightedMapeScore(
            real=real_data, predicted=predicted_result
        )
        return weightedMape


# モデル式の宣言
def processesDevidedByProblemSize(
    list_input: list[np.ndarray] = [], *list_coef_inte: list[float]
) -> list[float]:
    """processesDevidedByProblemSize(list_input: list[np.ndarray] = [], *list_coef_inte :list[float])

    inputDFのデータから構築されたモデルを使って予測を行う

    Args:
        list_input : 変数の入った行列。現状は一般の行列と異なり [[<列データ1>], [<列データ2>], ... , [<列データn>]]の形式
        list_coef_inte : 係数と切片の入ったリスト。最後尾の要素が切片でそれ以外は係数。係数とデータの関係はlist_input,list_coef_inteのインデックス番号に一対一対応している。

    Returns:
        np.ndarray[float] : 計算された値。
    """
    # list_inputの要素数と有効なa,b,c,d,eの個数が同じことを確認
    if len(list_input) != len(list_coef_inte) - 1:
        warnings.warn(
            f"len(list_input)[={len(list_input)}] != len(list_coef_inte)-1[={len(list_coef_inte)-1}]"
        )

    result: list[float] = []

    for i in range(len(list_input[0])):
        numerator: float = 0
        denominator: float = 0
        for j in range(len(list_input)):
            if j == 0:
                numerator = list_input[j][i] * list_coef_inte[j]
            else:
                denominator += list_input[j][i] * list_coef_inte[j]

        result.append(numerator / denominator + list_coef_inte[-1])

    return result


def test_processesDevidedByProblemSize():
    """test_processesDevidedByProblemSize()

    processesDevidedByProblemSizeのテスト
    """

    list_A: list[int] = [1, 2, 3, 4]
    list_B: list[int] = [10, 20, 30, 40]
    list_C: list[int] = [100, 200, 300, 400]

    a: int = 5
    b: int = 6
    c: int = 7
    d: int = 8

    list_T_expect: list[int] = []
    for i in range(len(list_A)):
        numerator: float = a * list_A[i]
        denominator: float = b * list_B[i] + c * list_C[i]
        list_T_expect.append(numerator / denominator + d)

    list_input_for_actually: list[list[float]] = [list_A, list_B, list_C]
    list_T_actually: np.ndarray = processesDevidedByProblemSize(
        list_input_for_actually, a, b, c, d
    )

    assert (
        list_T_expect == list_T_actually
    ), f"expect = {list_T_expect}, actually = {list_T_actually}"


def test_Model_ProcessesDevidedByProblemSize_ForMultipleRegression():
    """test_ModelMultipleEquationForMultipleRegression()

    ModelMultipleEquationForMultipleRegressionのテスト
    """

    # ____test_case_01____

    # 説明変数
    plotX_1 = np.linspace(10, 20, 11)
    plotX_2 = 10 * np.linspace(10, 20, 11)
    plotX_3 = 100 * np.linspace(10, 20, 11)
    plotX_4 = 1000 * np.linspace(10, 20, 11)
    plotX_5 = 10000 * np.linspace(10, 20, 11)
    # 目的変数
    a = 10
    b = 20
    c = 30
    d = 40
    e = 50
    f = 50
    plotT = (a * plotX_1) / (b * plotX_2 + c * plotX_3 + d * plotX_4 + e * plotX_5) + f

    # DFを作成する
    # カラム名のリスト
    columnNames = ["process", "plotX_2", "plotX_3", "plotX_4", "plotX_5", "plotT"]
    datumForDF = [plotX_1, plotX_2, plotX_3, plotX_4, plotX_5, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = ["process", "plotX_2", "plotX_3", "plotX_4", "plotX_5"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = Model_ProcessesDevidedByProblemSize_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape(____test_case_01____) = {mape}"

    # ____test_case_02____

    # 説明変数
    plotX = np.linspace(10, 20, 11)
    plotY = np.linspace(20, 30, 11)
    plotZ = np.linspace(30, 40, 11)
    # 目的変数
    a = 11
    b = 13
    c = 17
    d = 19
    plotT = (a * plotX) / (b * plotY + c * plotZ) + d

    # DFを作成する
    # カラム名のリスト
    columnNames = ["process", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = ["process", "plotY", "plotZ"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = Model_ProcessesDevidedByProblemSize_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape(____test_case_02____) = {mape}"


# In[45]:


def test_Model_BasicTree():
    """回帰木による予測を行うモデル

    「回帰木による予測を実現するためのクラス」のテストを実施する

    Attributes:

    Note:
    """

    # テストデータはsklearn付属のものを使用
    california_housing = fetch_california_housing()
    list_exp: list[str] = california_housing.feature_names
    list_res: list[str] = ["target"]
    df_data: pd.DataFrame = pd.DataFrame(california_housing.data, columns=list_exp)
    df_target: pd.DataFrame = pd.DataFrame(california_housing.target, columns=list_res)
    df_california_housing: pd.DataFrame = pd.concat([df_data, df_target], axis=1)
    array_california_housing: np.ndarray = df_california_housing.values
    array_x_california_housing: np.ndarray = array_california_housing[:, 0:-1]
    array_y_california_housing: np.ndarray = array_california_housing[:, -1]
    df_california_housing["functionName"] = "functionName"
    # テストデータによるモデルの構築
    reg = DecisionTreeRegressor(max_leaf_nodes=20)
    model: DecisionTreeRegressor = reg.fit(
        array_x_california_housing, array_y_california_housing
    )
    # テストデータによる予測
    YHat: np.ndarray = model.predict(array_x_california_housing)
    r2: float = r2_score(array_y_california_housing, YHat)
    # 構築したモデルによるMAPEの算出
    mape_expect: float = returnMapeScore(array_y_california_housing, YHat)

    # クラスを用いたモデル構築
    objectModel = Model_BasicTree(
        inputDF=df_california_housing,
        explanatoryVariableColumnNames=list_exp,
        responseVariableColumnNames=list_res,
        conditionDictForTest={},
    )
    objectModel.build_model()
    # クラスを用いたMAPEの受け取り
    mape_actually: float = objectModel.returnMAPE()
    assert math.isclose(
        mape_expect, mape_actually
    ), f"mape_expect({mape_expect}) != mape_actually({mape_actually})"


class Model_BasicTree(ModelBaseForMultipleRegression):
    """回帰木による予測を行うモデル

    回帰木による予測を実現するためのクラス

    Attributes:
        basicTree : モデルのオブジェクト
        dataXForPredict : 説明変数のDF
        dataTForPredict : 目的変数のDF
    Note:
    """

    def build_model(self) -> bool:
        """build_model(self)

        inputDFのデータからモデル構築する。

        Args:
            self : none

        Returns: boolean。成功ならTrue,失敗ならFalse
        """
        self.array_dataXForPredict: np.ndarray = self.rawExplanaoryVariable.values
        self.array_dataTForPredict: np.ndarray = self.rawResponseVariable.values

        self.reg = DecisionTreeRegressor(max_leaf_nodes=20)
        self.model: DecisionTreeRegressor = self.reg.fit(
            self.array_dataXForPredict, self.array_dataTForPredict
        )
        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame): 構築されたモデルを用いて予測に使うDF

        Returns:
            pandas.DataFrame: 構築されたモデルから予測された値。型に確証なし

        Note:
        """

        # モデル構築に用いた説明変数のカラム名と、与えられたDFのカラム名が一致していることを確認
        if self.explanatoryVariableColumnNames != inputDF.columns.to_list():
            warnings.warn(
                f"self.explanatoryVariableColumnNames != inputDF.columns.to_list(), {self.explanatoryVariableColumnNames} != {inputDF.columns.to_list()}"
            )
            inputDF = inputDF[self.explanatoryVariableColumnNames]
        # .predict()する
        array_forPredicting: np.ndarray = inputDF.values
        return_value: np.ndarray = self.model.predict(array_forPredicting)
        # .predict()で返された値を返す
        return return_value

    def returnMAPE(self) -> float:
        """calcMAPE(self)

        モデルの構築に使用されたデータからMAPEを算出する

        Args:
            self : none

        Returns:
            list: モデルの構築に用いたデータから予測された値
            int: 失敗した場合、-1

        Note:
        """

        array_real_data: np.ndarray = self.rawResponseVariable.values
        array_predicted_data: np.ndarray = self.predict(self.rawExplanaoryVariable)
        return_num: float = float(
            returnMapeScore(array_real_data, array_predicted_data)
        )

        return return_num


# In[46]:


class Model_ProblemSizeDevidedByProcesses_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """問題サイズの変数群の線形和をプロセス数で割ったモデル

    問題サイズの変数群の線形和をプロセス数で割ったモデルを実現するためのクラス

    Attributes:
        equationDict (dict): キー・バリューが列名・変形モデル（線形、反比例、対数など）
        lr : モデルのオブジェクト
        dataXForPredict : 説明変数のDF
        dataTForPredict : 目的変数のDF
    Note:
        モデルの式は次の通り
        関数コール回数 = (a * 問題サイズ1 + b * 問題サイズ2 + ... + n * 問題サイズn)/(プロセス数) + c
    """

    def build_model(self) -> bool:
        """build_model(self)

        inputDFのデータからモデル構築する。

        Args:
            self : none

        Returns: boolean。成功ならTrue,失敗ならFalse

        Note:
            必ず、説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。
        """

        # 説明変数を格納したDFに"process"列名がない場合は警告を出力
        if ("process" in self.rawExplanaoryVariable.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return False

        # 説明変数を格納したDFの一列目の列名が"process"でない場合は、一列目を"process"とする
        if "process" == self.rawExplanaoryVariable.columns.to_list()[0]:
            process_column = self.rawExplanaoryVariable.pop("process")
            self.rawExplanaoryVariable.insert(0, "process", process_column)

        # list_exp:説明変数のリスト
        # list_expをlist_inputとして利用
        list_exp: np.ndarray = self.rawExplanaoryVariable.to_numpy().T
        # list_res:目的変数のリスト
        list_res: np.ndarray = self.rawResponseVariable.to_numpy()
        # list_resが複数の列を持っていると予測ができるか不確かなため、警告を出す
        if list_res.shape[1] != 1:
            warnings.warn("目的変数の個数が想定と異なります")
            return False
        list_res = list_res.ravel()

        # モデルの構築
        self.popt: np.ndarray
        self.pcov: np.ndarray

        self.dataXForPredict: np.ndarray = list_exp
        self.dataTForPredict: np.ndarray = list_res

        # 説明変数の数だけp0用の1埋め配列を作成する
        list_p0: list[float] = [1] * (len(list_exp) + 1)

        self.popt, self.pcov = curve_fit(
            problemSizeDevidedByProcesses, list_exp, list_res, list_p0, maxfev=1000000
        )

        return True

    def predict(self, inputDF) -> np.ndarray:
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame): 構築されたモデルを用いて予測に使うDF

        Returns:
            pandas.DataFrame: 構築されたモデルから予測された値。型に確証なし

        Note:
            必ず、説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。
        """

        # inputDFとモデルの構築に用いた説明変数のDFの列の順番が同じことを確認
        if inputDF.columns.to_list() != self.rawExplanaoryVariable.columns.to_list():
            warnings.warn(
                f"inputDFとモデルの構築に用いた説明変数のDF列の順番が異なります。\ninputDF.columns.to_list()[{inputDF.columns.to_list()}] != self.rawExplanaoryVariable.to_list()[{self.rawExplanaoryVariable.to_list()}]"
            )
            return -1

        # inputDFに"process"列名がない場合は警告を出力
        if ("process" in inputDF.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return False

        # inputDFの一列目の列名が"process"でない場合は、一列目を"process"とする
        if "process" == inputDF.columns.to_list()[0]:
            process_column = inputDF.pop("process")
            inputDF.insert(0, "process", process_column)

        # inputDFから引数list_inputとして使われる変数ndarray_inputDFを作成
        ndarray_inputDF: np.ndarray = inputDF.to_numpy().T

        predicted_result: np.ndarray = problemSizeDevidedByProcesses(
            ndarray_inputDF, *self.popt
        )
        return predicted_result

    def returnMAPE(self) -> float:
        """calcMAPE(self)

        モデルの構築に使用されたデータからMAPEを算出する

        Args:
            self : none

        Returns:
            list: モデルの構築に用いたデータから予測された値
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray[float] = self.rawResponseVariable.to_numpy().ravel()
        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray 長さ[{len(predicted_result)}]と実際の値の ndarray の長さ[{len(real_data)}]が異なります"
            )
        mape: float = returnMapeScore(l1=predicted_result, l2=real_data)
        return mape

    def returnWeightedMAPE(self) -> float:
        """returnWeightedMAPE(self) -> float

        モデルの構築に使用されたデータからweightedMAPEを算出する

        Args:
            self: none

        Returns:
            float: weightedMAPE
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray[float] = self.rawResponseVariable.to_numpy().ravel()
        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray 長さ[{len(predicted_result)}]と実際の値の ndarray の長さ[{len(real_data)}]が異なります"
            )
        weightedMape: float = returnWeightedMapeScore(
            real=real_data, predicted=predicted_result
        )
        return weightedMape


# モデル式の宣言
def problemSizeDevidedByProcesses(
    list_input: list[np.ndarray] = [], *list_coef_inte: list[float]
) -> list[float]:
    """problemSizeDevidedByProcesses(list_input: list[np.ndarray] = [], *list_coef_inte :list[float])

    inputDFのデータから構築されたモデルを使って予測を行う

    Args:
        list_input : 変数の入った行列。現状は一般の行列と異なり [[<列データ1>], [<列データ2>], ... , [<列データn>]]の形式
        list_coef_inte : 係数と切片の入ったリスト。最後尾の要素が切片でそれ以外は係数。係数とデータの関係はlist_input,list_coef_inteのインデックス番号に一対一対応している。

    Returns:
        np.ndarray[float] : 計算された値。
    """
    # list_inputの要素数と有効なa,b,c,d,eの個数が同じことを確認
    if len(list_input) != len(list_coef_inte) - 1:
        warnings.warn(
            f"len(list_input)[={len(list_input)}] != len(list_coef_inte)-1[={len(list_coef_inte)-1}]"
        )

    result: list[float] = []

    for i in range(len(list_input[0])):
        numerator: float = 0
        denominator: float = 0
        for j in range(len(list_input)):
            if j == 0:
                denominator = list_input[j][i] * list_coef_inte[j]
            else:
                numerator += list_input[j][i] * list_coef_inte[j]

        result.append(numerator / denominator + list_coef_inte[-1])

    return result


def test_problemSizeDevidedByProcesses():
    """problemSizeDevidedByProcesses()

    problemSizeDevidedByProcessesのテスト
    """

    list_A: list[int] = [1, 2, 3, 4]
    list_B: list[int] = [10, 20, 30, 40]
    list_C: list[int] = [100, 200, 300, 400]

    a: int = 5
    b: int = 6
    c: int = 7
    d: int = 8

    list_T_expect: list[int] = []
    for i in range(len(list_A)):
        denominator: float = a * list_A[i]
        numerator: float = b * list_B[i] + c * list_C[i]
        list_T_expect.append(numerator / denominator + d)

    list_input_for_actually: list[list[float]] = [list_A, list_B, list_C]
    list_T_actually: np.ndarray = problemSizeDevidedByProcesses(
        list_input_for_actually, a, b, c, d
    )

    assert (
        list_T_expect == list_T_actually
    ), f"expect = {list_T_expect}, actually = {list_T_actually}"


def test_Model_ProblemSizeDevidedByProcesses_ForMultipleRegression():
    """test_ModelMultipleEquationForMultipleRegression()

    ModelMultipleEquationForMultipleRegressionのテスト
    """

    # ____test_case_01____

    # 説明変数
    plotX_1 = np.linspace(10, 20, 11)
    plotX_2 = 10 * np.linspace(10, 20, 11)
    plotX_3 = 100 * np.linspace(10, 20, 11)
    plotX_4 = 1000 * np.linspace(10, 20, 11)
    plotX_5 = 10000 * np.linspace(10, 20, 11)
    # 目的変数
    a = 10
    b = 20
    c = 30
    d = 40
    e = 50
    f = 50
    plotT = (b * plotX_2 + c * plotX_3 + d * plotX_4 + e * plotX_5) / (a * plotX_1) + f

    # DFを作成する
    # カラム名のリスト
    columnNames = ["process", "plotX_2", "plotX_3", "plotX_4", "plotX_5", "plotT"]
    datumForDF = [plotX_1, plotX_2, plotX_3, plotX_4, plotX_5, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = ["process", "plotX_2", "plotX_3", "plotX_4", "plotX_5"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = Model_ProblemSizeDevidedByProcesses_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape(____test_case_01____) = {mape}"

    # ____test_case_02____

    # 説明変数
    plotX = np.linspace(10, 20, 11)
    plotY = np.linspace(20, 30, 11)
    plotZ = np.linspace(30, 40, 11)
    # 目的変数
    a = 11
    b = 13
    c = 17
    d = 19
    plotT = (b * plotY + c * plotZ) / (a * plotX) + d

    # DFを作成する
    # カラム名のリスト
    columnNames = ["process", "plotY", "plotZ", "plotT"]
    datumForDF = [plotX, plotY, plotZ, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp = ["process", "plotY", "plotZ"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = Model_ProblemSizeDevidedByProcesses_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape(____test_case_02____) = {mape}"


# In[47]:


def test_Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression():
    """test_Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression()
    Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegressionのテスト
    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 10 * np.linspace(10, 20, 11)
    plotX_3: np.ndarray = 100 * np.linspace(10, 20, 11)
    plotX_4: np.ndarray = 1000 * np.linspace(10, 20, 11)
    plotX_5: np.ndarray = 10000 * np.linspace(10, 20, 11)
    # 目的変数
    a: int = 10
    f: int = 50
    plotT: np.ndarray = (a * plotX_1 * plotX_2 * plotX_3 * plotX_4 * plotX_5) + f

    # DFを作成する
    # カラム名のリスト
    columnNames: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
        "plotX_4",
        "plotX_5",
        "plotT",
    ]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotX_4, plotX_5, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
        "plotX_4",
        "plotX_5",
    ]
    # 説明変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = (
        Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression(
            inputDF=inputDFForTest,
            explanatoryVariableColumnNames=columnNamesForExp,
            responseVariableColumnNames=columnNamesForRes,
            conditionDictForTest={},
        )
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


def test_Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression():
    """test_Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression()
    Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegressionのテスト
    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 10 * np.linspace(10, 20, 11)
    plotX_3: np.ndarray = 100 * np.linspace(10, 20, 11)
    plotX_4: np.ndarray = 1000 * np.linspace(10, 20, 11)
    plotX_5: np.ndarray = 10000 * np.linspace(10, 20, 11)
    # 目的変数
    a: int = 10
    f: int = 50
    plotT: np.ndarray = a * (plotX_2 * plotX_3 * plotX_4 * plotX_5) / plotX_1 + f

    # DFを作成する
    # カラム名のリスト
    columnNames: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
        "plotX_4",
        "plotX_5",
        "plotT",
    ]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotX_4, plotX_5, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
        "plotX_4",
        "plotX_5",
    ]
    # 説明変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = (
        Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression(
            inputDF=inputDFForTest,
            explanatoryVariableColumnNames=columnNamesForExp,
            responseVariableColumnNames=columnNamesForRes,
            conditionDictForTest={},
        )
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[48]:


def test_infiniteProductOfProblemSizeMultipliedByProcesses():
    """test_infiniteProductOfProblemSizeMultipliedByProcesses()

    infiniteProductOfProblemSizeMultipliedByProcesses()のテスト
    """

    list_a: list[int] = [1, 2, 3, 4]
    list_b: list[int] = [10, 20, 30, 40]
    list_c: list[int] = [100, 200, 300, 400]

    a: int = 5
    f: int = 6

    list_T_expect: list[int] = []
    for i in range(len(list_a)):
        list_T_expect.append(a * list_a[i] * list_b[i] * list_c[i] + f)

    list_input_for_actually: list[list[float]] = [list_a, list_b, list_c]
    list_T_actually: np.ndarray = infiniteProductOfProblemSizeMultipliedByProcesses(
        list_input_for_actually, a, f
    )

    assert (
        list_T_expect == list_T_actually
    ), f"expect={list_T_expect}, actually={list_T_actually}"


def test_infiniteProductOfProblemSizeDividedByProcesses():
    """test_infiniteProductOfProblemSizeDividedByProcesses()

    infiniteProductOfProblemSizeDividedByProcesses()のテスト
    """

    list_a: list[int] = [1, 2, 3, 4]
    list_b: list[int] = [10, 20, 30, 40]
    list_c: list[int] = [100, 200, 300, 400]

    a: int = 5
    f: int = 6

    list_T_expect: list[float] = []
    for i in range(len(list_a)):
        list_T_expect.append(a * (list_b[i] * list_c[i]) / (list_a[i]) + f)

    list_input_for_actually: list[list[int]] = [list_a, list_b, list_c]
    list_T_actually: np.ndarray = infiniteProductOfProblemSizeDividedByProcesses(
        list_input_for_actually, a, f
    )

    assert (
        list_T_expect == list_T_actually
    ), f"expect={list_T_expect}, actually={list_T_actually}"


# In[49]:


def infiniteProductOfProblemSizeMultipliedByProcesses(
    list_input: list[np.ndarray] = [], *list_coef_inte: list[float]
) -> list[float]:
    """
    infiniteProductOfProblemSizeMultipliedByProcesses(list_input: list[np.ndarray] = [], *list_coef_inte: list[float])

    a * 問題サイズの総乗 * プロセス数 + b

    を計算する

    Args:
        list_input : 変数の入った行列。一般の行列とは異なり [[<列データ1>], [<列データ2>], ... , [<列データn>]]の形式。<列データ1> がプロセス数に関するデータが入っていると想定している。
        list_coef_inte : 係数と切片の入ったリスト。list_coef_inte[0]が係数でlist_coef_inte[1]が切片。

    Returns:
        np.ndarray[float] : 計算された値
    """

    # 切片と係数はそれぞれ1つなので、それらを格納したリストの要素数が2つであることを確認
    if len(list_coef_inte) != 2:
        warnings.warn(f"len(list_coef_inte) = {len(list_coef_inte)}, but should be 2.")

    result: list[float] = []

    for i in range(len(list_input[0])):
        term: float = list_coef_inte[0]
        for j in range(len(list_input)):
            term *= list_input[j][i]
        result.append(term + list_coef_inte[1])

    return result


def infiniteProductOfProblemSizeDividedByProcesses(
    list_input: list[np.ndarray] = [], *list_coef_inte: list[float]
) -> list[float]:
    """
    infiniteProductOfProblemSizeDividedByProcesses(list_input: list[np.ndarray] = [], *list_coef_inte: list[float])

    a * 問題サイズの総乗 / プロセス数 + b

    を計算する

    Args:
        list_input : 変数の入った行列。一般の行列とは異なり [[<列データ1>], [<列データ2>], ... , [<列データn>]]の形式。<列データ1> がプロセス数に関するデータが入っていると想定している。
        list_coef_inte : 係数と切片の入ったリスト。list_coef_inte[0]が係数でlist_coef_inte[1]が切片。

    Returns:
        np.ndarray[float] : 計算された値
    """

    # 切片と係数はそれぞれ1つなので、それらを格納したリストの要素数が2つであることを確認
    if len(list_coef_inte) != 2:
        warnings.warn(f"len(list_coef_inte) = {len(list_coef_inte)}, but should be 2.")

    result: list[float] = []

    for i in range(len(list_input[0])):
        numerator: float = list_coef_inte[0]
        denominator: float = 0
        for j in range(len(list_input)):
            if j == 0:
                denominator += list_input[j][i]
            else:
                numerator *= list_input[j][i]
        term: float = numerator / denominator
        result.append(term + list_coef_inte[1])

    return result


# In[50]:


class Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """問題サイズの値の総乗とプロセス数をかけたモデル
    Model_InfiniteProductOfProblemSizeMultipliedByProcesses_ForMultipleRegression(ModelBaseForMultipleRegression)

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def build_model(self) -> bool:
        """build_model(self)

        inputDFのデータからモデルを構築する。

        Args:
            self : none

        Returns: boolean。成功ならTrue、失敗ならFalse

        Note:
            必ず、説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。

        """

        # 説明変数を格納したDFに列名 "process" がない場合は警告を出力し失敗扱いにする
        if ("process" in self.explanatoryVariableColumnNames) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return False

        # 説明変数を格納したDFの1列目の列名が "process" 出ない場合は、1列名を "process" とする
        if "process" == self.explanatoryVariableColumnNames[0]:
            process_column = self.rawExplanaoryVariable.pop("process")
            self.rawExplanaoryVariable.insert(0, "process", process_column)

        # list_exp:説明変数のリスト
        # list_expをlist_inputとして利用
        list_exp: np.ndarray = self.rawExplanaoryVariable.to_numpy().T
        # list_res:目的変数のリスト
        list_res: np.ndarray = self.rawResponseVariable.to_numpy()
        # list_resが複数の列を持っている(=DF的な解釈では列が複数ある)と予測ができるか不確かなため、警告を出す
        if list_res.shape[1] != 1:
            warnings.warn("目的変数の個数が想定と異なります")
            return False
        # list_resを list[float] のような形式にする
        list_res = list_res.ravel()

        # モデルの構築
        self.popt: np.ndarray
        self.pcov: np.ndarray

        self.dataXForPredict: np.ndarray = list_exp
        self.dataTForPredict: np.ndarray = list_res

        list_p0: list[float] = [1] * 2

        self.popt, self.pcov = curve_fit(
            infiniteProductOfProblemSizeMultipliedByProcesses,
            list_exp,
            list_res,
            list_p0,
            maxfev=1000000,
        )
        return True

    def predict(self, inputDF) -> np.ndarray:
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame) : 構築されたモデルを用いて予測を行うためのDF

        Returns:
            numpy.ndarray: 構築されたモデルから予測された値

        Note:
            必ずinputDFにはプロセス数を意味する "process" を入れる。入れないと動作を保証できない。
        """

        # inputDFとモデルの構築に用いた説明変数のDFの列の順番が同じことを確認
        if inputDF.columns.to_list() != self.explanatoryVariableColumnNames:
            warnings.warn(
                f"inputDFとモデルの構築に用いた説明変数のDF列の順番が異なります。\ninputDF.columns.to_list()[{inputDF.columns.to_list()}] != self.explanatoryVariableColumnNames[{self.explanatoryVariableColumnNames}]"
            )
            return -1

        # inputDFに"process"列名がない場合は警告を出力
        if ("process" in inputDF.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return -1

        # inputDFの1列目の列名が"process"でない場合は、1列目を"process"とする
        if "process" == inputDF.columns.to_list()[0]:
            process_column = inputDF.pop("process")
            inputDF.insert(0, "process", process_column)

        # inputDFから引数list_inputとして使われる変数ndarray_inputDFを作成
        ndarray_inputDF: np.ndarray = inputDF.to_numpy().T

        predicted_result: np.ndarray = (
            infiniteProductOfProblemSizeMultipliedByProcesses(
                ndarray_inputDF, *self.popt
            )
        )
        return predicted_result

    def returnMAPE(self) -> float:
        """returnMAPE(self)

        モデルの構築に使用されたデータからMAPEを算出する

        Args:
            self : none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray[float] = self.rawResponseVariable.to_numpy().ravel()

        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray の長さ[{len(predicted_result)}] と実際の値の ndarray の長さ[{len(real_data)}] とが異なります。"
            )
            return -1

        mape: float = returnMapeScore(l1=predicted_result, l2=real_data)

        return mape


# In[51]:


class Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """問題サイズの値の総乗からプロセス数を割ったモデル
    Model_InfiniteProductOfProblemSizeDividedByProcesses_ForMultipleRegression(ModelBaseForMultipleRegression)

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def build_model(self) -> bool:
        """ "build_model(self) -> bool

        オブジェクトの初期化時に生成されたインスタンスの説明変数およびインスタンスの目的変数からモデルを構築する

        Args:
            self : none

        Returns: boolean。成功ならTrue、失敗ならFalse

        Note:
            必ず説明変数を格納したDFにはプロセス数を意味する "process" を入れること。入れないと動作を保証できない。
        """

        # 説明変数を格納したDFに列名 "process" がない場合は警告を出力し、失敗扱いにする
        if ("process" in self.explanatoryVariableColumnNames) == False:
            warnings.wanr("inputDFにprocess列がありません。")
            return False

        # 説明変数を格納したDFの1列目の列名が "process" でない場合は1列目を "process" とする
        if "process" == self.explanatoryVariableColumnNames[0]:
            process_column = self.rawExplanaoryVariable.pop("process")
            self.rawExplanaoryVariable.insert(0, "process", process_column)

        # list_exp:説明変数のリスト
        # list_expをlist_inputとして利用
        list_exp: np.ndarray = self.rawExplanaoryVariable.to_numpy().T
        # list_res:目的変数のリスト
        list_res: np.ndarray = self.rawResponseVariable.to_numpy()
        # list_resが複数の列を持っている(=DF的な解釈では列が複数ある)と予測ができるか不確かなため、警告を出す
        if list_res.shape[1] != 1:
            warnings.warn(f"目的変数の個数[{list_res.shape[1]}]が想定[1]と異なります")
            return False
        # list_resをlist[float]のような形式にする
        list_res = list_res.ravel()

        # モデルの構築
        self.popt: np.ndarray
        self.pcov: np.ndarray

        self.dataXForPredict: np.ndarray = list_exp
        self.dataXForPredict: np.ndarray = list_res

        list_p0: list[float] = [1] * 2

        self.popt, self.pcov = curve_fit(
            infiniteProductOfProblemSizeDividedByProcesses,
            list_exp,
            list_res,
            list_p0,
            maxfev=1000000,
        )

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF)

        inputDFのデータから構築されたモデルを使って予測を行う

        Args:
            self : none
            inputDF (pandas.DataFrame): 構築されたモデルを使って予測を行うDF

        Returns:
            numpy.ndarray: 構築されたモデルから予測された値。型に確証なし。

        Note:
            必ず説明変数を格納したDFにはプロセス数を意味する "process" を入れる。入れないと動作を保証できない。
        """

        # inputDF に "process" 列名がない場合は警告を出力
        if ("process" in inputDF.columns.to_list()) == False:
            warnings.warn("inputDFにprocess列がありません。")
            return -1

        # inputDF とモデルの構築に用いた説明変数の列の順番が同じことを確認
        if inputDF.columns.to_list() != self.explanatoryVariableColumnNames:
            warnings.warn(
                f"inputDFとモデルの構築に用いた説明変数のDF列の順番が異なります。\ninputDF.columns.to_list()[{inputDF.columns.to_list()}] != self.explanatoryVariableColumnNames[{self.explanatoryVariableColumnNames}]"
            )

        if "process" == inputDF.columns.to_list()[0]:
            process_column: pd.Series = inputDF.pop("process")
            inputDF.insert(0, "process", process_column)

        # inputDFから引数list_inputとして使われる変数ndarray_inputDFを作成
        ndarray_inputDF: np.ndarray = inputDF.to_numpy().T

        predicted_result: np.ndarray = infiniteProductOfProblemSizeDividedByProcesses(
            ndarray_inputDF, *self.popt
        )

        return predicted_result

    def returnMAPE(self) -> float:
        """returnMAPE(self)

        モデルの構築に使用されたデータからMAPEを算出する

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合、-1
        """

        predicted_result: list[float] = self.predict(self.rawExplanaoryVariable)
        real_data: np.ndarray = self.rawResponseVariable.to_numpy().ravel()

        if len(predicted_result) != len(real_data):
            warnings.warn(
                f"予測された値の ndarray の長さ[{len(predicted_result)}] と実際の値の ndarray の長さ[{len(real_data)}] とが異なります。"
            )
            return -1

        mape: float = returnMapeScore(l1=predicted_result, l2=real_data)

        return mape


# In[52]:


def returnAverageMAPEfromConditions(
    list_benchmarkNames: list[str],
    list_problemSizes: list[str],
    list_processes: list[int],
    list_modelNames: list[str],
) -> pd.DataFrame:
    """returnAverageMAPEfromConditions

    指定した条件から下記のような表を作成する関数

    | <ベンチマーク名> | <平均MAPE> |

    Args:
        list_benchmarkNames (list[str])
        list_problemSizes (list[str])
        list_processes (list[int])
        list_modelNames (list[str])

    Returns:
        pd.DataFrame : 上記の説明通りのデータフレーム
        int : なんらかの障害により実行できなくなったとき
    """

    if len(list_benchmarkNames) == 0:
        warnings.warn("ベンチマークプログラム名が指定されていません")
        return -1
    if len(list_problemSizes) == 0:
        warnings.warn("問題サイズが指定されていません")
        return -1
    if len(list_processes) == 0:
        warnings.warn("プロセス数が指定されていません")
        return -1
    if len(list_modelNames) == 0:
        warnings.warn("モデル名が指定されていません")
        return -1

    list_result_col_benchmarkNames: list[str] = []
    list_result_col_averageMAPE: list[float] = []
    dict_resultForDF: dict[list] = {}

    # ベンチマークプログラム名でループ
    for benchmarkName in list_benchmarkNames:
        # 生データを取得
        rawDF = return_rawDF_with_init_param(
            benchmark_name=benchmarkName,
            classes=list_problemSizes,
            processes=list_processes,
            csv_dir_path="./csv_files/",
        )
        # returnMAPEDFFromRawDF()を実行
        _MAPEtable = returnMAPEDFFromRawDF(
            input_rawDF=rawDF, model_names_list=list_modelNames
        )
        # returnAverageMAPEfromDF()を実行
        _averageMAPEfrom_MAPEtable = returnAverageMAPEfromDF(input_DF=_MAPEtable)
        # list_ret_col* に値を格納
        list_result_col_benchmarkNames.append(benchmarkName)
        list_result_col_averageMAPE.append(_averageMAPEfrom_MAPEtable)

    dict_resultForDF["benchmarkName"] = list_result_col_benchmarkNames
    dict_resultForDF["averageMAPE"] = list_result_col_averageMAPE

    return pd.DataFrame(data=dict_resultForDF)


# In[53]:


def returnAverageMAPEfromDF(input_DF: pd.DataFrame) -> float:
    """returnAverageMAPEfromDF

    MAPE表からMAPEの最小値の平均を返す

    Args:
        input_rawDF(pd.DataFrame): 入力データフレーム。カラム名には "functionName" が必須で、ほかのカラム名はモデル名になっている

    Returns:
        float : 渡されたデータフレームの平均MAPE
    """

    # 関数名の列をインデックスにする
    indexed_df: pd.DataFrame = input_DF.set_index("functionName")
    # 統計値のDFを作成
    describeDF: pd.DataFrame = indexed_df.T.describe()
    # 作成したDFの最低値の列から平均を算出
    mins: pd.Series = describeDF.T["min"]
    mean: float = float(mins.mean())
    # 算出された平均を返す
    return mean


def test_returnAverageMAPEfromDF():
    """test_returnAverageMAPEfromDF()


    returnAverageMAPEfromDF()のテスト
    """
    functionName: list[str] = ["func_a", "func_b", "func_c"]
    model_1: list[int] = [1, 2, 2]
    model_2: list[int] = [2, 1, 3]
    model_3: list[int] = [3, 3, 1]

    test_data: dict = {
        "functionName": functionName,
        "model_1": model_1,
        "model_2": model_2,
        "model_3": model_3,
    }

    test_inputDF = pd.DataFrame(data=test_data)
    ret = returnAverageMAPEfromDF(input_DF=test_inputDF)
    assert ret == 1

    functionName: list[str] = ["func_a", "func_b", "func_c"]
    model_1: list[int] = [2, 2, 2]
    model_2: list[int] = [2, 2, 2]
    model_3: list[int] = [2, 2, 2]

    test_data: dict = {
        "functionName": functionName,
        "model_1": model_1,
        "model_2": model_2,
        "model_3": model_3,
    }

    test_inputDF = pd.DataFrame(data=test_data)
    ret = returnAverageMAPEfromDF(input_DF=test_inputDF)
    assert ret == 2


def returnMAPEDFFromRawDF(
    input_rawDF: pd.DataFrame,
    model_names_list: list[str],
) -> pd.DataFrame:
    """returnMAPEDFFromRawDF

    入力DFからMAPE表を作成する関数

    Args:
        input_rawDF(pd.DataFrame): 入力データフレーム。カラム名には "functionName", "functionCallNum", "benchmarkName", "process", "intBenchmarkClass" などがある
        model_names_list(list[str]): モデル名を格納したリスト。要素の例として "modelLin", "modelLog", "modelIp" などがある

    Returns:
        pd.DataFrame : 下記のようなDataFrameが返される。

        <関数名>| <モデル名1>| ... <モデル名n>|
    """

    exp_var: list[str] = input_rawDF.columns.tolist()

    # ベンチマーク名が複数存在する場合は警告を出して -1 を返す
    if len(set(input_rawDF["benchmarkName"].to_list())) != 1:
        warnings.warn("複数のベンチマークが入力されたDFに格納されています")
        return -1

    for element_be_removed in [
        "functionName",
        "functionCallNum",
        "intBenchmarkClass",
        "benchmarkName",
        "benchmarkClass",
    ]:
        exp_var.remove(element_be_removed)
    res_var: list[str] = ["functionCallNum"]

    returnDF: pd.DataFrame = return_MAPE_Table_DF_from_rawDF(
        rawDF=input_rawDF,
        exp_var_list=exp_var,
        res_var_list=res_var,
        model_name_list=model_names_list,
    )

    return returnDF


# In[54]:


class Model_LinearSumOf2elementCombination_ForMultipleRegression(
    ModelBaseForMultipleRegression
):

    """説明変数2つの組み合わせの総和モデル
    Model_LinearSumOf2elementCombination_ForMultipleRegression(ModelBaseForMultipleRegression)
    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト

    Note:
        不要な列が含まれている場合は適切に「モデルの構築」・「モデルを利用した予測」などが行えない。
    """

    def build_model(self) -> bool:
        """build_model(self) -> bool
        オブジェクトの初期化時に生成された、インスタンスの説明変数およびインスタンスの目的変数からモデルを構築する

        Args:
            self : none

        Returns: boolean。成功ならTrue。失敗ならFalse。
        """

        df_mid_var: pd.DataFrame = self.return_df_for_2comibnations(
            self.rawExplanaoryVariable
        )

        self.lr = LinearRegression()

        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF: pd.DataFrame) -> np.ndarray

        Args:
            self : none
            inputDF (pd.DataFrame) : 構築されたモデルを使って予測を行うDF

        Returns:
            np.ndarray
        """

        df_mid_var: pd.DataFrame = self.return_df_for_2comibnations(inputDF)
        resultDF = self.lr.predict(df_mid_var)

        return resultDF

    def returnMAPE(self) -> float:
        """returnMAPE(self) -> float

        モデルに構築されたデータからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合、-1
        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)

        mape: float = returnMapeScore(return_expect, return_actually)

        return mape

    def returnWeightedMAPE(self) -> float:
        """returnWeightedMAPE(self) -> float

        モデルの構築に使用されたデータからweightedMAPEを算出する

        Args:
            self: none

        Returns:
            float: weightedMAPE
            int: 失敗した場合、-1
        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)
        weightedMape: float = returnWeightedMapeScore(
            real=return_expect, predicted=return_actually
        )
        return weightedMape

    def return_df_for_2comibnations(
        self,
        inputDF: pd.DataFrame,
    ) -> pd.DataFrame:
        """return_df_for_2comibnations()

        入力DFから説明変数の組み合わせを算出し、その組み合わせの要素同士を乗算した列で構成されたDFを返す関数

        """

        # 学習用PDを作成する
        # 0. 学習用DFを入力DFからコピーする
        # 1. 列名二つずつの組み合わせを作成
        # 2. 列名の組み合わせで計算し、それを学習用DFに入れる

        returnDF: pd.DataFrame = inputDF.copy(deep=True)
        returnDF_columns = returnDF.columns.tolist()
        returnDF = returnDF.drop(returnDF_columns, axis=1)

        list_combinations: list[set[str]] = list(
            itertools.combinations(self.explanatoryVariableColumnNames, 2)
        )
        for combination_index in range(len(list_combinations)):
            combination: set[str, str] = list_combinations[combination_index]
            exp_name0: str = combination[0]
            exp_name1: str = combination[1]
            returnDF[str(combination_index)] = inputDF[exp_name0] * inputDF[exp_name1]

        return returnDF


# In[55]:


def test_Model_LinearSumOf2elementCombination_ForMultipleRegression():
    """test_Model_LinearSumOf2elementCombination_ForMultipleRegression()
    Model_LinearSumOf2elementCombination_ForMultipleRegressionのテスト
    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 10 * np.linspace(10, 20, 11)
    plotX_3: np.ndarray = 100 * np.linspace(10, 20, 11)
    # 目的変数
    a: int = 100
    b: int = 90
    c: int = 80
    k: int = -500
    plotT: np.ndarray = (
        (a * plotX_1 * plotX_2) + (b * plotX_1 * plotX_3) + (c * plotX_2 * plotX_3) + k
    )

    # DFを作成する
    # カラム名のリスト
    columnNames: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
        "plotT",
    ]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数・説明変数のカラム名のリスト
    # 目的変数のカラム名のリスト
    columnNamesForExp: list[str] = [
        "process",
        "plotX_2",
        "plotX_3",
    ]
    # 説明変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測をする
    # モデルオブジェクトの作成
    objectModel = Model_LinearSumOf2elementCombination_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )
    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータとのMAPEによって実装がうまくいっているかどうかの判定を行う
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[56]:


def test_returnDFaboutDifferenceBetweenInput2DFs():
    """test_returnDFaboutDifferenceBetweenInput2DFs()

    returnDFaboutDifferenceBetweenInput2DFs() のテスト
    """

    functionName: list[str] = ["A()", "B()", "C()", "D()", "E()"]

    mape_model_1: list[float] = [4, 6, 6, 6, 8]
    mape_model_2: list[float] = [5, 5, 7, 5, 9]
    mape_model_3: list[float] = [6, 7, 5, 7, 4]
    mape_model_4: list[float] = [2, 3, 8, 8, 1]

    inputDF_1: pd.DataFrame = pd.DataFrame(
        {
            "functionName": functionName,
            "model_1": mape_model_1,
            "model_2": mape_model_2,
            "model_3": mape_model_3,
        }
    )
    inputDF_2: pd.DataFrame = pd.DataFrame(
        {
            "functionName": functionName,
            "model_1": mape_model_1,
            "model_2": mape_model_2,
            "model_3": mape_model_3,
            "model_4": mape_model_4,
        }
    )

    functionName_expected: list[str] = ["A()", "B()", "E()"]
    mape_model_1_expected: list[float] = [4, 6, 8]
    mape_model_2_expected: list[float] = [5, 5, 9]
    mape_model_3_expected: list[float] = [6, 7, 4]
    mape_model_4_expected: list[float] = [2, 3, 1]
    expectedDF: pd.DataFrame = pd.DataFrame(
        {
            "functionName": functionName_expected,
            "model_1": mape_model_1_expected,
            "model_2": mape_model_2_expected,
            "model_3": mape_model_3_expected,
            "model_4": mape_model_4_expected,
        }
    )

    inputDF_1 = inputDF_1.set_index("functionName")
    inputDF_2 = inputDF_2.set_index("functionName")
    expectedDF = expectedDF.set_index("functionName")

    model_name_list_1: list[str] = ["model_1", "model_2", "model_3"]
    model_name_list_2: list[str] = ["model_1", "model_2", "model_3", "model_4"]

    inputDF_1 = addLowestMAPEsModelNameColumn(
        inputDF=inputDF_1, model_name_list=model_name_list_1, version=2
    )
    inputDF_2 = addLowestMAPEsModelNameColumn(
        inputDF=inputDF_2, model_name_list=model_name_list_2, version=2
    )

    actuallyDF: pd.DataFrame = returnDFaboutDifferenceBetweenInput2DFs(
        inputDF_1=inputDF_1, inputDF_2=inputDF_2
    )

    assert actuallyDF.equals(expectedDF)


def returnDFaboutDifferenceBetweenInput2DFs(
    inputDF_1: pd.DataFrame, inputDF_2: pd.DataFrame
) -> pd.DataFrame:
    """returnDFaboutDifferenceBetweenInput2DFs(inputDF_1 :pd.DataFrame, inputDF_2 :pd.DataFrame)

    引数で渡された2つのDataFrameからDataFrameを返す。具体例は下記の通り。

    * 入力DF1

    | 関数名(index) | ... | 最適モデル  |
    |-----|-----|--------|
    | A() | ... | model1 |
    | B() | ... | model1 |
    | C() | ... | model1 |

    * 入力DF1

    | 関数名(index) | ... | 最適モデル  |
    |-----|-----|--------|
    | A() | ... | model1 |
    | B() | ... | model2 |
    | C() | ... | model1 |

    * 返値DF

    | 関数名(index) | model1      | model2      |
    |-----|-------------|-------------|
    | B() | mape@model1 | mape@model2 |

    Args:
        inputDF_1 (pd.DataFrame): 入力DF1
        inputDF_2 (pd.DataFrame): 入力DF2

    Returns:
        pd.DataFrame: 渡された入力DFどうしの違いがわかるDF

    """

    inputDF_1_tmp = inputDF_1.copy()
    inputDF_2_tmp = inputDF_2.copy()

    if len(inputDF_1_tmp) > len(inputDF_2_tmp):
        warnings.warn("inputDF_1のカラム数がinputDF_2のカラム数より多いです")
        return False

    set_inputDF_1_column: set[str] = set(inputDF_1_tmp.columns.tolist())
    set_inputDF_2_column: set[str] = set(inputDF_2_tmp.columns.tolist())

    if set_inputDF_1_column > set_inputDF_2_column:
        warnings.warn("inputDF_2のカラムはinputDF_1のカラムを完全に包含していません")
        return False

    inputDFs_diff = inputDF_1["最適モデル"] != inputDF_2["最適モデル"]

    inputDF_1_diff: pd.DataFrame = inputDF_1_tmp[inputDFs_diff]
    inputDF_2_diff: pd.DataFrame = inputDF_2_tmp[inputDFs_diff]

    set_best_model_names: set[str] = set(inputDF_1_diff["最適モデル"]) | set(
        inputDF_2_diff["最適モデル"]
    )
    set_all_model_names: set[str] = set_inputDF_2_column
    set_unBest_model_names: set[str] = set_all_model_names - set_best_model_names

    retDF: pd.DataFrame = inputDF_2_tmp.drop(columns=set_unBest_model_names)
    retDF = retDF[inputDFs_diff]

    return retDF


# In[57]:


def test_Model_LinearSumOfElementCombinations_ForMultipleRegression():
    """test_Model_LinearSumOfElementCombinations_ForMultipleRegression()

    Model_LinearSumOfElementCombinations_ForMultipleRegression() のテスト
    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 13 * np.linspace(10, 20, 11)
    plotX_3: np.ndarray = 17 * np.linspace(10, 20, 11)

    # 係数・切片
    a: int = 17
    b: int = 19
    c: int = 23
    d: int = 29
    e: int = 17
    f: int = 19
    g: int = 23
    h: int = 29
    i: int = 17
    j: int = 19
    k: int = 23
    l: int = 29
    l: int = 17

    # 目的変数

    plotT: np.ndarray = (
        (a * plotX_1 * plotX_2)
        + (b * 1 / plotX_1 * plotX_2)
        + (c * plotX_1)
        + (d * plotX_2)
        + (e * 1 / plotX_1)
        + f
    )

    # DFを作成する
    columnNames: list[str] = ["process", "plotX_2", "plotT"]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plotX_2"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測の実施
    objectModel = Model_LinearSumOfElementCombinations_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )

    # モデルの構築時に使用されるメソッドのテスト

    col0: np.ndarray = 1 / plotX_1
    col2: np.ndarray = plotX_2
    col1: np.ndarray = 1 / plotX_1 * plotX_2
    col4: np.ndarray = plotX_1
    col3: np.ndarray = plotX_1 * plotX_2

    inputDictForExpectedDF: dict[str, np.ndarray] = {
        "col0": col0,
        "col1": col1,
        "col2": col2,
        "col3": col3,
        "col4": col4,
    }
    expectedDF: pd.DataFrame = pd.DataFrame(data=inputDictForExpectedDF)
    actuallyDF: pd.DataFrame = objectModel.return_df_for_combinations(inputDFForTest)
    assert actuallyDF.equals(
        expectedDF
    ), f"actuallyDF = \n{actuallyDF}\nexpectedDF = \n{expectedDF}"

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装ができているかを判定する
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"

    plotT: np.ndarray = (
        (a * plotX_1 * plotX_2 * plotX_3)
        + (b * 1 / plotX_1 * plotX_2 * plotX_3)
        + (c * plotX_1 * plotX_2)
        + (d * plotX_1 * plotX_3)
        + (e * 1 / plotX_1 * plotX_2)
        + (f * 1 / plotX_1 * plotX_3)
        + (g * plotX_2 * plotX_3)
        + (h * plotX_1)
        + (i * plotX_2)
        + (j * plotX_3)
        + (k * 1 / plotX_1)
        + l
    )

    # DFを作成する
    columnNames = ["process", "plotX_2", "plotX_3", "plotT"]
    datumForDF = [plotX_1, plotX_2, plotX_3, plotT]
    inputDFForTest = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 目的変数のカラム名のリスト
    columnNamesForExp = ["process", "plotX_2", "plotX_3"]
    # 説明変数のカラム名のリスト
    columnNamesForRes = ["plotT"]

    # 予測の実施
    objectModel = Model_LinearSumOfElementCombinations_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装できているかを判定する
    mape = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


class Model_LinearSumOfElementCombinations_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """説明変数の組み合わせの線形和モデル

    Model_LinearSumOfElementCombinations_ForMultipleRegression(ModelBaseForMultipleRegression)

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数及びインスタンスの目的変数からモデルを構築する

        Args:
            self :none

        Returns: boolean。成功ならTrue,失敗ならFalse
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(
            self.rawExplanaoryVariable
        )

        self.lr = LinearRegression()

        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF :pd.DataFrame) -> np.ndarray

        Args:
            self :none
            inputDF (pd.DataFrame) : 構築されたモデルを使って予測を行うDF

        Returns:
            np.ndarray
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(inputDF)
        resultDF: pd.DataFrame = self.lr.predict(df_mid_var)

        return resultDF

    def returnMAPE(self) -> float:
        """returnMAPE(self) -> float

        モデルに構築されたデータからMAPEを算出する。

        Args:
            self :none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合、-1
        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)

        mape: float = returnMapeScore(return_expect, return_actually)

        return mape

    def returnWeightedMAPE(self) -> float:
        """returnWeightedMAPE(self) -> float

        モデルの構築に使用されたデータからweightedMAPEを算出する

        Args:
            self: none

        Returns:
            float: weightedMAPE
            int: 失敗した場合、-1
        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)
        weightedMape: float = returnWeightedMapeScore(
            real=return_expect, predicted=return_actually
        )
        return weightedMape

    def return_df_for_combinations(self, inputDF: pd.DataFrame) -> pd.DataFrame:
        """return_df_for_combinations()

        入力DFから説明変数の組み合わせの全通りを算出し、その組み合わせの要素同士を乗算した列で構成されたDFを返す関数

        Args:
            self :none
            inputDF (pd.DataFrame) : 説明変数の列を含んだDF

        Returns:
            pd.DataFrame: 説明通りのDF
            int: 失敗した場合、-1
        """

        # 入力DFが説明変数・説明変数のすべてを内包しているか
        set_input_column_names: set[str] = set(inputDF.columns.tolist())
        set_exp_column_names: set[str] = set(self.explanatoryVariableColumnNames)
        if set_input_column_names < set_exp_column_names:
            return -1
        # 説明変数のセットおよび入力DFのカラム名として "process" があるか
        if "process" not in set_exp_column_names:
            warnings.warn("説明変数の集合にプロセス数を表す process がありません")
            return -1
        if "process" not in set_input_column_names:
            warnings.warn("入力DFのカラム名にプロセス数を表す process がありません")
            return -1

        # 「プロセス数の逆数」に関連する処理
        set_another_exp_column_names: set[str] = set(
            self.explanatoryVariableColumnNames
        )
        set_another_exp_column_names.remove("process")
        set_another_exp_column_names.add("inverse_process")
        _inputDF: pd.DataFrame = inputDF.copy()
        _inputDF["inverse_process"] = 1 / inputDF["process"]

        """
        list_combinations1 list:
            モデル構築用リスト1  
        list_combinations2 list:
            モデル構築用リスト2
        set_combinations_for_model set:
            モデル構築用集合。モデル構築用集合1とモデル構築用集合2の和集合。
        list_combinations_for_model list:
            set_combinations_for_model をリスト化したリスト
        DF_for_model pd.DataFrame:
            モデル構築用DF
        """

        returnDF: pd.DataFrame = pd.DataFrame(index=np.arange(len(_inputDF)))

        # モデル構築用集合1の作成
        list_combinations1: list[set[str]] = self.return_list_combinations(
            inputSet=set_exp_column_names
        )
        # モデル構築用集合2の作成
        list_combinations2: list[set[str]] = self.return_list_combinations(
            inputSet=set_another_exp_column_names
        )
        # モデル構築用集合の作成
        set_combinations_for_model: set[set[str]] = set(list_combinations1) | set(
            list_combinations2
        )
        list_combinations_for_model: list[list[str]] = list(set_combinations_for_model)
        for i in range(len(list_combinations_for_model)):
            list_combinations_for_model[i] = sorted(
                list(list_combinations_for_model[i])
            )
        list_combinations_for_model = sorted(list_combinations_for_model)
        # モデル構築用DF1とモデル構築用DF2からモデル構築用DFの作成
        for combination_index in range(len(list_combinations_for_model)):
            combination: set[str] = list_combinations_for_model[combination_index]
            element: np.ndarray = np.ones(len(_inputDF))
            for each_combination_index in range(len(combination)):
                element *= _inputDF[combination[each_combination_index]].values
            returnDF[f"col{combination_index}"] = element

        return returnDF

    def return_list_combinations(self, inputSet: set[str]) -> list[set[str]]:
        """return_list_combinations(self, inputSet: set[str]) -> list[set[str]]

        入力setから入力setの組み合わせの全通りを算出し、リストに格納した結果を返す関数

        Args:
            self :none
            inputSet (set[str]) : 入力set。説明変数の値名が想定されている。

        Returns:
            pd.DataFrame: 説明通りのSet
            int: 失敗した場合、-1
        """

        return_list: list[set[str]] = []
        for i in range(len(inputSet) + 1):
            if i != 0:
                for set_combination in itertools.combinations(inputSet, i):
                    return_list.append(set_combination)

        return sorted(return_list)


# In[58]:


def returnWeightedMapeScore(real: list[float], predicted: list[float]) -> float:
    """returnWeightedMapeScore(real :list[float], predicted :list[float]) -> float

    重み付き平均絶対パーセント誤差 (weightedMAPE)(weighted Mean Absolute Percent Error)を返す関数

    Args:
        引数として長さの同じ二つのリストをとる
        real :list[float] 実測値のリスト
        predicted :list[float] 予測値のリスト

    Returns:
      float : 引数から計算されたweightedMAPEの値（単位％）
    """

    # 分母
    returnNumNumerator: float = 0
    # 分子
    returnNumDenominator: float = 0

    if len(real) != len(predicted):
        warnings.warn("引数のリストの長さが異なります")
        return -1

    for i in range(len(real)):
        real_num = real[i]
        predicted_num = predicted[i]

        returnNumNumerator += abs(real_num - predicted_num)
        returnNumDenominator += real_num

    return returnNumNumerator / returnNumDenominator


def test_returnWeightedMapeScore():
    test_real: list[float] = [1, 2, 3, 4, 5]
    test_predicted: list[float] = [1, 2, 3, 4, 5]
    expected: float = 0
    actually: float = returnWeightedMapeScore(real=test_real, predicted=test_predicted)
    assert expected == actually, f"expected = {expected}, actually = {actually}"

    test_real: list[float] = [10, 20, 30, 40, 50]
    test_predicted: list[float] = [1, 2, 3, 4, 5]
    expected: float = (9 + 18 + 27 + 36 + 45) / (10 + 20 + 30 + 40 + 50)
    actually: float = returnWeightedMapeScore(real=test_real, predicted=test_predicted)
    assert expected == actually, f"expected = {expected}, actually = {actually}"

    test_real: list[float] = [1, 2, 3, 4, 5]
    test_predicted: list[float] = [11, 22, 33, 44, 55]
    expected: float = (10 + 20 + 30 + 40 + 50) / (1 + 2 + 3 + 4 + 5)
    actually: float = returnWeightedMapeScore(real=test_real, predicted=test_predicted)
    assert expected == actually, f"expected = {expected}, actually = {actually}"


# In[59]:


def return_rawDFinLULESH(
    processes: list[int], iterations: list[int], sizes: list[int], csvDirPath: str
) -> pd.DataFrame:
    """return_rawDFinLULESH(processes :list[int], iterations :list[int], sizes :list[int]) -> pd.DataFrame

    ベンチマークプログラムLULESHの生データを取得するプログラム

    Args:
        processes (list[int]): プロセス数のリスト。要素数は1以上。
        iterations (list[int]): イテレーション数のリスト。要素数は1以上。
        sizes (list[int]): 問題サイズのリスト。要素数は1以上。
        csvDirPath (str): CSVファイルのディレクトリ

    Returns:
        pd.DataFrame: 下記の列構成のデータフレーム

    | 関数名 | 関数コール回数 | イテレーションサイズ | 問題サイズ |
    |-----|---------|------------|-------|
    |     |         |            |       |

    """

    whole_processes: list[int] = [8, 27, 64, 125, 216, 343, 512]
    whole_iterations: list[int] = [8, 16, 32, 64, 128, 256]
    whole_sizes: list[int] = [16, 24, 32, 48, 64, 128]

    # プロセス数・問題サイズ・イテレーション数を集合化してwhole_*と同等かそれ未満であるのを確認
    if set(processes).issubset(set(whole_processes)) == False:
        warnings.warn("プロセス数の指定に誤りがあります")
        return False
    if set(iterations).issubset(set(whole_iterations)) == False:
        warnings.warn("イテレーション数の指定に誤りがあります")
        return False
    if set(sizes).issubset(set(whole_sizes)) == False:
        warnings.warn("問題サイズの指定に誤りがあります")
        return False

    datumCSV: list[pd.DataFrame] = []

    for process in processes:
        for iteration in iterations:
            for size in sizes:
                # candidateFileNames[f"pprof_lulesh_p{process}i{iteration}s{size}"] = {}
                fileName: str = f"pprof_lulesh_p{process}i{iteration}s{size}.csv"
                filePath: pathlib.Path = os.path.join(csvDirPath, fileName)
                if os.path.exists(filePath) and os.stat(filePath).st_size != 0:
                    data_pd: pd.DataFrame = pd.read_csv(filePath)
                    data_pd["process"] = process
                    data_pd["iteration"] = iteration
                    data_pd["size"] = size
                    datumCSV.append(data_pd)

    returnDF = pd.concat(datumCSV, axis=0)
    return returnDF


# In[60]:


def returnMAPEtableInLULESH(
    rawDF: pd.DataFrame,
    list_expVar: list[str],
    list_resVar: list[str],
    list_modelName: list[str],
) -> pd.DataFrame:
    """returnMAPEtableInLULESH

    LULESHでのMAPE表を返す関数

    Args:
        rawDF (pd.DataFrame) : "functionName" が列名の一つとして入っていることが条件
        list_expVar (list[str]) : 説明変数のリスト
        list_modelName (list[str]) : 目的変数のリスト

    Returns:
        pd.DataFrame
        int : なんらかの不備・障害により正常な返り値を返すことができなくなったとき
    """

    if len(list_expVar) == 0 or (
        set(list_expVar).issubset(set(rawDF.columns.tolist())) == False
    ):
        warnings.warn("説明変数の指定に誤りがあります")
        return -1
    if len(list_resVar) == 0 or (
        set(list_resVar).issubset(set(rawDF.columns.tolist())) == False
    ):
        warnings.warn("目的変数の指定に誤りがあります")
        return -1

    function_names: list[str] = list(set(rawDF["functionName"].tolist()))
    if len(function_names) == 0:
        warnings.warn("与えられたDFに関数名がありません")
        return -1

    result_series_list: list[pd.Series] = []

    for function_name in function_names:
        # 関数ごとの生データ
        rawDF_per_function = rawDF[rawDF["functionName"] == function_name]
        # モデルの構築
        models = Models(
            inputDF=rawDF_per_function,
            expVarColNames=list_expVar,
            resVarColNames=list_resVar,
            modelNames=list_modelName,
        )

        models.setUpDataBeforeCalcLr()
        models.calcLr()
        models.calcMAPE()

        dictCalcedMAPE = models.returnCalculatedMAPE()

        # 算出されたMAPEの数値をfloatにする
        for key in dictCalcedMAPE.keys():
            dictCalcedMAPE[key] = float(dictCalcedMAPE[key])

        # 関数ごとの結果に格納
        dict_for_series: dict = copy.deepcopy(dictCalcedMAPE)
        dict_for_series["functionName"] = function_name

        series = pd.Series(dict_for_series)
        result_series_list.append(series)

    resultDF: pd.DataFrame = pd.DataFrame(result_series_list)
    return resultDF


# In[61]:


def calcWeightedMAPEscore(
    inputDF: pd.DataFrame,
    inputColumnDict: dict,
) -> float:
    """calcWeightedMAPEscore()

    重み付き平均MAPEを算出する関数

    Args:
        inputDF (pd.DataFrame) : 下記のようなテーブル構成
        |<関数名>|<コール回数>|<MAPE>|
        inputColumnDict (dict) : 上記のテーブル構成を前提に、次の辞書を構成する。{"funcName":<関数名>,"call":<コール回数>,"MAPE":<MAPE>,}

    Returns:
        float: 重み付き平均MAPE

    """
    _col_funcName: str = inputColumnDict["funcName"]
    _col_call: str = inputColumnDict["call"]
    _col_MAPE: str = inputColumnDict["MAPE"]

    _ser_funcName: pd.Series = inputDF[_col_funcName]
    _ser_call: pd.Series = inputDF[_col_call]
    _ser_MAPE: pd.Series = inputDF[_col_MAPE]

    _ret_numerator: float = 0
    _ret_denominator: float = sum(_ser_call)

    for index in range(len(_ser_funcName)):
        _ret_numerator += _ser_call[index] * _ser_MAPE[index]

    return _ret_numerator / _ret_denominator


def test_calcWeightedMAPEscore():
    関数名: list[str] = ["name0", "name1", "name2", "name3"]
    inputColumnDict = {"funcName": "関数名", "call": "コール回数", "MAPE": "MAPE"}
    # テストケース1
    コール回数: list[float] = [1, 2, 3, 4]
    MAPE: list[float] = [1, 2, 3, 4]
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"関数名": 関数名, "コール回数": コール回数, "MAPE": MAPE}
    )
    expected_result = 3.0
    actually_result = calcWeightedMAPEscore(
        inputDF=inputDF, inputColumnDict=inputColumnDict
    )
    assert (
        expected_result == actually_result
    ), f"expected_result={expected_result},actually_result={actually_result}"
    # テストケース2
    コール回数: list[float] = [5, 4, 3, 2]
    MAPE: list[float] = [7, 7, 7, 7]
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"関数名": 関数名, "コール回数": コール回数, "MAPE": MAPE}
    )
    expected_result = 7.0
    actually_result = calcWeightedMAPEscore(
        inputDF=inputDF, inputColumnDict=inputColumnDict
    )
    assert (
        expected_result == actually_result
    ), f"expected_result={expected_result},actually_result={actually_result}"
    # テストケース3
    コール回数: list[float] = [1, 2, 3, 4]
    MAPE: list[float] = [5, 4, 3, 2]
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"関数名": 関数名, "コール回数": コール回数, "MAPE": MAPE}
    )
    expected_result = 3.0
    actually_result = calcWeightedMAPEscore(
        inputDF=inputDF, inputColumnDict=inputColumnDict
    )
    assert (
        expected_result == actually_result
    ), f"expected_result={expected_result},actually_result={actually_result}"


# In[62]:


def return_bestModelObject(
    inputDF: pd.DataFrame,
    list_expVar: list[str],
    list_resVar: list[str],
    list_modelName: list[str],
):
    """return_bestModelObject()

    入力に対して最適なモデルを返す関数

    Args:
        inputDF (pd.DataFrame) : 下記のようなテーブル構成
        |<説明変数1>|<説明変数2>|...|<目的変数>|
        list_expVar (list[str]) : 説明変数のリスト
        list_resVar (list[str]) : 目的変数のリスト
        list_modelName (list[str]) : モデル名のリスト

    Returns:
        dict : {"object":各モデルのオブジェクト, "modelName":モデル名}

    """

    models = Models(
        inputDF=inputDF,
        expVarColNames=list_expVar,
        resVarColNames=list_resVar,
        modelNames=list_modelName,
    )

    models.setUpDataBeforeCalcLr()
    models.calcLr()
    models.calcMAPE()

    retObject = None
    retModelName: str = None

    dict_MAPE: dict[float] = models.returnCalculatedMAPE()

    retModelName = min(dict_MAPE, key=dict_MAPE.get)
    retObject = models.returnObject(modelName=retModelName)

    return {"object": retObject, "modelName": retModelName}


def test_return_bestModelObject():
    exp_1: np.ndarray = np.linspace(1, 10, 10)
    exp_2: np.ndarray = np.linspace(10, 1, 10)
    exp_3: np.ndarray = np.linspace(20, 10, 10)

    coefficient_1: int = 7
    coefficient_2: int = 5
    coefficient_3: int = -3

    list_modelName: list[str] = ["modelLin", "modelIp", "modelLog"]
    list_expVar: list[str] = ["process", "exp_2", "exp_3"]
    list_resVar: list[str] = ["res_"]

    # 線形モデル
    res_: np.ndarray = (
        coefficient_1 * exp_1 + coefficient_2 * exp_2 + coefficient_3 * exp_3
    )
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"process": exp_1, "exp_2": exp_2, "exp_3": exp_3, "res_": res_}
    )
    inputDF["functionName"] = "functionName"
    retDict = return_bestModelObject(
        inputDF=inputDF,
        list_expVar=list_expVar,
        list_resVar=list_resVar,
        list_modelName=list_modelName,
    )

    expected: str = "modelLin"
    actually: str = retDict["modelName"]
    assert actually == expected, f"expected={expected}, actually={actually}"
    assert retDict["object"] != None

    # 反比例モデル
    res_: np.ndarray = (
        coefficient_1 / exp_1 + coefficient_2 / exp_2 + coefficient_3 / exp_3
    )
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"process": exp_1, "exp_2": exp_2, "exp_3": exp_3, "res_": res_}
    )
    inputDF["functionName"] = "functionName"
    retDict = return_bestModelObject(
        inputDF=inputDF,
        list_expVar=list_expVar,
        list_resVar=list_resVar,
        list_modelName=list_modelName,
    )

    expected: str = "modelIp"
    actually: str = retDict["modelName"]
    assert actually == expected, f"expected={expected}, actually={actually}"
    assert retDict["object"] != None

    # 対数モデル
    res_: np.ndarray = (
        coefficient_1 * np.log10(exp_1)
        + coefficient_2 * np.log10(exp_2)
        + coefficient_3 * np.log10(exp_3)
    )
    inputDF: pd.DataFrame = pd.DataFrame.from_dict(
        {"process": exp_1, "exp_2": exp_2, "exp_3": exp_3, "res_": res_}
    )
    inputDF["functionName"] = "functionName"
    retDict = return_bestModelObject(
        inputDF=inputDF,
        list_expVar=list_expVar,
        list_resVar=list_resVar,
        list_modelName=list_modelName,
    )

    expected: str = "modelLog"
    actually: str = retDict["modelName"]
    assert actually == expected, f"expected={expected}, actually={actually}"
    assert retDict["object"] != None


# In[63]:


def returnWeightedMAPEScoreFromDF(
    inputDFtrain: pd.DataFrame,
    inputDFtest: pd.DataFrame,
    list_expVar: list[str],
    list_resVar: list[str],
    list_modelName,
):
    return -1


def returnWeightedMapeScoreFromCondition(
    benchmarkName: str,
    trainCondition: dict,
    testCondition: dict,
    csvDirPath: str,
    list_modelName: list[str],
):
    if benchmarkName == "lulesh":
        # lulesh の生データ取得処理
        rawDF_train: pd.DataFrmae = return_rawDFinLULESH(
            processes=trainCondition["processes"],
            iterations=trainCondition["iterations"],
            sizes=trainCondition["sizes"],
            csvDirPath=csvDirPath,
        )
        rawDF_test: pd.DataFrmae = return_rawDFinLULESH(
            processes=testCondition["processes"],
            iterations=testCondition["iterations"],
            sizes=testCondition["sizes"],
            csvDirPath=csvDirPath,
        )
        rawDF_train = rawDF_train.rename(columns={"Name": "functionName"})
        rawDF_test = rawDF_test.rename(columns={"Name": "functionName"})

        # 説明変数及び目的変数の処理
        list_expVar: list[str] = ["process", "iteration", "size"]
        list_resVar: list[str] = ["#Call"]

        functionNames: list[str] = list(set(rawDF_train["functionName"]))

    elif benchmarkName in ["cg", "ep", "ft", "is", "lu", "mg"]:
        # NPB の生データ取得処理
        rawDF_train: pd.DataFrame = return_rawDF_with_init_param(
            benchmark_name=benchmarkName,
            classes=trainCondition["sizes"],
            processes=trainCondition["processes"],
        )
        rawDF_test: pd.DataFrame = return_rawDF_with_init_param(
            benchmark_name=benchmarkName,
            classes=testCondition["sizes"],
            processes=testCondition["processes"],
        )

        # 説明変数及び目的変数の処理
        list_expVar: list[str] = rawDF_train.columns.tolist()
        for element_be_removed in [
            "functionName",
            "functionCallNum",
            "intBenchmarkClass",
            "benchmarkName",
            "benchmarkClass",
        ]:
            list_expVar.remove(element_be_removed)
        list_resVar: list[str] = ["functionCallNum"]

        functionNames: list[str] = list(set(rawDF_train["functionName"]))

    else:
        warnings.warn(f"{benchmarkName}は想定外のベンチマークプログラム名です")
        return -1

    list_series: list[pd.Series] = []

    # 関数ごとのDFを作成
    for functionName in functionNames:
        rawDF_per_function_train: pd.DataFrame = rawDF_train[
            rawDF_train["functionName"] == functionName
        ]
        rawDF_per_function_test: pd.DataFrame = rawDF_test[
            rawDF_test["functionName"] == functionName
        ]

        bestModelDict: dict = return_bestModelObject(
            inputDF=rawDF_per_function_train,
            list_expVar=list_expVar,
            list_resVar=list_resVar,
            list_modelName=list_modelName,
        )
        bestModel = bestModelDict["object"]
        predicted = float(
            np.array(bestModel.predict(inputDF=rawDF_per_function_test[list_expVar]))
        )
        _call: float = float(rawDF_per_function_test.iloc[0][list_resVar[0]])
        _MAPE: float = float(returnMapeScore(l1=[_call], l2=[predicted]))
        _series: pd.Series = pd.Series(
            {
                "functionName": functionName,
                "call": _call,
                "MAPE": _MAPE,
                "predicted_call": predicted,
            }
        )
        list_series.append(_series)

    DF_toCalcWeightedMAPE: pd.DataFrame = pd.concat(list_series, axis=1).T

    retNum: float = calcWeightedMAPEscore(
        inputDF=DF_toCalcWeightedMAPE,
        inputColumnDict={"funcName": "functionName", "call": "call", "MAPE": "MAPE"},
    )

    return retNum


# In[64]:


class Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """説明変数の組み合わせたうえで片方をn乗にするモデル

    Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def __init__(
        self,
        inputDF,
        explanatoryVariableColumnNames: list[str],
        exponent: int,
        responseVariableColumnNames: list[str],
        conditionDictForTest: Dict[str, str] = {},
        targetDF: pd.DataFrame = None,
    ):
        super().__init__(
            inputDF=inputDF,
            explanatoryVariableColumnNames=explanatoryVariableColumnNames,
            responseVariableColumnNames=responseVariableColumnNames,
            conditionDictForTest=conditionDictForTest,
            targetDF=targetDF,
        )
        self.exponent: int = exponent

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数及びインスタンスの目的変数からモデルを構築する

        Args:
            self :none

        Returns: boolean。成功ならTrue,失敗ならFalse
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(
            inputDF=self.rawExplanaoryVariable, exponent=self.exponent
        )

        self.lr = LinearRegression()
        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF: pd.DataFrame) -> np.ndarray

        Args:
            self: none
            inputDF (pd.DataFrame) :構築されたモデルを使って予測を行うDF

        Returns:
            np.ndarray
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(
            inputDF, exponent=self.exponent
        )
        resultDF = self.lr.predict(df_mid_var)

        return resultDF

    def returnMAPE(self) -> float:
        """returnMAPE() -> float

        モデルに構築に使用したデータからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合,-1

        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)

        return returnMapeScore(return_expect, return_actually)

    def return_df_for_combinations(
        self, inputDF: pd.DataFrame, exponent: int
    ) -> pd.DataFrame:
        """return_df_for_combinations(self, inputDF: pd.DataFrame) -> pd.DataFrame

        入力DFから説明変数の組み合わせを算出し、その組み合わせの要素同士を累乗計算をしつつかけ合わせた列で構成されたDFを返す関数

        Args:
            inputDF (pd.DataFrame):入力DF
            exponent (int): 累乗計算時に用いる指数
        """

        returnDF: pd.DataFrame = inputDF.copy(deep=True)
        returnDF_columns = returnDF.columns.tolist()
        returnDF = returnDF.drop(returnDF_columns, axis=1)

        list_combinations: list[set[str]] = list(
            itertools.combinations(self.explanatoryVariableColumnNames, 2)
        )
        for combination_index in range(len(list_combinations)):
            combination: set[str, str] = list_combinations[combination_index]
            exp_name0: str = combination[0]
            exp_name1: str = combination[1]
            returnDF["col" + str(combination_index)] = (
                inputDF[exp_name0] ** exponent * inputDF[exp_name1]
            )
        for combination_index in range(len(list_combinations)):
            combination: set[str, str] = list_combinations[combination_index]
            exp_name0: str = combination[0]
            exp_name1: str = combination[1]
            returnDF["col" + str(combination_index + len(list_combinations))] = (
                inputDF[exp_name0] * inputDF[exp_name1] ** exponent
            )

        return returnDF


def test_Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression():
    """test_Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression()

    クラス Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression のテスト

    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 13 * np.linspace(20, 10, 11)
    plotX_3: np.ndarray = 23 * np.linspace(10, 20, 11)
    plotX_4: np.ndarray = 29 * np.linspace(30, 40, 11)

    # 係数・切片
    a: int = 17
    b: int = 19
    c: int = 23
    d: int = 29
    e: int = 17
    f: int = 19
    z: int = 31

    # 目的変数
    plotT: np.ndarray = (
        (a * plotX_1 * plotX_1 * plotX_2)
        + (b * plotX_1 * plotX_1 * plotX_3)
        + (c * plotX_2 * plotX_2 * plotX_3)
        + (d * plotX_1 * plotX_2 * plotX_2)
        + (e * plotX_1 * plotX_3 * plotX_3)
        + (f * plotX_2 * plotX_3 * plotX_3)
        + z
    )

    # DFの作成
    columnNames: list[str] = ["process", "plotX_2", "plotX_3", "plotT"]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plotX_2", "plotX_3"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測の実施
    objectModel = Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
        exponent=2,
    )

    # モデルの構築時に使用されるメソッドのテスト
    col0: np.ndarray = plotX_1 * plotX_1 * plotX_2
    col1: np.ndarray = plotX_1 * plotX_1 * plotX_3
    col2: np.ndarray = plotX_2 * plotX_2 * plotX_3
    col3: np.ndarray = plotX_1 * plotX_2 * plotX_2
    col4: np.ndarray = plotX_1 * plotX_3 * plotX_3
    col5: np.ndarray = plotX_2 * plotX_3 * plotX_3
    inputDictForExpectedDF: dict[str, np.ndarray] = {
        "col0": col0,
        "col1": col1,
        "col2": col2,
        "col3": col3,
        "col4": col4,
        "col5": col5,
    }
    expectedDF: pd.DataFrame = pd.DataFrame(data=inputDictForExpectedDF)
    actuallyDF: pd.DataFrame = objectModel.return_df_for_combinations(inputDFForTest, 2)

    assert actuallyDF.equals(
        expectedDF
    ), f"actuallyDF.head()=\n{actuallyDF.head()}\nexpectedDF.head()=\n{expectedDF.head()}"

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装ができているかを確認する
    mape: float = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"

    # 目的変数
    plotT: np.ndarray = (
        (a * plotX_1 * plotX_1 * plotX_1 * plotX_2)
        + (b * plotX_1 * plotX_1 * plotX_1 * plotX_3)
        + (c * plotX_2 * plotX_2 * plotX_2 * plotX_3)
        + (d * plotX_1 * plotX_2 * plotX_2 * plotX_2)
        + (e * plotX_1 * plotX_3 * plotX_3 * plotX_3)
        + (f * plotX_2 * plotX_3 * plotX_3 * plotX_3)
        + z
    )

    # DFの作成
    columnNames: list[str] = ["process", "plotX_2", "plotX_3", "plotT"]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plotX_2", "plotX_3"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測の実施
    objectModel = Model_LinearSumOfElementCombinationWithPower_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
        exponent=3,
    )

    # モデルの構築時に使用されるメソッドのテスト
    col0: np.ndarray = plotX_1 * plotX_1 * plotX_1 * plotX_2
    col1: np.ndarray = plotX_1 * plotX_1 * plotX_1 * plotX_3
    col2: np.ndarray = plotX_2 * plotX_2 * plotX_2 * plotX_3
    col3: np.ndarray = plotX_1 * plotX_2 * plotX_2 * plotX_2
    col4: np.ndarray = plotX_1 * plotX_3 * plotX_3 * plotX_3
    col5: np.ndarray = plotX_2 * plotX_3 * plotX_3 * plotX_3
    inputDictForExpectedDF: dict[str, np.ndarray] = {
        "col0": col0,
        "col1": col1,
        "col2": col2,
        "col3": col3,
        "col4": col4,
        "col5": col5,
    }
    expectedDF: pd.DataFrame = pd.DataFrame(data=inputDictForExpectedDF)
    actuallyDF: pd.DataFrame = objectModel.return_df_for_combinations(inputDFForTest, 3)

    assert actuallyDF.equals(
        expectedDF
    ), f"actuallyDF.head()=\n{actuallyDF.head()}\nexpectedDF.head()=\n{expectedDF.head()}"

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装ができているかを確認する
    mape: float = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[65]:


class Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """説明変数の組み合わせたうえで片方をn乗にするモデル

    Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def __init__(
        self,
        inputDF,
        explanatoryVariableColumnNames: list[str],
        exponent: int,
        responseVariableColumnNames: list[str],
        conditionDictForTest: Dict[str, str] = {},
        targetDF: pd.DataFrame = None,
    ):
        super().__init__(
            inputDF=inputDF,
            explanatoryVariableColumnNames=explanatoryVariableColumnNames,
            responseVariableColumnNames=responseVariableColumnNames,
            conditionDictForTest=conditionDictForTest,
            targetDF=targetDF,
        )
        self.exponent: int = exponent
        self._explanatoryVariableColumnNames = copy.copy(
            self.explanatoryVariableColumnNames
        )
        self._explanatoryVariableColumnNames.remove("process")
        self._rawExplanaoryVariable: pd.DataFrame = self.rawExplanaoryVariable[
            self._explanatoryVariableColumnNames
        ]
        self._rawExplanaoryVariableForTest: pd.DataFrame = (
            self.rawExplanaoryVariableForTest[self._explanatoryVariableColumnNames]
        )

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数及びインスタンスの目的変数からモデルを構築する

        Args:
            self :none

        Returns: boolean。成功ならTrue,失敗ならFalse
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(
            inputDF=self.rawExplanaoryVariable, exponent=self.exponent
        )

        self.lr = LinearRegression()
        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF: pd.DataFrame) -> np.ndarray

        Args:
            self: none
            inputDF (pd.DataFrame) :構築されたモデルを使って予測を行うDF

        Returns:
            np.ndarray
        """

        df_mid_var: pd.DataFrame = self.return_df_for_combinations(
            inputDF, exponent=self.exponent
        )
        resultDF = self.lr.predict(df_mid_var)

        return resultDF

    def returnMAPE(self) -> float:
        """returnMAPE() -> float

        モデルに構築に使用したデータからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合,-1

        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self._rawExplanaoryVariable)

        return returnMapeScore(return_expect, return_actually)

    def return_df_for_combinations(
        self, inputDF: pd.DataFrame, exponent: int
    ) -> pd.DataFrame:
        """return_df_for_combinations(self, inputDF: pd.DataFrame) -> pd.DataFrame

        入力DFから説明変数の組み合わせを算出し、その組み合わせの要素同士を累乗計算をしつつかけ合わせた列で構成されたDFを返す関数

        Args:
            inputDF (pd.DataFrame):入力DF
            exponent (int): 累乗計算時に用いる指数
        """

        returnDF: pd.DataFrame = inputDF.copy(deep=True)
        returnDF_columns = returnDF.columns.tolist()
        returnDF = returnDF.drop(returnDF_columns, axis=1)

        list_combinations: list[set[str]] = list(
            itertools.combinations(self._explanatoryVariableColumnNames, 2)
        )
        for combination_index in range(len(list_combinations)):
            combination: set[str, str] = list_combinations[combination_index]
            exp_name0: str = combination[0]
            exp_name1: str = combination[1]
            returnDF["col" + str(combination_index)] = (
                inputDF[exp_name0] ** exponent * inputDF[exp_name1]
            )
        for combination_index in range(len(list_combinations)):
            combination: set[str, str] = list_combinations[combination_index]
            exp_name0: str = combination[0]
            exp_name1: str = combination[1]
            returnDF["col" + str(combination_index + len(list_combinations))] = (
                inputDF[exp_name0] * inputDF[exp_name1] ** exponent
            )

        return returnDF


def test_Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression():
    """test_Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression()

    クラス Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression のテスト

    """

    # 説明変数
    plotX_1: np.ndarray = np.linspace(10, 20, 11)
    plotX_2: np.ndarray = 13 * np.linspace(20, 10, 11)
    plotX_3: np.ndarray = 23 * np.linspace(10, 20, 11)
    plotX_4: np.ndarray = 29 * np.linspace(30, 40, 11)

    # 係数・切片
    a: int = 17
    b: int = 19
    c: int = 23
    d: int = 29
    e: int = 17
    f: int = 19
    z: int = 31

    # 目的変数
    plotT: np.ndarray = (
        (a * plotX_2**3 * plotX_3)
        + (b * plotX_2**3 * plotX_4)
        + (c * plotX_3**3 * plotX_4)
        + (d * plotX_2 * plotX_3**3)
        + (e * plotX_2 * plotX_4**3)
        + (f * plotX_3 * plotX_4**3)
        + z
    )

    # DFの作成
    columnNames: list[str] = ["process", "plotX_2", "plotX_3", "plotX_4", "plotT"]
    datumForDF: list[np.ndarray] = [plotX_1, plotX_2, plotX_3, plotX_4, plotT]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plotX_2", "plotX_3", "plotX_4"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plotT"]

    # 予測の実施
    objectModel = Model_LinearSumOfElementCombinationWithPowerWithoutProcess_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
        exponent=2,
    )

    # モデルの構築時に使用されるメソッドのテスト
    col0: np.ndarray = plotX_2**3 * plotX_3
    col1: np.ndarray = plotX_2**3 * plotX_4
    col2: np.ndarray = plotX_3**3 * plotX_4
    col3: np.ndarray = plotX_2 * plotX_3**3
    col4: np.ndarray = plotX_2 * plotX_4**3
    col5: np.ndarray = plotX_3 * plotX_4**3
    inputDictForExpectedDF: dict[str, np.ndarray] = {
        "col0": col0,
        "col1": col1,
        "col2": col2,
        "col3": col3,
        "col4": col4,
        "col5": col5,
    }
    expectedDF: pd.DataFrame = pd.DataFrame(data=inputDictForExpectedDF)
    actuallyDF: pd.DataFrame = objectModel.return_df_for_combinations(inputDFForTest, 3)

    assert actuallyDF.equals(
        expectedDF
    ), f"actuallyDF.head()=\n{actuallyDF.head()}\nexpectedDF.head()=\n{expectedDF.head()}"

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装ができているかを確認する
    mape: float = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[66]:


def return_rawDF_cg(
    list_process: list[int],
    list_na: list[int],
    list_nonzer: list[int],
    list_niter: list[int],
    list_shift: list[int],
    csvDir: str,
) -> pd.DataFrame:
    pass
    """return_rawDF_cg()
    
    ベンチマークプログラムCGの手動で変更した初期変数におけるプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_na(list[int]):初期変数naのリスト
        list_nonzer(list[int]):初期変数nonzerのリスト
        list_niter(list[int]):初期変数niterのリスト
        list_shift(list[int]):初期変数shiftのリスト
        csvDir(str):CSVファイルを格納したディレクトリのパスを表す文字列

    Returns:
        pd.DataFrame
    
    """

    list_before_concat_DF: list[pd.DataFrame] = []
    # col_name :list[str] = ["%Time", "Exclusive", "Inclusive", "#Call", "#Subrs", "Inclusive", "Name"]

    for elem_process in list_process:
        for elem_na in list_na:
            for elem_nonzer in list_nonzer:
                for elem_niter in list_niter:
                    for elem_shift in list_shift:
                        filePath: str = f"{csvDir}cg_na{elem_na}_nonzer{elem_nonzer}_niter{elem_niter}_shift{elem_shift}_process{elem_process}.csv"
                        if os.path.isfile(filePath):
                            try:
                                DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                                DF_read_raw["process"] = elem_process
                                DF_read_raw["na"] = elem_na
                                DF_read_raw["nonzer"] = elem_nonzer
                                DF_read_raw["niter"] = elem_niter
                                DF_read_raw["shift"] = elem_shift
                                list_before_concat_DF.append(DF_read_raw)
                            except:
                                warnings.warn(f"{filePath} is empty.")
                        else:
                            warnings.warn(f"{filePath} doesn't exist")
    return pd.concat(objs=list_before_concat_DF, axis=0)


# In[67]:


def return_rawDF_mg(
    list_process: list[int],
    list_problem_size: list[int],
    list_nit: list[int],
    csvDir: str,
) -> pd.DataFrame:
    """return_rawDF_g()

    ベンチマークプログラムMGの手動で変更した初期変数におけるプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_problem_size(list[int]):初期変数problem_sizeのリスト
        list_nit(list[int]):初期変数nitのリスト

    Returns:
        pd.DataFrame

    """

    list_before_concat_DF: list[pd.DataFrame] = []

    for elem_process in list_process:
        for elem_problem_size in list_problem_size:
            for elem_nit in list_nit:
                filePath: str = f"{csvDir}mg_problem_size{elem_problem_size}_nit{elem_nit}_process{elem_process}.csv"
                if os.path.isfile(filePath):
                    try:
                        DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                        DF_read_raw["process"] = elem_process
                        DF_read_raw["problem_size"] = elem_problem_size
                        DF_read_raw["nit"] = elem_nit
                        list_before_concat_DF.append(DF_read_raw)
                    except:
                        warnings.warn(f"{filePath} is empty.")
                else:
                    warnings.warn(f"{filePath} doesn't exist")
    return pd.concat(objs=list_before_concat_DF, axis=0)


# In[68]:


class Model_squareRootOfProcess_ForMultipleRegression(ModelBaseForMultipleRegression):
    """プロセス数を表す説明変数に平方根をかけたモデル

    Y = a * X ** 1/2 + b

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト

    """

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数およびインスタンスの目的変数からモデルを構築する
        """

        df_mid_var: pd.DataFrame = self.process_df(inputDF=self.rawExplanaoryVariable)
        self.lr = LinearRegression()
        df_mid_var = df_mid_var.drop(columns="process")
        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF: pd.DataFrame)->np.ndarray

        構築したモデルを用いて予測をする

        Args:
            self: none
            inputDF (pd.DataFrame) :構築されたモデルを用いて予測を行うDF

        Returns:
            np.ndarray
        """

        df_mid_var: pd.DataFrame = self.process_df(inputDF)
        df_mid_var = df_mid_var.drop(columns="process")
        result_ndarray: np.ndarray = self.lr.predict(df_mid_var)

        return result_ndarray

    def returnMAPE(self) -> float:
        """returnMAPE() -> float

        モデル構築のための学習用データからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合,-1

        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)

        return returnMapeScore(return_expect, return_actually)

    def process_df(self, inputDF: pd.DataFrame) -> pd.DataFrame:
        """process_df(self, inputDF:pd.DataFrame) -> pd.DataFrame

        inputDF内にあるprocess列を1/2乗した列を追加する

        Args:
            inputDF (pd.DataFrame) : 入力DF

        """

        returnDF: pd.DataFrame = inputDF.copy(deep=True)
        returnDF["square_root_of_process"] = np.sqrt(returnDF["process"])
        reutrnDF = returnDF.drop("process", axis=1)

        return returnDF


def test_Model_squareRootOfProcess_ForMultipleRegression():
    """test_Model_squareRootOfProcess_ForMultipleRegression()

    クラスModel_squareRootOfProcess_ForMultipleRegressionのテスト
    """

    # 説明変数
    plot_process: np.ndarray = np.linspace(10, 20, 11)
    plot_other: np.ndarray = -1 * plot_process
    plot_other2: np.ndarray = 5 * plot_process

    # 切片・係数
    a: int = 8
    b: int = -37

    # 目的変数
    plot_call: np.ndarray = np.sqrt(plot_process)

    # DFの作成
    columnNames: list[str] = ["process", "plot_other", "plot_other2", "plot_call"]
    datumForDF: list[np.ndarray] = [plot_process, plot_other, plot_other2, plot_call]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plot_other", "plot_other2"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plot_call"]

    # 予測の実施
    objectModel = Model_squareRootOfProcess_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装ができているかを確認
    mape: float = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[69]:


class Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression(
    ModelBaseForMultipleRegression
):
    """プロセス数の平方根にほかの初期変数をかけたモデル

    Y = a * sqrt(X) * Z + b

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト

    """

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数およびインスタンスの目的変数からモデルを構築する

        """

        df_mid_var: pd.DataFrame = self.process_df(inputDF=self.rawExplanaoryVariable)

        self.lr = LinearRegression()
        self.lr.fit(df_mid_var, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDFF: pd.DataFrame) -> np.ndarray

        構築したモデルを用いて予測をする

        Args:
            self: none
            inputDF (pd.DataFrame) :構築したモデルを用いて予測を行うDF

        Returns:
            np.ndarray

        """

        df_mid_var: pd.DataFrame = self.process_df(inputDF=inputDF)
        predicted_result = self.lr.predict(df_mid_var)

        return predicted_result

    def process_df(self, inputDF: pd.DataFrame) -> pd.DataFrame:
        """process_df(self, inputDFF: pd.DataFrame): -> pd.DataFrame

        | functionName | process | other_exp | other_exp2 |
        |--------------|---------|-----------|------------|
        | 関数名          | プロセス数   |  説明変数     |  説明変数2     |
        | 関数名          | プロセス数   |  説明変数     |  説明変数2     |

        関数コール回数 = a * sqrt(process) * other_exp + b * sqrt(process) * other_exp2 + c

        上記のモデル式を満たすためのDFを作成する。

        """

        returnDF: pd.DataFrame = inputDF.copy(deep=True)
        returnDF_columns = returnDF.columns.tolist()
        returnDF = returnDF.drop(columns=returnDF_columns)

        # プロセス数とほかの説明変数の要素をかけてreturnDF_columnsに入れる
        _process = inputDF["process"]
        for expElem_index in range(len(self.explanatoryVariableColumnNames)):
            expElem: str = self.explanatoryVariableColumnNames[expElem_index]
            if expElem == "process":
                continue
            _expCol: np.ndarray = inputDF[expElem]
            returnDF["col" + str(expElem_index)] = _process * _expCol

        return returnDF

    def returnMAPE(self) -> float:
        """returnMAPE(self) -> float

        モデル構築のための学習用データからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合,-1


        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)

        return returnMapeScore(return_expect, return_actually)


def test_Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression():
    """test_Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression(ModelBaseForMultipleRegression)

    Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression(ModelBaseForMultipleRegression)のテスト
    """

    # 説明変数
    plot_process: np.ndarray = np.linspace(210, 220, 11)
    plot_other: np.ndarray = np.linspace(123, 133, 11)
    plot_other2: np.ndarray = -1 * plot_process

    # 切片・係数
    a: int = 8
    b: int = -41

    # 目的変数
    plot_call: np.ndarray = a * plot_process * plot_other + b

    # DFの作成
    columnNames: list[str] = ["process", "plot_other", "plot_other2", "plot_call"]
    datumForDF: list[np.ndarray] = [plot_process, plot_other, plot_other2, plot_call]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plot_other", "plot_other2"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plot_call"]

    # 予測の実施
    objectModel = Model_sqrtProcessTimesOtherExpElem_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装できているかを確認
    mape: float = objectModel.returnMAPE()
    assert 0 <= mape < 1, f"mape = {mape}"


# In[70]:


"""
関数コール回数と初期変数の値が完全に一致していることがある。（ベンチマークプログラムCGのSPRNVC,VECSETなど）

これを満たすモデルを作成する

"""


class Model_obeyOneParameter_ForMultipleRegression(ModelBaseForMultipleRegression):
    """一つの説明変数に完全に従うモデル

    Y = X（パラメータはXだけではないがXのみに従っている）

    Attributes:
        explanatoryVariableColumnNames (list[str]): 説明変数の列名のリスト
        rawExplanatoryVariable (pd.DataFrame): 説明変数のデータフレーム
        rawExplanatoryVariableForTest (pd.DataFrame): テスト用の説明変数のデータフレーム。説明変数のデータフレームと同様の値が入っている(?)
        rawResponseVariable (pd.DataFrame): 目的変数のデータフレーム
        rawResponseVariableForTest (pd.DataFrame): テスト用の目的変数のデータフレーム。目的変数のデータフレームと同様の値が入っている(?)
        responseVariableColumnNames (list[str]): 目的変数の列名のリスト
    """

    def build_model(self) -> bool:
        """build_model(self) -> bool

        オブジェクトの初期化時に生成された、インスタンスの説明変数およびインスタンスの目的変数からモデルを構築する
        """

        self.oneParam = ""

        _resVar: np.ndarray = self.rawResponseVariable.values.reshape(-1)
        for expColName in self.explanatoryVariableColumnNames:
            _expVar: np.ndarra = self.rawExplanaoryVariable[expColName].values
            if (_resVar == _expVar).all():
                self.oneParam = expColName
                continue
        if self.oneParam == "":
            self.lr = LinearRegression()
            self.lr.fit(self.rawExplanaoryVariable, self.rawResponseVariable)

        return True

    def predict(self, inputDF: pd.DataFrame) -> np.ndarray:
        """predict(self, inputDF: pd.DataFrame)->np.ndarray

        構築したモデルを用いて予測をする

        Args:
            self: ___
            inputDF (pd.DataFrame) :構築されたモデルを用いて予測を行うDF

        Returns:
            np.ndarray
        """

        inputDFonlyExpVarCol: pd.DataFrame = inputDF[
            self.explanatoryVariableColumnNames
        ]
        if self.oneParam == "":
            predicted_result: np.ndarray = self.lr.predict(
                inputDFonlyExpVarCol
            ).reshape(-1)
        else:
            predicted_result: np.ndarray = np.reshape(
                inputDFonlyExpVarCol[self.oneParam].values, (-1, 1)
            )

        return predicted_result

    def returnMAPE(self) -> float:
        """returnMAPE() -> float

        モデル構築のための学習用データからMAPEを算出する。

        Args:
            self: none

        Returns:
            float: 「モデルの構築に用いたデータから予測された値」と「実際の値」から算出されたMAPE
            int: 失敗した場合,-1

        """

        return_expect: np.ndarray = self.rawResponseVariable[
            self.responseVariableColumnNames
        ].values
        return_actually: np.ndarray = self.predict(self.rawExplanaoryVariable)
        return returnMapeScore(return_expect, return_actually)


def test_Model_obeyOneParameter_ForMultipleRegression():
    """test_Model_obeyOneParameter_ForMultipleRegression()

    クラス Model_obeyOneParameter_ForMultipleRegression のテスト
    """

    # 説明変数
    plot_process: np.ndarray = np.linspace(10, 20, 11)
    plot_other: np.ndarray = 1000 * np.random.random_sample(11)
    plot_other2: np.ndarray = -3 * plot_process

    # 目的変数
    plot_call: np.ndarray = plot_other

    # DFの作成
    columnNames: list[str] = ["process", "plot_other", "plot_other2", "plot_call"]
    datumForDF: list[np.ndarray] = [plot_process, plot_other, plot_other2, plot_call]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T
    inputDFForTest["functionName"] = "functionName"

    # 説明変数のカラム名のリスト
    columnNamesForExp: list[str] = ["process", "plot_other", "plot_other2"]
    # 目的変数のカラム名のリスト
    columnNamesForRes: list[str] = ["plot_call"]

    # 予測の実施
    objectModel = Model_obeyOneParameter_ForMultipleRegression(
        inputDF=inputDFForTest,
        explanatoryVariableColumnNames=columnNamesForExp,
        responseVariableColumnNames=columnNamesForRes,
        conditionDictForTest={},
    )

    # モデルの構築
    objectModel.build_model()
    # モデル構築に用いたデータと予測されたデータとのMAPEを比較して、実装できているかを確認
    mape: float = objectModel.returnMAPE()

    assert 0 <= mape < 0.01, f"mape = {mape}"


# In[71]:


from operator import index


def convertPprofTime(input: str) -> float:
    """TAUで取得しpprofで集計したプロファイルの'Inclusive～', 'Exclusive～'列の時刻を文字列(string)として取得して、数値(float)として返す関数。単位は秒

    Args:
        input (str) : 文字列となっているpprofで集計したプロファイルの'Inclusive～'もしくは'Exclusive～'列の時刻

    Returns:
        float : inputを秒単位にした値
    """

    # ',' を削除
    input = str(input).replace(",", "")
    # ':' の有無を確認
    # ':' のインデックスを探す
    index_colon_first: int = input.find(":")
    index_colon_last: int = input.rfind(":")
    # '.' の有無を確認
    # '.' のインデックスを探す
    index_period: int = input.find(".")

    ret_seconds: float = 0
    hours: int = 0
    minutes: int = 0
    seconds: float = 0

    if index_colon_first == -1 and index_colon_last == -1:
        # 秒のみ
        seconds = float(input) / 1000
    elif index_colon_first == index_colon_last:
        # 分,秒のみ
        minutes = int(input[:index_colon_first])
        seconds = float(input[index_colon_first + 1 :])
    elif index_colon_first != index_colon_last:
        # 時,分,秒
        hours = int(input[:index_colon_first])
        minutes = int(input[index_colon_first + 1 : index_colon_last])
        seconds = float(input[index_colon_last + 1 :])

    # print(f"colon = {index_colon}, period = {index_period}\ninput = {input}\nminutes={minutes}, seconds={seconds}")

    ret_seconds = 3600 * hours + 60 * minutes + seconds

    return ret_seconds


def test_convertPprofTime():
    """convertPprofTime(input :str) -> float のテスト関数"""

    test_case_00_input: str = "1:44.607"
    test_case_00_output_actually: float = 1 * 60 + 44 + 0.607
    test_case_00_output_expect: float = convertPprofTime(test_case_00_input)
    assert (
        test_case_00_output_actually == test_case_00_output_expect
    ), f"actually = {test_case_00_output_actually}, expect = {test_case_00_output_expect}"

    test_case_01_input: str = "33,350"
    test_case_01_output_actually: float = 33.350
    test_case_01_output_expect: float = convertPprofTime(test_case_01_input)
    assert (
        test_case_01_output_actually == test_case_01_output_expect
    ), f"actually = {test_case_01_output_actually}, expect = {test_case_01_output_expect}"

    test_case_02_input: str = "111:111"
    test_case_02_output_actually: float = 111 * 60 + 111
    test_case_02_output_expect: float = convertPprofTime(test_case_02_input)
    assert (
        test_case_02_output_actually == test_case_02_output_expect
    ), f"actually = {test_case_02_output_actually}, expect = {test_case_02_output_expect}"

    test_case_03_input: str = "0:123"
    test_case_03_output_actually: float = 123
    test_case_03_output_expect: float = convertPprofTime(test_case_03_input)
    assert (
        test_case_03_output_actually == test_case_03_output_expect
    ), f"actually = {test_case_03_output_actually}, expect = {test_case_03_output_expect}"

    test_case_04_input: str = "0.0269"
    test_case_04_output_actually: float = 0.0269 / 1000
    test_case_04_output_expect: float = convertPprofTime(test_case_04_input)
    assert (
        test_case_04_output_actually == test_case_04_output_expect
    ), f"actually = {test_case_04_output_actually}, expect = {test_case_04_output_expect}"

    test_case_05_input: str = "1:16:15.802"
    test_case_05_output_expect: float = 1 * 3600 + 16 * 60 + 15.802
    test_case_05_output_actually: float = convertPprofTime(test_case_05_input)
    assert (
        test_case_05_output_expect == test_case_05_output_actually
    ), f"actually = {test_case_05_output_actually}, expect = {test_case_05_output_expect}"


# In[72]:


def get_execTime(
    benchmarkName: str,
    condition: dict[str, int],
    csvDir: str,
    columnName: str = "Inclusive",
):
    """TAUで取得し、pprofで整形したcsvから実行時間を取得する関数

    Args:
        benchmarkName (str) : ベンチマーク名
        condition (dict[str, int]) : 実行条件（プロセス数や問題サイズなど）
        csvDir (str) : CSVファイルのディレクトリ

    Returns:
        float : inputを秒単位にした値
    """

    if benchmarkName == "cg":
        target_rawDF_cg: pd.DataFrame = return_rawDF_cg(
            list_process=[condition["process"]],
            list_na=[condition["na"]],
            list_niter=[condition["niter"]],
            list_nonzer=[condition["nonzer"]],
            list_shift=[condition["shift"]],
            csvDir=csvDir,
        )

        target_function_DF: pd.DataFrame = target_rawDF_cg[
            target_rawDF_cg["Name"] == ".TAU_application"
        ]
        target_num: str = target_function_DF.loc[0, columnName]
        return convertPprofTime(target_num)
    elif benchmarkName == "lulesh":
        target_rawDF_lulesh: pd.DataFrame = return_rawDF_lulesh(
            list_process=[condition["process"]],
            list_iteration=[condition["iteration"]],
            list_size=[condition["size"]],
            csvDir=csvDir,
        )
        target_function_DF: pd.DataFrame = target_rawDF_cg[
            target_rawDF_cg["Name"] == ".TAU_application"
        ]
        target_num: str = target_function_df.loc[0, columnName]
        return convertPprofTime(target_num)
    else:
        warnings.warn(f"there no process for {benchmarkName}(condition = {condition})")
        return -1


# In[73]:


def ret_relativeCost(
    base_DFs: list[pd.DataFrame], target_DFs: list[pd.DataFrame]
) -> float:
    """相対コストを算出する関数

    Args:
        base_DFs (list[pd.DataFrame]) : モデルの構築に利用したDFのリスト
        targetDFs (list[pd.DataFrame]) : 予測対象のDFのリスト
        上記の2つのDFは下記のDFの要素を満たす

        | Exclusive | #Name |
        |-----------|-------|
        |           |       |

    Returns:
        float : 相対コスト（％）
    """

    base_DF: pd.DataFrame = pd.concat(base_DFs)
    base_num: float = sum(base_DF[base_DF["Name"] == ".TAU_application"]["Exclusive"])

    target_DF: ps.DataFrame = pd.concat(target_DFs)
    target_num: float = sum(
        target_DF[target_DF["Name"] == ".TAU_application"]["Exclusive"]
    )

    return base_num / target_num * 100


def test_ret_relativeCost():
    """ret_relativeCost()のテスト"""

    base_01_DF: pd.DataFrame = pd.DataFrame(
        {"Exclusive": [1], "Name": [".TAU_application"]}
    )
    base_02_DF: pd.DataFrame = pd.DataFrame(
        {"Exclusive": [2], "Name": [".TAU_application"]}
    )
    base_03_DF: pd.DataFrame = pd.DataFrame(
        {"Exclusive": [3], "Name": [".TAU_application"]}
    )

    target_DF: pd.DataFrame = pd.DataFrame(
        {"Exclusive": [17], "Name": [".TAU_application"]}
    )

    result_expected: float = (1 + 2 + 3) / 17 * 100
    result_actually: float = ret_relativeCost(
        base_DFs=[base_01_DF, base_02_DF, base_03_DF], target_DFs=[target_DF]
    )

    assert (
        result_expected == result_actually
    ), f"expected={result_expected}, actually={result_actually}"


# In[74]:


def gen_ExtraPinputData(
    benchmarkName: str,
    conditions: dict[str, list[int]],
    csvDir: str = "./csv_files/",
    version: int = 1,
    timeColumnName: str = "Inclusive",
) -> str:
    if version == 1:
        if benchmarkName == "cg":
            target_rawDF_cg: pd.DataFrame = return_rawDF_cg(
                list_process=conditions["process"],
                list_na=conditions["na"],
                list_nonzer=conditions["nonzer"],
                list_niter=conditions["niter"],
                list_shift=conditions["shift"],
                csvDir=csvDir,
            )
            target_rawDF_cg = target_rawDF_cg[
                target_rawDF_cg["Name"] == ".TAU_application"
            ]
            target_rawDF_cg["convertedInclusive"] = target_rawDF_cg["Inclusive"].map(
                convertPprofTime
            )

            # めも：conditionsのキーがパラメータになっている
            # PARAMETER
            ss_PARAMETER: str = ""
            list_conditions_keys = sorted(list(conditions.keys()))
            for key in list_conditions_keys:
                ss_PARAMETER += f"PARAMETER {key}\n"
            # POINTS
            ss_POINTS: str = "\n"
            # REGION & METRIC
            ss_REGIONandMETRIC: str = "\nREGION reg\nMETRIC time\n"
            # DATA
            ss_DATA: str = "\n"

            for i, sr in target_rawDF_cg.iterrows():
                ss_POINTS += "POINTS ("
                for key in list_conditions_keys:
                    ss_POINTS += f" {sr[key]}"
                ss_POINTS += " )\n"
                ss_DATA += f"DATA ( {sr['convertedInclusive']} )\n"

            # print(target_rawDF_cg)

            return ss_PARAMETER + ss_POINTS + ss_REGIONandMETRIC + ss_DATA

        else:
            warnings.warn(f"benchmarkName={benchmarkName} は非対応です")
            return ""
    else:
        timeColumnName_converted: str = f"converted{timeColumnName}"
        if benchmarkName == "cg":
            target_rawDF_cg: pd.DataFrame = return_rawDF_cg(
                list_process=conditions["process"],
                list_na=conditions["na"],
                list_nonzer=conditions["nonzer"],
                list_niter=conditions["niter"],
                list_shift=conditions["shift"],
                csvDir=csvDir,
            )
            target_rawDF_cg = target_rawDF_cg[
                target_rawDF_cg["Name"] == ".TAU_application"
            ]
            target_rawDF_cg[timeColumnName_converted] = target_rawDF_cg[
                timeColumnName
            ].map(convertPprofTime)

            # めも：conditionsのキーがパラメータになっている
            # PARAMETER
            ss_PARAMETER: str = ""
            list_conditions_keys = sorted(list(conditions.keys()))
            for key in list_conditions_keys:
                ss_PARAMETER += f"PARAMETER {key}\n"
            # POINTS
            ss_POINTS: str = "\n"
            # REGION & METRIC
            ss_REGIONandMETRIC: str = "\nREGION reg\nMETRIC time\n"
            # DATA
            ss_DATA: str = "\n"

            for i, sr in target_rawDF_cg.iterrows():
                ss_POINTS += "POINTS ("
                for key in list_conditions_keys:
                    ss_POINTS += f" {sr[key]}"
                ss_POINTS += " )\n"
                ss_DATA += f"DATA ( {sr[timeColumnName_converted]} )\n"

            # print(target_rawDF_cg)

            return ss_PARAMETER + ss_POINTS + ss_REGIONandMETRIC + ss_DATA

        elif benchmarkName == "lulesh":
            target_rawDF_lulesh: pd.DataFrame = return_rawDF_lulesh(
                list_process=conditions["process"],
                list_iteration=conditions["iteration"],
                list_size=conditions["size"],
                csvDir=csvDir,
            )
            target_rawDF_lulesh = target_rawDF_lulesh[
                target_rawDF_lulesh["Name"] == ".TAU_application"
            ]
            target_rawDF_lulesh[timeColumnName_converted] = target_rawDF_lulesh[
                timeColumnName
            ].map(convertPprofTime)

            # PARAMETER
            ss_PARAMETER: str = ""
            list_conditions_keys: list[str] = sorted(list(conditions.keys()))
            for key in list_conditions_keys:
                ss_PARAMETER += f"PARAMETER {key}\n"
            # POINTS
            ss_POINTS: str = "\n"
            # REGION & METRIC
            ss_REGIONandMETRIC: str = "\nREGION reg\nMETRIC time\n"
            # DATA
            ss_DATA: str = "\n"

            for i, sr in target_rawDF_lulesh.iterrows():
                ss_POINTS += "POINTS ("
                for key in list_conditions_keys:
                    ss_POINTS += f" {sr[key]}"
                ss_POINTS += " )\n"
                ss_DATA += f"DATA {sr[timeColumnName_converted]} {sr[timeColumnName_converted]} {sr[timeColumnName_converted]}\n"

            return ss_PARAMETER + ss_POINTS + ss_REGIONandMETRIC + ss_DATA

        else:
            warnings.warn(f"benchmarkName={benchmarkName} は非対応です")
            return ""


# In[75]:


def return_rawDF_lulesh(
    list_process: list[int],
    list_iteration: list[int],
    list_size: list[int],
    csvDir: str,
):
    """ベンチマークプログラムLULESHのプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_iteration(list[int]):初期変数iterationのリスト
        list_size(list[int]):初期変数sizeのリスト
        csvDir(str):CSVファイルを格納したディレクトリのパスを表す文字列

    Returns:
        pd.DataFrame

    """

    list_before_concat_DF: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_iteration in list_iteration:
            for elem_size in list_size:
                filePath: str = (
                    f"{csvDir}lulesh_p{elem_process}i{elem_iteration}s{elem_size}.csv"
                )
                DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                if os.path.isfile(filePath):
                    try:
                        DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                        DF_read_raw["process"] = elem_process
                        DF_read_raw["iteration"] = elem_iteration
                        DF_read_raw["size"] = elem_size
                        list_before_concat_DF.append(DF_read_raw)
                    except:
                        warnings.warn(f"{filePath} is empty.")
                else:
                    warnings.warn(f"{filePath} doesn't exist")
    return pd.concat(objs=list_before_concat_DF, axis=0)


# In[76]:


def gen_ExtraPinputDataFromDF(
    inputDF: pd.DataFrame,
    expVar: list[str],
    resVar: str,
) -> str:
    """Extra-Pへの入力ファイルを文字列型で作成してそれを返す関数

    Args:
        inputDF(pd.DataFrame):データフレーム
        expVar(list[str]):説明変数のカラム名のリスト
        resVar(str):目的変数のカラム名

    """

    # PARAMETER
    ss_PARAMETER: str = ""
    sorted_expVar: str = sorted(expVar)
    for exp in sorted_expVar:
        ss_PARAMETER += f"PARAMETER {exp}\n"
    # POINTS
    ss_POINTS: str = "\n"
    # REGION & METRIC
    ss_REGINONandMETRIC: str = "\nREGION reg\nMETRIC time\n"
    # DATA
    ss_DATA: str = "\n"

    for i, sr in inputDF.iterrows():
        ss_POINTS += "POINTS ("
        for exp in sorted_expVar:
            ss_POINTS += f" {sr[exp]}"
        ss_POINTS += " )\n"
        ss_DATA += f"DATA {sr[resVar]} {sr[resVar]} {sr[resVar]}\n"

    return ss_PARAMETER + ss_POINTS + ss_REGINONandMETRIC + ss_DATA


def test_gen_ExtraPinputDataFromDF():
    sequence_1: list[int] = [1, 2, 3, 4, 5]
    sequence_2: list[int] = [6, 7, 8, 9, 10]
    sequence_3: list[int] = [11, 12, 13, 14, 15]
    sequence_4: list[int] = [16, 17, 18, 19, 20]

    columnNames: list[str] = ["sq1", "sq2", "sq3", "sq4"]
    datumForDF: list[list[int]] = [sequence_1, sequence_2, sequence_3, sequence_4]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=columnNames, data=datumForDF).T

    expVar = ["sq1", "sq2", "sq3"]
    resVar = "sq4"

    # sq1, sq2, sq3 を説明変数、sq4 を目的変数とする
    ss_parameter = """PARAMETER sq1
PARAMETER sq2
PARAMETER sq3

"""
    ss_points = """POINTS ( 1 6 11 )
POINTS ( 2 7 12 )
POINTS ( 3 8 13 )
POINTS ( 4 9 14 )
POINTS ( 5 10 15 )

"""
    ss_region_and_metric = """REGION reg
METRIC time

"""
    ss_data = """DATA 16 16 16
DATA 17 17 17
DATA 18 18 18
DATA 19 19 19
DATA 20 20 20
"""

    expected = ss_parameter + ss_points + ss_region_and_metric + ss_data
    actually = gen_ExtraPinputDataFromDF(
        inputDF=inputDFForTest, expVar=expVar, resVar=resVar
    )

    assert expected == actually, f"expected={expected}__actually={actually}__"


# In[77]:


def convert_log(inputStr: str) -> str:
    """Extra-Pで出力されたモデル式の文字列のlogの形式をsympyで読み取れるように変換する関数

    Args:
        inputStr(str):Extra-Pで出力されたモデル式の文字列

    """
    pattern = re.compile(r"log(?P<base>\d{1,})")
    result = pattern.sub(r"1/ln(\g<base>)*ln", inputStr)
    return result


def test_convert_log():
    inputStr: str = """0.008938888890764687 + 0.0003391531590741526 * iteration^(1) + 2.645514630961448e-08 * iteration^(1) * size^(3) * log2(size)^(1)"""
    expected: str = "0.008938888890764687 + 0.0003391531590741526 * iteration^(1) + 2.645514630961448e-08 * iteration^(1) * size^(3) * 1/ln(2)*ln(size)^(1)"
    actually: str = convert_log(inputStr=inputStr)
    assert expected == actually, f"expected = {expected}, actually = {actually}"

    inputStr: str = """0.008938888890764687 + 0.0003391531590741526 * iteration^(1) + 2.645514630961448e-08 * iteration^(1) * size^(3) * log25(size)^(1)"""
    expected: str = "0.008938888890764687 + 0.0003391531590741526 * iteration^(1) + 2.645514630961448e-08 * iteration^(1) * size^(3) * 1/ln(25)*ln(size)^(1)"
    actually: str = convert_log(inputStr=inputStr)
    assert expected == actually, f"expected = {expected}, actually = {actually}"


# In[78]:


def add_relativeErrorRateCol(
    inputDF: pd.DataFrame, real_colName: str, predicted_colName: str, targetColName: str
) -> pd.DataFrame:
    """入力DFに指定された列からなる相対誤差率を計算した列を追加する関数

    Args:
        inputDF(pd.DataFrame):入力DF
        real_colName(str):実測値を保持した列名
        predicted_colName(str):予測値を保持した列名
        targetColName(str):相対誤差率を保持した列名
    """

    returnDF: pd.DataFrame = inputDF.copy(deep=True)

    real_col: list(float) = list(returnDF.loc[:, real_colName])
    predicted_col: list(float) = list(returnDF.loc[:, predicted_colName])

    target_col: list(float) = returnRawRelativeErrorRateList(
        real=real_col, predicted=predicted_col
    )

    try:
        returnDF = returnDF.astype({f"{targetColName}": float})
    except:
        pass

    returnDF[targetColName] = target_col

    return returnDF


def test_add_relativeErrorRateCol():
    """add_relativeErrorRateCol()のテスト"""
    sq1: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
    sq2: list[float] = [6.0, 7.0, 8.0, 9.0, 10.0]
    colNames: list[str] = ["sq1", "sq2"]
    datumForDF: list[list[int]] = [sq1, sq2]
    inputDFForTest: pd.DataFrame = pd.DataFrame(index=colNames, data=datumForDF).T

    real_colName: str = colNames[0]
    predicted_colName: str = colNames[1]
    targetColName: str = "targetColName"
    targetCol: list[float] = [
        (6 - 1) / 1 * 100,
        (7 - 2) / 2 * 100,
        (8 - 3) / 3 * 100,
        (9 - 4) / 4 * 100,
        (10 - 5) / 5 * 100,
    ]

    expected: pd.DataFrame = pd.DataFrame(
        index=["sq1", "sq2", "targetColName"], data=[sq1, sq2, targetCol]
    ).T
    actually: pd.DataFrame = add_relativeErrorRateCol(
        inputDF=inputDFForTest,
        real_colName=real_colName,
        predicted_colName=predicted_colName,
        targetColName=targetColName,
    )

    assert actually.equals(
        expected
    ), f"actually.head() = \n{actually.head()}expected.head() = \n{expected.head()}"


def returnRawRelativeErrorRate(real: float | int, predicted: float | int) -> float:
    """相対誤差率を算出する関数

    Args:
        real(float|int):実測値
        predicted(float|int):予測値

    Return:
        float: 相対誤差率（単位：％）
    """
    retNum: float
    try:
        retNum = abs(real - predicted) / real * 100
    except:
        retNum = np.nan
    return retNum


def returnRawRelativeErrorRateList(
    real: list[float | int], predicted: list[float | int]
) -> list[float]:
    """相対誤差率のリストを算出する関数

    Args:
        real(list[float|int]):実測値のリスト
        predicted(list[float|int]):予測値のリスト

    Return:
        list[float]: 相対誤差率（単位：％）のリスト
    """

    if len(real) != len(predicted):
        warnings.warn("引数のリストの長さが異なります")
    else:
        return_list: list[float] = []
        for i in range(len(real)):
            real_num: float | int = real[i]
            predicted_num: float | int = predicted[i]
            relativeErrorRate: float = returnRawRelativeErrorRate(
                real=real_num, predicted=predicted_num
            )
            return_list.append(relativeErrorRate)
    return return_list


def test_returnRawRelativeErrorRateList():
    """returnRawRelativeErrorRateList()のテスト"""

    sq1: list[int] = [1, 2, 3, 4, 5]
    sq2: list[int] = [6, 7, 8, 9, 10]
    expected: list[float] = [
        abs((1 - 6) / 6 * 100),
        abs((2 - 7) / 7 * 100),
        abs((3 - 8) / 8 * 100),
        abs((4 - 9) / 9 * 100),
        abs((5 - 10) / 10 * 100),
    ]
    actually: list[float] = returnRawRelativeErrorRateList(real=sq2, predicted=sq1)

    assert expected == actually, f"expected=\n{expected}\nactually=\n{actually}"


# In[79]:


def ret_averagedDF(inputDFs: list[pd.DataFrame], resVar: str) -> pd.DataFrame:
    """入力されたDFのリストをまとめる。まとめる対象はresVarで指定したカラム名

    Args
        inputDFs(list[pd.DataFrame]):入力DFのリスト
        resVar(str):まとめる対象となるカラム名

    Returns
        pd.DataFrame:概要通りの処理がなされたDF
    """

    if len(inputDFs) == 0:
        warnings.warn("len(inputDFs) == 0")
        return -1

    set_index_size = set()
    set_column_size = set()
    for i in range(len(inputDFs)):
        set_index_size.add(len(inputDFs[i]))
        set_column_size.add(len(inputDFs[i].columns))
    if len(set_index_size) != 1 or len(set_column_size) != 1:
        warnings.warn("some DF in inputDFs is different size")
        return -1

    list_res_series: list[pd.Series] = []
    res_series_col: list[int] = []
    for i in range(len(inputDFs)):
        list_res_series.append(inputDFs[i][resVar])
        res_series_col.append(i)

    averaged_res_series: pd.Series = pd.DataFrame(
        index=res_series_col, data=list_res_series
    ).mean()

    returnDF = inputDFs[0].copy(deep=True)
    returnDF[resVar] = averaged_res_series

    return returnDF


def test_ret_averagedDF():
    sq1: list[float] = [0.0, 1.0, 2.0, 3.0, 4.0]
    sq2: list[float] = [5.0, 6.0, 7.0, 8.0, 9.0]
    sq3: list[float] = [10.0, 11.0, 12.0, 13.0, 14.0]

    sq_datum1: list[int] = [20, 21, 22, 23, 24]
    sq_datum2: list[int] = [30, 31, 32, 33, 34]
    sq_datum3: list[int] = [40, 41, 42, 43, 44]

    colNames: list[str] = ["sq1", "sq2", "sq3", "data"]

    datumList: list[list[int]] = [sq1, sq2, sq3, sq_datum1]
    inputDF1: pd.DataFrame = pd.DataFrame(index=colNames, data=datumList).T
    datumList: list[list[int]] = [sq1, sq2, sq3, sq_datum2]
    inputDF2: pd.DataFrame = pd.DataFrame(index=colNames, data=datumList).T
    datumList: list[list[int]] = [sq1, sq2, sq3, sq_datum3]
    inputDF3: pd.DataFrame = pd.DataFrame(index=colNames, data=datumList).T

    expVar: list[str] = ["sq1", "sq2", "sq3"]
    resVar: str = "data"

    expected_datum: list[float] = [
        (20 + 30 + 40) / 3,
        (21 + 31 + 41) / 3,
        (22 + 32 + 42) / 3,
        (23 + 33 + 43) / 3,
        (24 + 34 + 44) / 3,
    ]
    datumList = [sq1, sq2, sq3, expected_datum]
    expected: pd.DataFrame = pd.DataFrame(index=colNames, data=datumList).T

    actually: pd.DataFrame = ret_averagedDF(
        inputDFs=[inputDF1, inputDF2, inputDF3], resVar=resVar
    )

    assert expected.equals(actually), f"expected=\n{expected}\nactually=\n{actually}"


# In[80]:


def add_perCallColumn(
    inputDF: pd.DataFrame,
    dividendColName: str,
    divisorColName: str,
    targetColumnName: str,
) -> pd.DataFrame:
    """指定した列名にある列Aの値をある列Bの値で割った値を格納して、そのDFを返す関数

    Args:
        inputDF(pd.DataFrame):入力DF
        divisorColName(str):割る値を保持した列
        dividendColName(str):割られる値を保持した列
        targetColumnName(str):計算結果を保持した列

    Return:
        pd.DataFrame:入力DFにtarfgetColumnName列が追加されたDF
    """

    returnDF: pd.DataFrame = inputDF.copy(deep=True)
    returnDF[targetColumnName] = -1
    for i, sr in returnDF.iterrows():
        dividend: float = sr[dividendColName]
        divisor: float = float(sr[divisorColName])
        _tmp_num: float = dividend / divisor
        returnDF.at[i, targetColumnName] = _tmp_num

    return returnDF


def test_add_perCallColumn():
    col1: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    col2: list[float] = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    expectedCol: list[float] = [
        1 / 9,
        2 / 8,
        3 / 7,
        4 / 6,
        5 / 5,
        6 / 4,
        7 / 3,
        8 / 2,
        9 / 1,
    ]
    datumForDF: list[list[int]] = [col1, col2]
    colName: list[str] = ["col1", "col2"]
    test_inputDF: pd.DataFrame = pd.DataFrame(data=datumForDF, index=colName).T
    datumForDF.append(expectedCol)
    colName.append("target")
    expected: pd.DataFrame = pd.DataFrame(data=datumForDF, index=colName).T
    actually: pd.DataFrame = add_perCallColumn(
        inputDF=test_inputDF,
        dividendColName="col1",
        divisorColName="col2",
        targetColumnName="target",
    )

    assert expected.equals(actually), f"actually=\n{actually}\nexpected=\n{expected}"


# In[81]:


def ret_averaged_rawDF_lulesh(
    list_process: list[int],
    list_iteration: list[int],
    list_size: list[int],
    list_csvDir: list[str],
    resVar: str,
):
    """複数のCSVからDFを取得する関数

    引数resVarで指定された列がInclusiveもしくはExclusiveの場合はそれらが秒に変換され、InclusivePerCallもしくはExclusivePerCall列が生成されている

    Args:
        list_process(list[int]):プロセス数

    """
    list_DFs_for_return: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_iteration in list_iteration:
            for elem_size in list_size:
                list_inputDFs_for_averaged: list[pd.DataFrame] = []
                for elem_csvDir in list_csvDir:
                    _raw_DF: pd.DataFrame = return_rawDF_lulesh(
                        list_process=[elem_process],
                        list_iteration=[elem_iteration],
                        list_size=[elem_size],
                        csvDir=elem_csvDir,
                    )

                    if resVar in ["Exclusive", "Inclusive", "#Call", "#Subrs"]:
                        # resVar 列の整形
                        if resVar in ["Exclusive", "Inclusive"]:
                            _tmp_converted = map(
                                convertPprofTime, list(_raw_DF[resVar])
                            )
                            _raw_DF[resVar] = list(_tmp_converted)
                        # {resVar}PerCall 列の生成
                        _raw_DF = add_perCallColumn(
                            inputDF=_raw_DF,
                            divisorColName="#Call",
                            dividendColName=resVar,
                            targetColumnName=f"{resVar}PerCall",
                        )

                    list_inputDFs_for_averaged.append(_raw_DF)
                list_DFs_for_return.append(
                    ret_averagedDF(inputDFs=list_inputDFs_for_averaged, resVar=resVar)
                )
    return pd.concat(objs=list_DFs_for_return, axis=0)


# In[82]:


def ret_relativeError_fromSumOfCol(
    inputDF: pd.DataFrame, realCol: str, predictedCol: str
) -> float:
    """入力DFにおいて指定された列の総和の相対誤差率を返す関数

    Args:
        inputDF(pd.DataFrame):入力DF
        realCol(str):実測値のカラム名
        predictedCol(str):予測値のカラム名
    """

    sumReal: float = sum(inputDF[realCol])
    sumPredicted: float = sum(inputDF[predictedCol])

    return abs(sumReal - sumPredicted) / sumReal * 100


def test_ret_relativeError_fromSumOfCol():
    col1: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # 実測値
    col2: list[float] = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # 予測値

    colNames: list[str] = ["col1", "col2"]
    inputDFforTest: pd.DataFrame = pd.DataFrame(index=colNames, data=[col1, col2]).T

    expected: float = abs(sum(col1) - sum(col2)) / sum(col1) * 100
    actually: float = ret_relativeError_fromSumOfCol(
        inputDF=inputDFforTest, realCol="col1", predictedCol="col2"
    )

    assert expected == actually, f"expected = {expected}, actually = {actually}"


# In[83]:


def get_CostAtInputDF(
    inputDF: pd.DataFrame, numOfProcess: int, targetColName: str = "Exclusive"
) -> float:
    """DFにおけるコストを算出する関数
    Args:
        inputDF(pd.DataFrame):入力
        targetColName(str):inputDFにおける合計する列名
        numOfProcess(int):プロセス数
    """
    sumOfTargetCol: float = sum(inputDF[targetColName])
    return sumOfTargetCol * numOfProcess


def test_get_CostAtInputDF():
    col1: list[int] = [1, 2, 3, 4, 5]
    col2: list[int] = [5, 6, 7, 8, 9]
    colName: list[str] = ["col1", "col2"]
    numOfProcess: int = 512

    inputDFforTest: pd.DataFrame = pd.DataFrame(index=colName, data=[col1, col2]).T

    expected: float = sum(col1) * numOfProcess
    actually: float = get_CostAtInputDF(
        inputDF=inputDFforTest, numOfProcess=numOfProcess, targetColName="col1"
    )
    assert expected == actually

    expected: float = sum(col2) * numOfProcess
    actually: float = get_CostAtInputDF(
        inputDF=inputDFforTest, numOfProcess=numOfProcess, targetColName="col2"
    )
    assert expected == actually


# In[84]:


def get_str_modelFromExtraP(
    filePath: str,
    modelerName: str,
    modelerOption: str,
    extraPinputData: str,
):
    """

    引数からExtra-Pでのモデルを構築し、構築したモデルの文字列を返す関数

    Args:
        filePath(str) : モデル構築用ファイルのパス
        modelerName(str) : Extra-Pのモデル名、'default' or 'multi-parameter'
        modelerOption(str) : Extra-Pのオプション、'--options ~~~' の形式
        extraPinputData(str) : モデル構築用ファイルの中身となる文字列

    """

    with open(file=filePath, mode="w") as f:
        f.write(extraPinputData)

    str_resFromExtraP: str = subprocess.run(
        f"extrap --text {filePath} --modeler {modelerName} {modelerOption} 2>/dev/null | grep Model",
        stdout=subprocess.PIPE,
        text=True,
        shell=True,
    ).stdout
    str_resFromExtraP = str_resFromExtraP.replace("Model: ", "")
    str_resFromExtraP = convert_log(str_resFromExtraP)

    return str_resFromExtraP


def get_ExtraP_model(
    inputDF_perFunc: pd.DataFrame,
    expVar: list[str],
    resVar: str,
    functionName: str,
    dict_symbols: dict[any, any],
    benchmarkName: str,
    modelerName: str,
    modelerOption: str,
):
    """引数からExtra-Pによるモデルを構築し、それを返す関数

    Args:
        inputDF_perFunc(pd.DataFrame):入力DF
        expVar()
    """
    str_ExtraPinputData: str = gen_ExtraPinputDataFromDF(
        inputDF=inputDF_perFunc,
        expVar=expVar,
        resVar=resVar,
    )

    # Extra-Pへの入力ファイル作成
    converted_functionName: str = re.sub(
        r'[\\|/|:|?|.|"|<|>|\|\(|\)|]', "-", functionName
    )
    fileName: str = f"input_{benchmarkName}_{resVar}@{converted_functionName}.txt"
    fileDir: str = f"./extra-p_docker/share/"
    filePath: str = fileDir + fileName

    str_resFromExtraP: str = get_str_modelFromExtraP(
        filePath=filePath,
        modelerName=modelerName,
        modelerOption=modelerOption,
        extraPinputData=str_ExtraPinputData,
    )

    model_fromExtraP = sympify(str_resFromExtraP, locals=dict_symbols)
    return model_fromExtraP


# In[85]:


# def return_rawDF_mg(
#     list_process: list[int],
#     list_nit: list[int],
#     list_size: list[int],
#     csvDir: str,
# ) -> pd.DataFrame:

#     """return_rawDF_mg()

#     ベンチマークプログラムMGの手動で変更した初期変数におけるプロファイルを取得する関数

#     Args:
#         list_process(list[int]):プロセス数のリスト
#         list_nit(list[int]):初期変数nitのリスト
#         list_size(list[int]):初期変数sizeのリスト
#         csvDir(str):CSVファイルを格納したディレクトリのパスを表す文字列

#     Returns:
#         pd.DataFrame

#     """

#     list_before_concat_DF: list[pd.DataFrame] = []

#     for elem_process in list_process:
#         for elem_nit in list_nit:
#             for elem_size in list_size:
#                 filePath: str = f"{csvDir}mg_problem_size{elem_size}_nit{elem_nit}_process{elem_process}.csv"
#                 if os.path.isfile(filePath):
#                     try:
#                         DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
#                         DF_read_raw["process"] = elem_process
#                         DF_read_raw["size"] = elem_size
#                         DF_read_raw["nit"] = elem_nit
#                         list_before_concat_DF.append(DF_read_raw)
#                     except:
#                         warnings.warn(f"{filePath} is empty.")
#                 else:
#                     warnings.warn(f"{filePath} doesn't exist")
#     return pd.concat(objs=list_before_concat_DF, axis=0)


# In[86]:


def returnConvertedTargetPprofTimeDF(
    inputDF: pd.DataFrame,
    resVars: list[str],
):
    """
    学習用データおよび予測対象用データの任意のデータ列を変換する関数

    Args:
        inputDF(pd.DataFrame) : 入力DF
        resVars(list[str]) : 入力DFにおいて変換したい列名のリスト

    Returns:
        pd.DataFrame : 指定された列が変換されたデータフレーム
    """
    returnDF: pd.DataFrame = inputDF.copy(deep=True)

    for resVar in resVars:
        _tmp_series: pd.Series = returnDF[resVar]
        returnDF[resVar] = _tmp_series.map(convertPprofTime)

    return returnDF


def test_returnConvertedTargetPprofTimeDF():
    _tmp_dict: dict[str, list[str]] = {}

    _tmp_dict["col1"] = ["1:44.607", "33,350"]
    _tmp_dict["col2"] = ["111:111", "0:123"]
    inputDF: pd.DataFrame = pd.DataFrame(data=_tmp_dict)

    _tmp_dict["col1"] = [1 * 60 + 44 + 0.607, 33.350]
    _tmp_dict["col2"] = [111 * 60 + 111.0, 123.0]
    DF_expected: pd.DataFrame = pd.DataFrame(data=_tmp_dict)

    DF_actually: pd.DataFrame = returnConvertedTargetPprofTimeDF(
        inputDF=inputDF, resVars=["col1", "col2"]
    )

    assert DF_expected.equals(
        DF_actually
    ), f"expected=\n{DF_expected}\nactually=\n{DF_actually}"


def addPerCallCol(
    inputDF: pd.DataFrame,
    targetColNames: list[str],
    CallColName: str,
) -> pd.DataFrame:
    """
    PerCall列を追加して返す関数

    Args:
        inputDF(pd.DataFrame):入力DF
        targetColNames(list[str]):perCall列を作成したい列名のリスト
        CallColName(str):関数コール回数が保持された列名
    """

    _tmp_list = []
    for i, sr in inputDF.iterrows():
        dividend: float
        divisor: float
        for targetColName in targetColNames:
            divindend = sr[targetColName]
            divisor = sr[CallColName]
            sr[f"{targetColName}PerCall"] = divindend / divisor
        _tmp_list.append(sr)
    return pd.DataFrame(_tmp_list)


def test_addPerCallCol():
    _tmp_dict: dict[str, list[float]] = {}
    _tmp_dict["col1"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    _tmp_dict["col2"] = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    _tmp_dict["#Call"] = [2.0, 8.0, 38.0, 8.0, 8.0, 9.0, 10.0, 11.0]
    inputDF = pd.DataFrame(data=_tmp_dict)
    _tmp_dict["col1PerCall"] = []
    _tmp_dict["col2PerCall"] = []
    for i in range(len(_tmp_dict["#Call"])):
        _tmp_dict["col1PerCall"].append(_tmp_dict["col1"][i] / _tmp_dict["#Call"][i])
        _tmp_dict["col2PerCall"].append(_tmp_dict["col2"][i] / _tmp_dict["#Call"][i])

    DF_actually: pd.DataFrame = addPerCallCol(
        inputDF=inputDF,
        targetColNames=["col1", "col2"],
        CallColName="#Call",
    )
    DF_exptected: pd.DataFrame = pd.DataFrame(data=_tmp_dict)
    assert DF_exptected.equals(
        DF_actually
    ), f"expected=\n{DF_exptected}\nactually=\n{DF_actually}"


# In[87]:


def ret_averaged_rawDF_mg(
    list_process: list[int],
    list_nit: list[int],
    list_size: list[int],
    list_csvDir: list[str],
    resVar: str,
):
    """複数のCSVからDFを取得する関数

    引数resVarで指定された列がInclusiveもしくはExclusiveの場合はそれらが秒に変換され、InclusivePerCallもしくはExclusivePerCall列が生成されている

    Args:
        list_process(list[int]):プロセス数

    """
    list_DFs_for_return: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_nit in list_nit:
            for elem_size in list_size:
                list_inputDFs_for_averaged: list[pd.DataFrame] = []
                for elem_csvDir in list_csvDir:
                    _raw_DF: pd.DataFrame = return_rawDF_mg(
                        list_process=[elem_process],
                        list_nit=[elem_nit],
                        list_problem_size=[elem_size],
                        csvDir=elem_csvDir,
                    )

                    if resVar in ["Exclusive", "Inclusive", "#Call", "#Subrs"]:
                        # resVar 列の整形
                        if resVar in ["Exclusive", "Inclusive"]:
                            _tmp_converted = map(
                                convertPprofTime, list(_raw_DF[resVar])
                            )
                            _raw_DF[resVar] = list(_tmp_converted)
                        # {resVar}PerCall 列の生成
                        _raw_DF = add_perCallColumn(
                            inputDF=_raw_DF,
                            divisorColName="#Call",
                            dividendColName=resVar,
                            targetColumnName=f"{resVar}PerCall",
                        )

                    list_inputDFs_for_averaged.append(_raw_DF)
                list_DFs_for_return.append(
                    ret_averagedDF(inputDFs=list_inputDFs_for_averaged, resVar=resVar)
                )
    return pd.concat(objs=list_DFs_for_return, axis=0)


# In[88]:


def return_rawDF_ft(
    list_process: list[int],
    list_grid_size: list[int],
    list_nit: list[int],
    csvDir: str,
) -> pd.DataFrame:
    """return_rawDF_ft()

    ベンチマークプログラムFTの手動で変更した初期変数におけるプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_grid_size(list[int]):初期変数grid_sizeのリスト
        list_nit(list[int]):初期変数nitのリスト

    Returns:
        pd.DataFrame

    """

    list_before_concat_DF: list[pd.DataFrame] = []

    for elem_process in list_process:
        for elem_grid_size in list_grid_size:
            for elem_nit in list_nit:
                filePath: str = f"{csvDir}ft_grid_size{elem_grid_size}_nit{elem_nit}_process{elem_process}.csv"
                if os.path.isfile(filePath):
                    try:
                        DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                        DF_read_raw["process"] = elem_process
                        DF_read_raw["grid_size"] = elem_grid_size
                        DF_read_raw["nit"] = elem_nit
                        list_before_concat_DF.append(DF_read_raw)
                    except:
                        warnings.warn(f"{filePath} is empty.")
                else:
                    warnings.warn(f"{filePath} doesn't exist")
                    continue
    return pd.concat(objs=list_before_concat_DF, axis=0)


def ret_averaged_rawDF_ft(
    list_process: list[int],
    list_grid_size: list[int],
    list_nit: list[int],
    list_csvDir: list[str],
    resVar: str,
):
    """複数のCSVからDFを取得する関数（ベンチマークプログラムFT）

    引数resVarで指定された列がInclusiveもしくはExclusiveの場合はそれらが秒に変換され、InclusivePerCallもしくはExclusivePerCall列が生成されている

    Args:
        list_process(list[int]):プロセス数のリスト
        list_grid_size(list[int]):グリッドサイズのリスト
        list_nit(list[int]):イテレーション数のリスト
        list_csvDir(list[str]):CSVを保持したディレクトリ名のリスト
        resVar(str):説明変数の文字列

    """

    list_DFs_for_return: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_grid_size in list_grid_size:
            for elem_nit in list_nit:
                list_inputDFs_for_averaged: list[pd.DataFrame] = []
                for elem_csvDir in list_csvDir:
                    _raw_DF: pd.DataFrame = return_rawDF_ft(
                        list_process=[elem_process],
                        list_grid_size=[elem_grid_size],
                        list_nit=[elem_nit],
                        csvDir=elem_csvDir,
                    )

                    if resVar in ["Exclusive", "Inclusive", "#Call", "#Subrs"]:
                        # resVar 列の整形
                        if resVar in ["Exclusive", "Inclusive"]:
                            _tmp_converted = map(
                                convertPprofTime, list(_raw_DF[resVar])
                            )
                            _raw_DF[resVar] = list(_tmp_converted)
                        # {resVar}PerCall 列の生成
                        _raw_DF = add_perCallColumn(
                            inputDF=_raw_DF,
                            divisorColName="#Call",
                            dividendColName=resVar,
                            targetColumnName=f"{resVar}PerCall",
                        )

                    list_inputDFs_for_averaged.append(_raw_DF)
                list_DFs_for_return.append(
                    ret_averagedDF(inputDFs=list_inputDFs_for_averaged, resVar=resVar)
                )
    return pd.concat(objs=list_DFs_for_return, axis=0)


# In[89]:


def return_rawDF_ep(
    list_process: list[int],
    list_size: list[int],
    csvDir: str,
) -> pd.DataFrame:
    """return_rawDF_ep()

    ベンチマークプログラムEPの手動で変更した初期変数におけるプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_size(list[int]):初期変数sizeのリスト
        csvDir(str):CSVファイルの保持されているディレクトリ

    Returns:
        pd.DataFrame

    """

    list_before_concat_DF: list[pd.DataFrame] = []

    for elem_process in list_process:
        for elem_size in list_size:
            filePath: str = f"{csvDir}ep_size{elem_size}_process{elem_process}.csv"
            if os.path.isfile(filePath):
                try:
                    DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                    DF_read_raw["process"] = elem_process
                    DF_read_raw["size"] = elem_size
                    list_before_concat_DF.append(DF_read_raw)
                except:
                    warnings.warn(f"{filePath} is empty.")
            else:
                warnings.warn(f"{filePath} doesn't exist")
                continue
    return pd.concat(objs=list_before_concat_DF, axis=0)


def ret_averaged_rawDF_ep(
    list_process: list[int],
    list_size: list[int],
    list_csvDir: list[str],
    resVar: str,
):
    """複数のCSVからDFを取得する関数（ベンチマークプログラムEP）

    列Inclusiveおよび列Exclusiveが秒に変換され、InclusivePerCallもしくはExclusivePerCall列が生成される。

    Args:
        list_process(list[int]):プロセス数のリスト
        list_grid_size(list[int]):グリッドサイズのリスト
        list_nit(list[int]):イテレーション数のリスト
        list_csvDir(list[str]):CSVを保持したディレクトリ名のリスト
        resVar(str):説明変数の文字列

    """

    list_DFs_for_return: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_size in list_size:
            list_inputDFs_for_averaged: list[pd.DataFrame] = []
            for elem_csvDir in list_csvDir:
                try:
                    _raw_DF: pd.DataFrame = return_rawDF_ep(
                        list_process=[elem_process],
                        list_size=[elem_size],
                        csvDir=elem_csvDir,
                    )

                    if resVar in ["Exclusive", "Inclusive", "#Call", "#Subrs"]:
                        # resVar 列の整形
                        if resVar in ["Exclusive", "Inclusive"]:
                            _tmp_converted = map(
                                convertPprofTime, list(_raw_DF[resVar])
                            )
                            _raw_DF[resVar] = list(_tmp_converted)
                        # {resVar}PerCall 列の生成
                        _raw_DF = add_perCallColumn(
                            inputDF=_raw_DF,
                            divisorColName="#Call",
                            dividendColName=resVar,
                            targetColumnName=f"{resVar}PerCall",
                        )
                except:
                    continue

                list_inputDFs_for_averaged.append(_raw_DF)
            list_DFs_for_return.append(
                ret_averagedDF(inputDFs=list_inputDFs_for_averaged, resVar=resVar)
            )
    return pd.concat(objs=list_DFs_for_return, axis=0)


# In[ ]:


def return_rawDF_is(
    list_process: list[int],
    list_no_of_keys: list[int],
    list_key_max_value: list[int],
    csvDir: str,
) -> pd.DataFrame:
    """return_rawDF_is()

    ベンチマークプログラムISの手動で変更した初期変数におけるプロファイルを取得する関数

    Args:
        list_process(list[int]):プロセス数のリスト
        list_no_of_keys(list[int]):初期変数no_of_keysのリスト
        list_key_max_value(list[int]):初期変数key_max_valueのリスト
        csvDir(str):CSVファイルの保持されているディレクトリ

    Returns:
        pd.DataFrame

    """

    list_before_concat_DF: list[pd.DataFrame] = []

    for elem_process in list_process:
        for elem_no_of_keys in list_no_of_keys:
            for elem_key_max_value in list_key_max_value:
                filePath: str = f"{csvDir}is_no_of_keys{elem_no_of_keys}_key_max_value{elem_key_max_value}_process{elem_process}.csv"
                if os.path.isfile(filePath):
                    try:
                        DF_read_raw: pd.DataFrame = pd.read_csv(filePath)
                        DF_read_raw["process"] = elem_process
                        DF_read_raw["no_of_keys"] = elem_no_of_keys
                        DF_read_raw["key_max_value"] = elem_key_max_value
                        list_before_concat_DF.append(DF_read_raw)
                    except:
                        warnings.warn(f"{filePath} is empty.")
                else:
                    warnings.warn(f"{filePath} doesn't exist")
                    continue
    return pd.concat(objs=list_before_concat_DF, axis=0)


def ret_averaged_rawDF_is(
    list_process: list[int],
    list_no_of_keys: list[int],
    list_key_max_value: list[int],
    list_csvDir: list[str],
    resVar: str,
):
    """複数のCSVからDFを取得する関数（ベンチマークプログラムEP）

    列Inclusiveおよび列Exclusiveが秒に変換され、InclusivePerCallもしくはExclusivePerCall列が生成される。

    Args:
        list_process(list[int]):プロセス数のリスト
        list_no_of_keys(list[int]):初期変数no_of_keysのリスト
        list_key_max_value(list[int]):初期変数key_max_valueのリスト
        list_csvDir(list[str]):CSVを保持したディレクトリ名のリスト
        resVar(str):説明変数の文字列

    """

    list_DFs_for_return: list[pd.DataFrame] = []
    for elem_process in list_process:
        for elem_no_of_keys in list_no_of_keys:
            for elem_key_max_value in list_key_max_value:
                list_inputDFs_for_averaged: list[pd.DataFrame] = []
                for elem_csvDir in list_csvDir:
                    try:
                        _raw_DF: pd.DataFrame = return_rawDF_is(
                            list_process=[elem_process],
                            list_no_of_keys=[elem_no_of_keys],
                            list_key_max_value=[elem_key_max_value],
                            csvDir=elem_csvDir,
                        )

                        if resVar in ["Exclusive", "Inclusive", "#Call", "#Subrs"]:
                            # resVar 列の整形
                            if resVar in ["Exclusive", "Inclusive"]:
                                _tmp_converted = map(
                                    convertPprofTime, list(_raw_DF[resVar])
                                )
                                _raw_DF[resVar] = list(_tmp_converted)
                            # {resVar}PerCall 列の生成
                            _raw_DF = add_perCallColumn(
                                inputDF=_raw_DF,
                                divisorColName="#Call",
                                dividendColName=resVar,
                                targetColumnName=f"{resVar}PerCall",
                            )
                    except:
                        continue

                    list_inputDFs_for_averaged.append(_raw_DF)
                list_DFs_for_return.append(
                    ret_averagedDF(inputDFs=list_inputDFs_for_averaged, resVar=resVar)
                )
    return pd.concat(objs=list_DFs_for_return, axis=0)


# In[ ]:


def ret_query(list_expVar: list[str], list_product: list[int]) -> str:
    retStr: str = ""
    for i_expVar in range(len(list_expVar)):
        retStr += f"{list_expVar[i_expVar]}=={list_product[i_expVar]}"
        if i_expVar != len(list_expVar) - 1:
            retStr += " & "
    return retStr


class ContentsForExtraP:
    """class ContentsForExtraP

    ExtraPによるモデルの生成

    生成されるモデルは次の通り。

    * 関数コール回数の予測モデル
    * 関数の総実行時間の予測モデル
    * 1コール当たりの関数の総実行時間の予測モデル

    """

    def __init__(
        self,
        trainDF: pd.DataFrame,
        testDF: pd.DataFrame,
        expVars: list[str],
        resVar: str,
        resVarPerCall: str,
        benchmarkName: str,
        modelerName: str = "multi-parameter",
        modelerOption: str = """ --options \#spm=Basic \#spo=poly_exponents=-1,0,1,2,3,log_exponents=0,1,force_combination_exponents=1,allow_negative_exponents=True""",
    ):
        """__init__()

        初期化変数

        Args:
            trainDF(pd.DataFrame):モデル構築用データ
            testDF(pd.DataFrame):予測対象用データ
            expVars(list[str]):説明変数を格納したリスト
            resVar(str):目的変数を示した文字列
            resVarsPerCall(str):1コール当たりの目的変数を示した文字列

        """
        self.trainDF = trainDF.reset_index()
        self.testDF = testDF.reset_index()
        self.expVars = expVars
        self.resVar = resVar
        self.resVarPerCall = resVarPerCall
        self.functionNames: list[str] = sorted(list(set(trainDF["Name"].to_list())))
        self.benchmarkName: str = benchmarkName
        self.modelerName: str = modelerName
        self.modelerOption: str = modelerOption
        self.dict_symbols: dict[str, str] = {}
        for elem in expVars:
            self.dict_symbols[elem] = symbols(elem, real=True)

        self.dict_resVar = {}
        self.dict_resVarPerCall = {}
        self.dict_call = {}

    def build_all_models(self):
        """build_all_models()

        モデルを構築する関数。目的変数はself.resVars, self.resVarsPerCallを利用。

        """

        for functionName in self.functionNames:
            trainDF_perFunc: pd.DataFrame = self.trainDF[
                self.trainDF["Name"] == functionName
            ]

            model_resVar: str = get_ExtraP_model(
                inputDF_perFunc=trainDF_perFunc,
                expVar=self.expVars,
                resVar=self.resVar,
                functionName=functionName,
                dict_symbols=self.dict_symbols,
                benchmarkName=self.benchmarkName,
                modelerName=self.modelerName,
                modelerOption=self.modelerOption,
            )

            model_resVarPerCall: str = get_ExtraP_model(
                inputDF_perFunc=trainDF_perFunc,
                expVar=self.expVars,
                resVar=self.resVarPerCall,
                functionName=functionName,
                dict_symbols=self.dict_symbols,
                benchmarkName=self.benchmarkName,
                modelerName=self.modelerName,
                modelerOption=self.modelerOption,
            )

            model_call: str = get_ExtraP_model(
                inputDF_perFunc=trainDF_perFunc,
                expVar=self.expVars,
                resVar="#Call",
                functionName=functionName,
                dict_symbols=self.dict_symbols,
                benchmarkName=self.benchmarkName,
                modelerName=self.modelerName,
                modelerOption=self.modelerOption,
            )

            self.dict_resVar[functionName] = model_resVar
            self.dict_resVarPerCall[functionName] = model_resVarPerCall
            self.dict_call[functionName] = model_call

    def predict_train(self):
        """predict_train(self)

        学習データに対して予測を実施する関数

        列構成は下記の通り

        |<expVar>|Name(関数名)|#Call(コール回数実測値)|predicted_resVar（）|predicted_ExclusivePerCall	predicted_#Call>|<>|

        """

        _list_series: list[pd.Series] = []
        for index, _sr in self.trainDF.iterrows():
            _series: pd.Series = pd.Series(dtype="object")
            functionName: str = _sr["Name"]

            _target_env: list[set[any]] = []
            for _expVar in self.expVars:
                _series[_expVar] = _sr[_expVar]
                _target_env.append((self.dict_symbols[_expVar], _sr[_expVar]))

            _series["Name"] = _sr["Name"]
            _series["#Call"] = _sr["#Call"]
            _series[f"{self.resVar}"] = _sr[f"{self.resVar}"]

            _predicted_resVar: float
            _predicted_resVarPerCall: float
            _predicted_call: float

            _predicted_resVar = self.dict_resVar[functionName].subs(_target_env).evalf()
            _predicted_resVarPerCall = (
                self.dict_resVarPerCall[functionName].subs(_target_env).evalf()
            )
            _predicted_call = self.dict_call[functionName].subs(_target_env).evalf()

            _series[f"predicted_{self.resVar}"] = _predicted_resVar
            _series[f"predicted_{self.resVarPerCall}"] = _predicted_resVarPerCall
            _series[f"predicted_#Call"] = _predicted_call
            _series[f"predicted_{self.resVar}_indirectly"] = (
                _predicted_resVarPerCall * _predicted_call
            )

            _list_series.append(_series)

        self.DF_predicted_by_train: pd.DataFrame = pd.DataFrame(
            data=_list_series
        ).astype(
            {
                f"predicted_{self.resVar}": float,
                f"predicted_{self.resVarPerCall}": float,
                f"predicted_#Call": float,
                f"predicted_{self.resVar}_indirectly": float,
            }
        )

    def predict_test(self):
        """predict_test(self)

        予測対象データに対して予測を実施する関数
        """

        _list_series: list[pd.Series] = []

        for index, _sr in self.testDF.iterrows():
            _series: pd.Series = pd.Series(dtype="object")
            functionName: str = _sr["Name"]

            _target_env: list[set[any]] = []
            for _expVar in self.expVars:
                _series[_expVar] = _sr[_expVar]
                _target_env.append((self.dict_symbols[_expVar], _sr[_expVar]))

            _series["Name"] = functionName
            _series["#Call"] = _sr["#Call"]
            _series[f"{self.resVar}"] = _sr[f"{self.resVar}"]

            _predicted_resVar: float = (
                self.dict_resVar[functionName].subs(_target_env).evalf()
            )
            _predicted_resVarPerCall: float = (
                self.dict_resVarPerCall[functionName].subs(_target_env).evalf()
            )
            _predicted_call: float = (
                self.dict_call[functionName].subs(_target_env).evalf()
            )

            _series[f"predicted_{self.resVar}"] = _predicted_resVar
            _series[f"predicted_{self.resVarPerCall}"] = _predicted_resVarPerCall
            _series[f"predicted_#Call"] = _predicted_call
            _series[f"predicted_{self.resVar}_indirectly"] = (
                _predicted_resVarPerCall * _predicted_call
            )

            _list_series.append(_series)

        self.DF_predicted_by_test: pd.DataFrame = pd.DataFrame(data=_list_series)

        self.DF_predicted_by_test: pd.DataFrame = add_relativeErrorRateCol(
            inputDF=self.DF_predicted_by_test,
            real_colName="#Call",
            predicted_colName="predicted_#Call",
            targetColName="APE_#Call",
        )
        self.DF_predicted_by_test = add_relativeErrorRateCol(
            inputDF=self.DF_predicted_by_test,
            real_colName=f"{self.resVar}",
            predicted_colName=f"predicted_{self.resVar}",
            targetColName=f"APE_{self.resVar}",
        )
        self.DF_predicted_by_test = add_relativeErrorRateCol(
            inputDF=self.DF_predicted_by_test,
            real_colName=f"{self.resVar}",
            predicted_colName=f"predicted_{self.resVar}_indirectly",
            targetColName=f"APE_{self.resVarPerCall}",
        )

    def get_result_of_predict_train(self):
        return self.DF_predicted_by_train

    def get_result_of_predict_test(self):
        return self.DF_predicted_by_test

    def get_DF_all_fitness_with_relativeErrorRate(self) -> pd.DataFrame:
        """get_DF_all_fitness_with_relativeErrorRate()

        モデル適合度のテーブルを返す関数

        列APE～の平均をとるとMAPEになる。

        """

        _returnDF: pd.DataFrame = add_relativeErrorRateCol(
            inputDF=self.DF_predicted_by_train,
            real_colName="#Call",
            predicted_colName="predicted_#Call",
            targetColName="APE_#Call",
        )
        _returnDF = add_relativeErrorRateCol(
            inputDF=_returnDF,
            real_colName=f"{self.resVar}",
            predicted_colName=f"predicted_{self.resVar}",
            targetColName=f"APE_{self.resVar}",
        )
        _returnDF = add_relativeErrorRateCol(
            inputDF=_returnDF,
            real_colName=f"{self.resVar}",
            predicted_colName=f"predicted_{self.resVar}_indirectly",
            targetColName=f"APE_{self.resVarPerCall}",
        )

        return _returnDF

    def return_products(self):
        """return_products()

        予測対象の組合せを返す関数

        """
        _dict_expVar_set: dict[str, set[int]] = {}
        for _expVar in self.expVars:
            _dict_expVar_set[_expVar] = set(sorted(self.testDF[_expVar]))

        if len(self.expVars) == 2:
            _product = itertools.product(
                _dict_expVar_set[self.expVars[0]], _dict_expVar_set[self.expVars[1]]
            )
        elif len(self.expVars) == 3:
            _product = itertools.product(
                _dict_expVar_set[self.expVars[0]],
                _dict_expVar_set[self.expVars[1]],
                _dict_expVar_set[self.expVars[2]],
            )
        else:
            warnings.warn(f"self.expVars == {len(self.expVars)}")
            return -1

        return _product

    def get_DF_weightedMAPE(self) -> pd.DataFrame:
        """get_DF_all_fitness_with_weightedMAPE()

        予測対象環境における重み付きMAPEをまとめたDFを返す関数

        """

        _product = self.return_products()

        _list_series: pd.Series = []
        for _elem_product in _product:
            _query: str = ret_query(
                list_expVar=self.expVars, list_product=list(_elem_product)
            )

            _targetDF: pd.DataFrame = self.testDF.query(_query)
            _target_env: list[set[any]] = []

            _series: pd.Series = pd.Series(dtype="object")

            for i_expVar in range(len(self.expVars)):
                _target_env.append(
                    (self.dict_symbols[self.expVars[i_expVar]], _elem_product[i_expVar])
                )
                _series[self.expVars[i_expVar]] = _elem_product[i_expVar]

            _c_sum: float = sum(_targetDF["#Call"].tolist())
            _target_weightedMAPE_resVar: float = 0.0
            _target_weightedMAPE_resVarPerCall: float = 0.0
            _target_weightedMAPE_call: float = 0.0
            for i, sr in _targetDF.iterrows():
                _c_t: float = sr["#Call"]
                functionName: str = sr["Name"]

                _A_t_resVar: float = sr[self.resVar]
                _A_t_resVarPerCall: float = sr[self.resVarPerCall]
                _A_t_call: float = sr["#Call"]

                _F_t_resVar: float = (
                    self.dict_resVar[functionName].subs(_target_env).evalf()
                )
                _F_t_resVarPerCall: float = (
                    self.dict_resVarPerCall[functionName].subs(_target_env).evalf()
                )
                _F_t_call: float = (
                    self.dict_call[functionName].subs(_target_env).evalf()
                )

                if _A_t_resVar == 0.0:
                    warnings.warn(f"{sr['Name'] = }, {sr[self.resVar] = }")
                    continue
                elif _A_t_resVarPerCall == 0.0:
                    warnings.warn(f"{sr['Name'] = }, {sr[self.resVarPerCall] = }")
                    continue

                _target_weightedMAPE_resVar += (
                    abs(_A_t_resVar - _F_t_resVar) / _A_t_resVar
                )
                _target_weightedMAPE_resVarPerCall += (
                    abs(_A_t_resVarPerCall - _F_t_resVarPerCall) / _A_t_resVarPerCall
                )
                _target_weightedMAPE_call += abs(_A_t_call - _F_t_call) / _A_t_call

            _target_weightedMAPE_resVar *= 100 / _c_sum
            _target_weightedMAPE_resVarPerCall *= 100 / _c_sum
            _target_weightedMAPE_call *= 100 / _c_sum

            _series[f"weightedMAPE_{self.resVar}"] = _target_weightedMAPE_resVar
            _series[
                f"weightedMAPE_{self.resVarPerCall}"
            ] = _target_weightedMAPE_resVarPerCall
            _series[f"weightedMAPE_#Call"] = _target_weightedMAPE_call

            _list_series.append(_series)
        self.DF_weightedMAPE: pd.DataFrame = pd.DataFrame(data=_list_series)
        # self.DF_weightedMAPE :pd.DataFrame = pd.DataFrame(data=_list_series).astype({f"weightedMAPE_{self.resVar}": float, f"weightedMAPE_{self.resVarPerCall}": float, f"weightedMAPE_call": float})
        return self.DF_weightedMAPE

    def get_DF_MAPE(self) -> pd.DataFrame:
        """get_DF_MAPE()

        予測対象環境におけるMAPEをまとめたDFを返す関数

        """

        _product = self.return_products()

        _list_series: pd.Series = []
        for _elem_product in _product:
            _query: str = ret_query(
                list_expVar=self.expVars, list_product=list(_elem_product)
            )

            _targetDF: pd.DataFrame = self.get_result_of_predict_test().query(_query)

            _targetDF: pd.DataFrame = add_relativeErrorRateCol(
                inputDF=_targetDF,
                real_colName="#Call",
                predicted_colName="predicted_#Call",
                targetColName="APE_#Call",
            )
            _targetDF = add_relativeErrorRateCol(
                inputDF=_targetDF,
                real_colName=f"{self.resVar}",
                predicted_colName=f"predicted_{self.resVar}",
                targetColName=f"APE_{self.resVar}",
            )
            _targetDF = add_relativeErrorRateCol(
                inputDF=_targetDF,
                real_colName=f"{self.resVar}",
                predicted_colName=f"predicted_{self.resVar}_indirectly",
                targetColName=f"APE_{self.resVarPerCall}",
            )

            _series: pd.Series = pd.Series(dtype="object")

            for i_expVar in range(len(self.expVars)):
                _series[self.expVars[i_expVar]] = _elem_product[i_expVar]
            _series[f"MAPE_{self.resVar}"] = np.mean(
                _targetDF[f"APE_{self.resVar}"].to_list()
            )
            _series[f"MAPE_{self.resVarPerCall}"] = np.mean(
                _targetDF[f"APE_{self.resVarPerCall}"].to_list()
            )
            _series[f"MAPE_#Call"] = np.mean(_targetDF[f"APE_#Call"].to_list())

            _list_series.append(_series)

        return pd.DataFrame(data=_list_series)

    def exec_all(self) -> dict[str, pd.DataFrame]:
        """exec_all()

        個別で実施していた
        1. モデルの構築
        2. 学習データへの予測
        3. 予測対象データへの予測
        4. 各種表への整形
        一貫して行う関数

        """

        self.build_all_models()
        self.predict_train()
        self.predict_test()

        _適合度: pd.DataFrame = self.get_DF_all_fitness_with_relativeErrorRate()
        _重み付きMAPE: pd.DataFrame = self.get_DF_weightedMAPE()
        _MAPE: pd.DataFrame = self.get_DF_MAPE()
        _予測精度: pd.DataFrame = self.get_result_of_predict_test()
        _関数ごとのMAPE: pd.DataFrame = return_summarizedDF_on_functionName(
            inputDF=_適合度, resVar=self.resVar
        )

        returnDict: dict[str, pd.DataFrame] = {
            "適合度": _適合度,
            "重み付きMAPE": _重み付きMAPE,
            "MAPE": _MAPE,
            "予測精度": _予測精度,
            "関数ごとのMAPE": _関数ごとのMAPE,
        }

        return returnDict


def return_summarizedDF_on_functionName(
    inputDF: pd.DataFrame,
    resVar: str,
) -> pd.DataFrame:
    """return_summarizedDF_on_functionName()

    入力されたDFを関数ごとにまとめ、各数値を平均値化し、そのDFを返す関数

    Args:
        inputDF(pd.DataFrame): 入力DF
        resVar(string): 目的変数名

    """

    functionNames: list[str] = list(set(inputDF["Name"].to_list()))
    _list_series: list[pd.Series] = []
    for functionName in functionNames:
        inputDF_perFunc: pd.DataFrame = inputDF[inputDF["Name"] == functionName]
        list_APE_resVar: list[float] = inputDF_perFunc[f"APE_{resVar}"].to_list()
        list_APE_resVarPerCall: list[float] = inputDF_perFunc[
            f"APE_{resVar}PerCall"
        ].to_list()
        list_APE_call: list[float] = inputDF_perFunc[f"APE_#Call"].to_list()

        float_MAPE_resVar: float = np.mean(list_APE_resVar)
        float_MAPE_resVarPerCall: float = np.mean(list_APE_resVarPerCall)
        float_MAPE_call: float = np.mean(list_APE_call)

        _series: pd.Series = pd.Series(dtype="object")
        _series["Name"] = functionName
        _series[f"MAPE_#Call"] = float_MAPE_call
        _series[f"MAPE_{resVar}"] = float_MAPE_resVar
        _series[f"MAPE_{resVar}PerCall"] = float_MAPE_resVarPerCall

        _list_series.append(_series)

    return pd.DataFrame(data=_list_series)
