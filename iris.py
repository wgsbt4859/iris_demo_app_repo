'''
あやめの花の分類予測を行うWEBアプリを作成する.
分類はロジスティック回帰を用いる.
入力変数:
出力: あやめの分類結果
'''

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# データセットの読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値をデータフレームに追加
df["target"] = iris.target

# 目標値の数字を花名へ変換
df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# 予測モデル構築
# 今回はsepal lengthおよびpetal lengthの2変数から予測する.
x = iris.data[:, [0, 2]]
y = iris.target

# ロジスティック回帰モデルのインスタンス化と学習
clf = LogisticRegression()
clf.fit(x, y)


########################## 以下、streamlitの実装##########################

# サイドバー (入力変数設定)

with st.sidebar:
    st.header("Input Features")
    sepal_value = st.slider("sepal length (cm)",
                            min_value=0.0, max_value=10.0, step=0.1)
    petal_value = st.slider("petal length (cm)",
                            min_value=0.0, max_value=10.0, step=0.1)

# メイン画面

st.title("アヤメの分類 モックアプリ")

img = Image.open("iris.png")
st.image(img, caption="Iris", use_column_width=True)


st.write("## Input values")

# インプットデータの配列化 (予測用)
input_values = [[sepal_value, petal_value]]

# インプットデータのデータフレーム化 (表示用)
df = pd.DataFrame(input_values, index=["Input values"], columns=[
                  "sepal length (cm)", "petal length (cm)"])

# 入力値をテーブル形式で表示
st.table(df)

# 予測結果の出力
pred_probs = clf.predict_proba(input_values)

# 予測結果 (各クラスの確率)
pred_df = pd.DataFrame(pred_probs, index=["Probability"], columns=[
                       "setosa", "versicolor", "virginica"])

st.write("## Result")

# 確率表
st.dataframe(pred_df.style.highlight_max(axis=1))

# 最終結果の出力
name = pred_df.idxmax(axis=1).values
result_prob = float(pred_df[name].values)

st.write(f"## {result_prob:.1%}の確率で{name[0]}です.")
