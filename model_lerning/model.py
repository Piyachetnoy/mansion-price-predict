#このコードはgooglecolabで作成したものをそのままもってきたものです
# -*- coding: utf-8 -*-
"""yoshitomi_model_rent_ver2.ipynb のコピー

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cQwcTGjBzMHaTrXqAx9DIXQQS9u7tdmp

#実装

##ライブラリ
"""

from google.colab import drive
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error,  r2_score

import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

"""##機械学習モデルの作成"""

# Google Driveをマウント
drive.mount('/content/drive')

# ファイルパスを指定してCSVファイルを読み込む
file_path = '#データファイル'
train= pd.read_csv(file_path)

# データの表示
print(train.head())
train.shape

train

features = ['築年数', '階数', '階', '間取り_label', '部屋数', 'LDK', 'S','23区_label','最寄駅_label']

target = '賃料_管理費合計'

X = train[features]
y = train[target]

print(np.shape(X))
print(np.shape(y))

#標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

"""###線形回帰モデル"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 線形回帰モデル
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_train_pred = lr.predict(X_train)
lr_valid_pred = lr.predict(X_test)

# 評価指標の計算
mae_train = mean_absolute_error(y_train, lr_train_pred)
r2_train = r2_score(y_train, lr_train_pred)


mae_valid = mean_absolute_error(y_test, lr_valid_pred)
r2_valid = r2_score(y_test, lr_valid_pred)

# ベースラインモデルの評価 (平均値モデル)
baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)


# 精度
percentage = 100 * (1 - mae_valid / baseline_mae)


# 結果の表示
print(f"Train MAE: {mae_train}")
print(f"Train R^2: {r2_train}")


print(f"Valid MAE: {mae_valid}")
print(f"Valid R^2: {r2_valid}")
print(f'Performance Percentage: {percentage:.2f}%')

"""###GBDT(xgboost)"""

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# XGBoostの回帰モデル
xgb = XGBRegressor(random_state=0)
xgb.fit(X_train, y_train)

# 予測
xgb_train_pred = xgb.predict(X_train)
xgb_test_pred = xgb.predict(X_test)

# 評価指標の計算
mae_train = mean_absolute_error(y_train, xgb_train_pred)
r2_train = r2_score(y_train, xgb_train_pred)


mae_test = mean_absolute_error(y_test, xgb_test_pred)
r2_test = r2_score(y_test, xgb_test_pred)


# ベースラインモデルの評価 (平均値モデル)
baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)


# 精度
percentage = 100 * (1 - mae_valid / baseline_mae)

# 結果の表示
print(f"Train MAE: {mae_train}")
print(f"Train R^2: {r2_train}")

print(f"Test MAE: {mae_test}")
print(f"Test R^2: {r2_test}")
print(f'Performance Percentage: {percentage:.2f}%')

"""線形回帰モデルとGBDTのアンサンブル"""
train_pred = (lr_train_pred + xgb_train_pred) / 2
valid_pred = (lr_valid_pred + xgb_test_pred) / 2

# 評価指標の計算
mae_train = mean_absolute_error(y_train, train_pred)
r2_train = r2_score(y_train, train_pred)

mae_valid = mean_absolute_error(y_test, valid_pred)
r2_valid = r2_score(y_test, valid_pred)

# ベースラインモデルの評価 (平均値モデル)
baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)


# 精度
percentage = 100 * (1 - mae_valid / baseline_mae)

# 結果の表示
print(f"Train MAE: {mae_train}")
print(f"Train R^2: {r2_train}")

print(f"Valid MAE: {mae_valid}")
print(f"Valid R^2: {r2_valid}")
print(f'Performance Percentage: {percentage:.2f}%')

"""###LightGBM"""

# LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBMのハイパーパラメータ
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# モデルのトレーニング
model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data])

# 予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
print(model.best_iteration)


# モデルの評価
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# モデルの評価
model_mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ベースラインモデルの評価 (平均値モデル)
baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)


# 精度
percentage = 100 * (1 - model_mae / baseline_mae)
print(f'Model MAE: {model_mae}')
print(f'Baseline MAE: {baseline_mae}')
print(f'Performance Percentage: {percentage:.2f}%')
print(f'R² (Coefficient of Determination): {r2:.4f}')

"""#家賃推定テスト

"""

new_data = pd.DataFrame({
    '築年数': [13],
    '階数': [5],
    '階': [2],
    '間取り': ['1LDK'],
    '23区':['葛飾区'],
    '最寄駅':['綾瀬駅']
})

le = LabelEncoder()
new_data['間取り_label'] = le.fit_transform(new_data['間取り'])
new_data['23区_label'] = le.fit_transform(new_data['23区'])
new_data['最寄駅_label'] = le.fit_transform(new_data['最寄駅'])


new_data['部屋数'] = new_data['間取り'].apply(lambda x: 1 if 'ワンルーム' in x else int(x[0]) if x[0].isdigit() else 0)
new_data['LDK'] = new_data['間取り'].apply(lambda x: sum([x.count(c) for c in 'LDK']))
new_data['S'] = new_data['間取り'].apply(lambda x: 1 if 'S' in x else 0)

# 標準化
X_new = new_data[features]
X_new_scaled = scaler.transform(X_new)


#賃料の推定
y_pred = model.predict(X_new_scaled, num_iteration=model.best_iteration)
print(X_new)
print(X_new_scaled)

print('推定賃料:', y_pred[0])

"""#補足"""

"""['間取り']のデータ処理"""

# selected_data['間取り_label'] = LabelEncoder().fit_transform(selected_data['間取り'])

# # 部屋数、LDK数、Sの有無を特徴量として追加
# # ワンルームは部屋数を1に設定
# selected_data['部屋数'] = selected_data['間取り'].apply(lambda x: 1 if 'ワンルーム' in x else int(x[0]) if x[0].isdigit() else 0)
# selected_data['LDK'] = selected_data['間取り'].apply(lambda x: sum([x.count(c) for c in 'LDK']))
# selected_data['S'] = selected_data['間取り'].apply(lambda x: 1 if 'S' in x else 0)

"""['23区']のデータ処理"""
# le = LabelEncoder()
# selected_data['23区_label'] = le.fit_transform(selected_data['23区'])

"""['賃料']['管理費']のデータ処理"""
#selected_data['賃料_管理費合計'] = selected_data['賃料'] + selected_data['管理費']

import joblib

joblib.dump(model, "/content/drive/MyDrive/lgb_model3.pkl", compress=True)