import pandas as pd
import config

# データの読み込みとカラムの取り出し
df = pd.read_csv(r'C:\Users\nyugo\team-a-2024-summer-08-26\root\Backend\data\0829_train.csv', encoding='utf-8')
print("DataFrame columns:", df.columns)
y = df[config.TARGET]
X = df[config.COLUMNS]
col_names = X.columns.tolist()  # カラム名のリストを取得
print("Column names for the form:", col_names)