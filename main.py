# pandasのインポート
import pandas as pd 

# numpyのインポート
import numpy as np

# matplotlibのインポート
from matplotlib import pyplot as plt 

# seabornのインポート
import seaborn  as sns

# 線形回帰のインポート
from sklearn.linear_model import LinearRegression

# MSEのインポート
from sklearn.metrics import mean_squared_error

# XGBoostのライブラリのインポート
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, KFold

def RMSE(var1, var2):
    
    # まずMSEを計算
    mse = mean_squared_error(var1, var2)
    
    # 平方根を取った値を返す
    return np.sqrt(mse)


train = pd.read_csv("data/train.tsv", sep='\t')
# train = pd.get_dummies(train, columns=["weekday"],drop_first = True)
train = pd.get_dummies(train, columns=["weathersit"],drop_first = True)
train["hr_sin"] = np.sin(2 * np.pi * train["hr"]/24)
train["hr_cos"] = np.cos(2 * np.pi * train["hr"]/24)
# drop_index = train.index[(train['mnth'] == 4) & (train['holiday'] == 1)]
# train = train.drop(drop_index)
# print( train.head(25) )
# train = train.drop(columns=['dteday','yr'])
# print(train.corr())

## hold-out
# train,test = train_test_split(train, random_state = 0, test_size = 0.2)
test = train.iloc[8500:, :]
train = train.iloc[2000:, :]
features = ["hr_sin","hr_cos","workingday","temp","weathersit_2","weathersit_3","weathersit_4","hum"]
train_X = train[features]
train_y = train["cnt"]
test_X = test[features]
test_y = test["cnt"]
# print(test_X.info())

# lm = LinearRegression()
# lm.fit(train_X, train_y)
# pred1 = lm.predict( test_X )

# xgboostモデルの作成
model = xgb.XGBRegressor()
# ハイパーパラメータ探索
param = {
    'colsample_bytree': [0.9],
    'eta': [0.08], 
    'gamma': [0], 
    'max_depth':[6], 
    'min_child_weight': [7],
    'subsample': [0.9]
}
cv_shuffle = KFold(n_splits=5, shuffle=True, random_state = 0)
reg = GridSearchCV(estimator=model, param_grid=param, cv=cv_shuffle, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
reg.fit(train_X, train_y)
print(reg.best_params_, reg.best_score_)
pred1 = reg.predict(test_X)

# pred1=model.predict(train_X)
# pred1 = model.predict( test_X )

# RMSEの計算
var = RMSE( test_y, pred1 )
print(var)

plt.plot(test_y.values, label="actual")
plt.plot(pred1, label="forecast")
plt.xlabel("time step")
plt.ylabel("count")
plt.legend()
# plt.show()


# df_test = pd.read_csv("data/test.tsv", sep='\t')
# df_test= pd.get_dummies(df_test, columns=["weathersit"],drop_first = True)
# df_test["hr_sin"] = np.sin(2 * np.pi * df_test["hr"]/24)
# df_test["hr_cos"] = np.cos(2 * np.pi * df_test["hr"]/24)
# df_test_X = df_test[features]
# pred = model.predict(df_test_X)
# df_test['cnt'] = pred
# df_test[['id', 'cnt']].to_csv('data/submit.csv', header=False,index=False)


# plt.figure(figsize=(10,8))
# sns.heatmap(train.corr(), vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
# plt.show()

# 折れ線グラフの描画
# train['cnt'].plot( title="bycycle users" )
# # x軸とy軸にラベルを付けて表示します
# plt.xlabel("time step")
# plt.ylabel("users")
# plt.show()

# 散布図の描画
# train.plot.scatter( x='weathersit', y='cnt', c="blue", title="scatter plot of temperature and sales" )
# # x軸とy軸にラベルを付けます
# plt.xlabel("temperature")
# plt.ylabel("sales")
# # グラフの表示
# plt.show()

# print( train[["cnt", "temp", "atemp", "hum", "windspeed"]].corr() )

# グラフのサイズを指定して、折れ線グラフの描画
# plt.figure(figsize=(10,6))
# train = train[(train["holiday"] == 1) & (train["weekday"] == 5)]
# sns.lineplot( x='hr', y='cnt', hue='weekday', data=train )
# # x軸・y軸にラベルを付けます
# plt.xlabel("time step")
# plt.ylabel("sales")
# # グラフの表示
# plt.show()

# 箱ひげ図の描画（表示順序をorderで指定している）
# train = train[train["holiday"] == 1]
# sns.boxplot( x="weekday", y="cnt", data=train, order=["0","1","2","3","4","5","6"] )
# # y軸にラベルを付けます
# plt.ylabel("sales")
# # グラフの表示
# plt.show()
