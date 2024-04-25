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

# ランダムフォレストのインポート
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

def RMSE(var1, var2):
    
    # まずMSEを計算
    mse = mean_squared_error(var1, var2)
    
    # 平方根を取った値を返す
    return np.sqrt(mse)


train = pd.read_csv("data/train.tsv", sep='\t')
# train = pd.get_dummies(train, columns=["weekday"],drop_first = True)
train = pd.get_dummies(train, columns=["weathersit"],drop_first = True)
# train = pd.get_dummies(train, columns=["season"],drop_first = True)
train["hr_sin"] = np.sin(2 * np.pi * train["hr"]/24)
train["hr_cos"] = np.cos(2 * np.pi * train["hr"]/24)
train["btemp"] = train["temp"] + train["atemp"]
drop_index = train.index[(train['mnth'] == 4) & (train['holiday'] == 1)]
train = train.drop(drop_index)
train["triple"] = [1 if x and y else 0 for x,y in zip(train["weekday"] == 1,train["holiday"] == 1)]
# print( train.head(25) )
# train = train.drop(columns=['dteday','yr'])
# print(train.corr())

## hold-out
test = train.iloc[:2000, :]
train = train.iloc[3000:, :]
train,valid = train_test_split(train, random_state = 0, test_size = 0.2)
# train,test = train_test_split(train, random_state = 0, test_size = 0.1)
features = ["hr_sin","hr_cos","workingday","weathersit_2","weathersit_3","weathersit_4","hum","btemp","triple"]
# features = ["hr_sin","hr_cos","workingday","temp","weathersit_2","weathersit_3","weathersit_4","hum","season_2","season_3","season_4"]
train_X = train[features]
train_y = train["cnt"]
valid_X = train[features]
valid_y = train["cnt"]
test_X = test[features]
test_y = test["cnt"]
# print(test_X.info())

# lm = LinearRegression()
# lm.fit(train_X, train_y)
# pred1 = lm.predict( test_X )

# xgboostモデルの作成
model_1 = xgb.XGBRegressor()
# ハイパーパラメータ探索
param = {
    'colsample_bytree': [0.9],
    'eta': [0.13], 
    'gamma': [5], 
    'max_depth':[6], 
    'min_child_weight': [7],
    'subsample': [0.9]
}
# cv_shuffle = KFold(n_splits=5, shuffle=True, random_state = 0)
# model_1 = GridSearchCV(estimator=model_1, param_grid=param, cv=cv_shuffle, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
model_1.fit(train_X, train_y)
# print(model_1.best_params_, model_1.best_score_)
base_pred_1 = model_1.predict(train_X)

# アンサンブル用のモデルのインスタンス
model_2 = LinearRegression()
model_3 = RandomForestRegressor()

model_2.fit(train_X, train_y)
model_3.fit(train_X, train_y)

base_pred_2 = model_2.predict(valid_X)
base_pred_3 = model_3.predict(valid_X)

stacked_predictions = np.column_stack((base_pred_1, base_pred_2, base_pred_3))

# train meta model 
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, valid_y)

# ベースモデルで予測
# pred1 = reg.predict(test_X)
pred1 = model_1.predict(test_X)
pred2 = model_2.predict(test_X)
pred3 = model_3.predict(test_X)

test_stacked_predictions = np.column_stack((pred1, pred2, pred3))
pred4 = meta_model.predict(test_stacked_predictions)

# pred1 = model.predict( test_X )

# RMSEの計算
var1 = RMSE( test_y, pred1 )
var2 = RMSE( test_y, pred2 )
var3 = RMSE( test_y, pred3 )
var4 = RMSE( test_y, pred4 )
print(var1)
print(var2)
print(var3)
print(var4)

plt.plot(test_y.values, label="actual")
plt.plot(pred1, label="forecast")
plt.xlabel("time step")
plt.ylabel("count")
plt.legend()
# plt.show()


df_test = pd.read_csv("data/test.tsv", sep='\t')
df_test= pd.get_dummies(df_test, columns=["weathersit"],drop_first = True)
df_test["hr_sin"] = np.sin(2 * np.pi * df_test["hr"]/24)
df_test["hr_cos"] = np.cos(2 * np.pi * df_test["hr"]/24)
df_test["btemp"] = df_test["temp"] + df_test["atemp"]
df_test["triple"] = [1 if x and y else 0 for x,y in zip(df_test["weekday"] == 1,df_test["holiday"] == 1)]
df_test_X = df_test[features]
# pred = model_1.predict(df_test_X)
# pred = reg.predict(df_test_X)
pred1 = model_1.predict(df_test_X)
pred2 = model_2.predict(df_test_X)
pred3 = model_3.predict(df_test_X)
test_stacked_predictions = np.column_stack((pred1, pred2, pred3))
pred = meta_model.predict(test_stacked_predictions)
df_test['cnt'] = pred
df_test[['id', 'cnt']].to_csv('data/submit.csv', header=False,index=False)


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
