from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('submission.csv')
train.isnull().sum()
test.isnull().sum()

train = train.drop(['County', 'Province_State',
                    'Country_Region', 'Target'], axis=1)
test = test.drop(['County', 'Province_State',
                  'Country_Region', 'Target'], axis=1)


def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


def train_dev_split(df, days):
    # Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]


test_date_min = test['Date'].min()
test_date_max = test['Date'].max()


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date'] < date]


def to_integer(dt_time):
    return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
test['Date'] = test['Date'].dt.strftime("%Y%m%d")
train['Date'] = train['Date'].dt.strftime("%Y%m%d").astype(int)


predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.22, random_state=0)

model = RandomForestRegressor(n_jobs=-1)
estimators = 100
scores = []
model.set_params(n_estimators=estimators)
model.fit(X_train, y_train)
scores.append(model.score(X_test, y_test))
test.drop(['ForecastId'], axis=1, inplace=True)
test.index.name = 'Id'
y_pred2 = model.predict(X_test)

predictions = model.predict(test)

pred_list = [int(x) for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)

a = output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b = output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c = output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns = ['Id', 'q0.05']
b.columns = ['Id', 'q0.5']
c.columns = ['Id', 'q0.95']
a = pd.concat([a, b['q0.5'], c['q0.95']], 1)
a['q0.05'] = a['q0.05'].clip(0, 10000)
a['q0.5'] = a['q0.5'].clip(0, 10000)
a['q0.95'] = a['q0.95'].clip(0, 10000)
a['Id'] = a['Id'] + 1

sub = pd.melt(a, id_vars=['Id'], value_vars=['q0.05', 'q0.5', 'q0.95'])
sub['variable'] = sub['variable'].str.replace("q", "", regex=False)
sub['ForecastId_Quantile'] = sub['Id'].astype(str) + '_' + sub['variable']
sub['TargetValue'] = sub['value']
sub = sub[['ForecastId_Quantile', 'TargetValue']]
sub.reset_index(drop=True, inplace=True)
sub.to_csv("submission.csv", index=False)

print(sub.head())
