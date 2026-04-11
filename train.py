# Generated from: train.ipynb
# Converted at: 2026-04-11T07:07:09.696Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd

df=pd.read_csv('Marketing-Customer-Value-Analysis.csv')
df

df.info()

df.describe()

df.isnull().sum()

df.drop(columns=['Customer','State','Effective To Date'],inplace=True)
df



df['EmploymentStatus'].value_counts()



import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Customer Lifetime Value'])
plt.show()

df['Customer Lifetime Value'].hist(bins=50)
plt.show()

Q1 = df['Customer Lifetime Value'].quantile(0.25)
Q3 = df['Customer Lifetime Value'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df['Customer Lifetime Value'] >= lower_bound) & 
              (df['Customer Lifetime Value'] <= upper_bound)]
print(df_clean.shape[0])

sns.boxplot(x=df_clean['Customer Lifetime Value'])
plt.title("After Removing Outliers")
plt.show()

x=df_clean.drop('Customer Lifetime Value',axis=1)
x

import numpy as np
y = np.log1p(df_clean['Customer Lifetime Value'])

df_clean.columns=df_clean.columns.str.strip()
df_clean

num_col=x.select_dtypes(exclude='object').columns.tolist()
num_col

cat_col=x.select_dtypes(include='object').columns.tolist()
cat_col

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

preprocessing=ColumnTransformer([
    ('num_col',StandardScaler(),num_col),
    ('cat_col',OneHotEncoder(handle_unknown='ignore'),cat_col)
])
preprocessing


from xgboost import XGBRegressor

pipeline=Pipeline([
    ('preprocessing',preprocessing),
    ('model',XGBRegressor(n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1))
])
pipeline

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, x, y, cv=5, scoring='r2')
print(scores.mean())

pipeline.fit(x_train,y_train)
y_pred=pipeline.predict(x_test)
y_pred

y_test

from sklearn.metrics import accuracy_score,r2_score, mean_squared_error, mean_absolute_error
print('R2 Score:',r2_score(y_test,y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

import joblib
joblib.dump(pipeline,'ibm_clv.pkl')
print('pkl file is created successfully!...')