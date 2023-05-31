# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:33:36 2022

@author: Harshitha
"""
import pandas as pd
from matplotlib import pyplot as plt
import os
os.getcwd()
import pandas as pd
os.chdir("C:/Users/Harshitha/OneDrive/Documents/python code")
df=pd.read_csv("HR_comma_sep.csv")
left = df[df.left==1]
left.shape
retained = df[df.left==0]
retained.shape
df.groupby('left').mean()
pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.head()
df_with_dummies.drop('salary',axis='columns',inplace=True)
df_with_dummies.head()
X = df_with_dummies
X.head()
y = df.left
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.predict(X_test)
model.score(X_test,y_test)
