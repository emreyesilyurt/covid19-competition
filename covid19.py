#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:01:35 2020

@author: revo
"""


import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Date'] = pd.to_datetime(train['Date'], format = '%Y-%m-%d')
test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')


"""

def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    return df



def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]




#df_train = categoricalToInteger(df_train)
train = create_features(train)



#train, df_dev = train_dev_split(train,0)






columns = ['day','month','Province_State', 'Country_Region','ConfirmedCases','Fatalities']
train = train[columns]
#df_dev = df_dev[columns]
"""




train['Day'] = train['Date'].dt.day
train['Month'] = train['Date'].dt.month

train = train[['Day', 'Month', 'Province_State', 'Country_Region', 'ConfirmedCases', 'Fatalities']]

train['Province_State'].fillna('Na', inplace = True)



test['Day'] = test['Date'].dt.day
test['Month'] = test['Date'].dt.month

test = test[['ForecastId', 'Day', 'Month', 'Province_State', 'Country_Region']]
test['Province_State'].fillna('Na', inplace = True)




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Country_Region'] = le.fit_transform(train['Country_Region'])
train['Province_State'] = le.fit_transform(train['Province_State'])
test['Country_Region'] = le.fit_transform(test['Country_Region'])
test['Province_State'] = le.fit_transform(test['Province_State'])


columns = ['Day','Month','dayofweek','dayofyear','quarter','weekofyear']

sub = []

from xgboost import XGBRegressor
model1 = XGBRegressor(n_estimators = 1000)
model2 = XGBRegressor(n_estimators = 1000)


sub = []
for country in train.Country_Region.unique():
    trainCountry = train[train['Country_Region'] == country]
    for state in trainCountry.Province_State.unique():
        trainProvince = trainCountry[trainCountry['Province_State'] == state]
        train = trainProvince.drop(['Country_Region', 'Province_State'], axis = 1).values
        X_train, y_train = train[:,:-2], train[:,-2:]
        model1 = XGBRegressor(n_estimators=1000)
        model1.fit(X_train, y_train[:,0])
        model2 = XGBRegressor(n_estimators=1000)
        model2.fit(X_train, y_train[:,1])

        #testCountry = test[test['Country_Region'] == country]
        #testProvince = test[(test["Province_State"] == state)]
        
        test3 = test[(test["Country_Region"]==country) & (test["Province_State"] == state)]
        ForecastId = test3.ForecastId.values
        #test1 = test3[columns]
        test1 = test3[['Day', 'Month']] 
        #test2 = test1[columns]
        y_pred1 = model1.predict(test1.values)
        y_pred2 = model2.predict(test1.values)
        
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}
            sub.append(d)
        
        



df_submit = pd.DataFrame(sub)
df_submit.to_csv(r'submission.csv', index=False)
