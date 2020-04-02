# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:59:01 2020

@author: dell
"""
# Data Wrangling 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Machine Learning 
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron 
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train_data = pd.read_csv(r'C:\Users\dell\Desktop\insurance\train.csv')
test_data = pd.read_csv(r'C:\Users\dell\Desktop\insurance\test.csv')
combine = [train_data, test_data]

for dataset in combine: 
    dataset['age'] = dataset['age_in_days']//365
    dataset.drop(['age_in_days'], axis = 1, inplace = True)

train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()

train_data['IncomeBands'] = pd.cut(train_data['Income'], 5)
train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()

scaler = MinMaxScaler()
scaler = scaler.fit(train_data[['Income']])
x_scaled = scaler.transform(train_data[['Income']])

train_data['scaled_income'] = x_scaled

train_data['IncomeBands'] = pd.cut(train_data['scaled_income'], 5)
train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()

upper_bound = 0.95
lower_bound = 0.1
res = train_data['Income'].quantile([lower_bound, upper_bound])

true_index = (train_data['Income'] < res.loc[upper_bound])

false_index = ~true_index

no_outlier_data = train_data[true_index].copy()

no_outlier_data['IncomeBands'] = pd.cut(no_outlier_data['Income'], 5)
no_outlier_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()

combine = [train_data, test_data]
for dataset in combine: 
    dataset.loc[ dataset['Income'] <= 23603.99, 'Income'] = 0
    dataset.loc[(dataset['Income'] > 23603.99) & (dataset['Income'] <= 109232.0), 'Income'] = 1
    dataset.loc[(dataset['Income'] > 109232.0) & (dataset['Income'] <= 194434.0), 'Income'] = 2
    dataset.loc[(dataset['Income'] > 194434.0) & (dataset['Income'] <= 279636.0), 'Income'] = 3
    dataset.loc[(dataset['Income'] > 279636.0) & (dataset['Income'] <= 364838.0), 'Income'] = 4
    dataset.loc[(dataset['Income'] > 364838.0) & (dataset['Income'] <= 450040.0), 'Income'] = 5
    dataset.loc[ dataset['Income'] > 450040.0, 'Income'] = 6

train_data.loc[false_index, 'Income'] = 5

train_data.drop(['IncomeBands', 'scaled_income'], axis = 1, inplace = True)

train_data['AgeBands'] = pd.cut(train_data['age'], 5)
train_data[['AgeBands', 'target']].groupby('AgeBands', as_index = False).count()

for dataset in combine:    
    dataset.loc[ dataset['age'] <= 37.4, 'age'] = 0
    dataset.loc[(dataset['age'] > 37.4) & (dataset['age'] <= 53.8), 'age'] = 1
    dataset.loc[(dataset['age'] > 53.8) & (dataset['age'] <= 70.2), 'age'] = 2
    dataset.loc[(dataset['age'] > 70.2) & (dataset['age'] <= 86.6), 'age'] = 3
    dataset.loc[ dataset['age'] > 86.6, 'age'] = 4
train_data.drop('AgeBands', axis = 1, inplace = True)
combine = [train_data, test_data]

train_data[['age', 'application_underwriting_score']].groupby('age').mean()

train_data['PremBand'] = pd.cut(train_data['no_of_premiums_paid'], 5)
train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()

train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()

train_data[['sourcing_channel', 'application_underwriting_score']].groupby('sourcing_channel', as_index = False).mean()

train_data[['residence_area_type', 'application_underwriting_score']].groupby('residence_area_type', as_index = False).mean()

combine = [train_data, test_data]
for dataset in combine: 
    mask1 = dataset['application_underwriting_score'].isnull()
    for source in ['A', 'B', 'C', 'D', 'E']:
        mask2 = (dataset['sourcing_channel'] == source)
        source_mean = dataset[dataset['sourcing_channel'] == source]['application_underwriting_score'].mean()
        dataset.loc[mask1 & mask2, 'application_underwriting_score'] = source_mean

test_data[test_data['Count_3-6_months_late'].isnull()]
dataset['application_underwriting_score'].isnull()

combine = [train_data, test_data]
for dataset in combine: 
    dataset['late_premium'] = 0.0

combine = [train_data, test_data]
for dataset in combine:
        dataset.loc[(dataset['Count_3-6_months_late'].isnull()),  'late_premium'] = np.NaN
        dataset.loc[(dataset['Count_3-6_months_late'].notnull()), 'late_premium'] = dataset['Count_3-6_months_late'] + dataset['Count_6-12_months_late'] + dataset['Count_more_than_12_months_late']

train_data[['late_premium', 'target']].groupby('late_premium').mean()

train_data.loc[(train_data['target'] == 0) & (train_data['late_premium'].isnull()),'late_premium'] = 7
train_data.loc[(train_data['target'] == 1) & (train_data['late_premium'].isnull()),'late_premium'] = 2


guess_prem = np.zeros(5)
for dataset in [test_data]:
    for i in range(1, 6):
        guess_df = dataset[(dataset['Income'] == i)]['late_premium'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        premium_guess = guess_df.median()
        guess_prem[i - 1] = int(premium_guess) 

    for j in range(1, 6):
        dataset.loc[(dataset.late_premium.isnull()) & (dataset.Income == j), 'late_premium'] = guess_prem[j - 1] + 1

    dataset['late_premium'] = dataset['late_premium'].astype(int)

train_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)
test_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)

combine = [train_data, test_data]
for dataset in combine: 
    dataset['residence_area_type'] = dataset['residence_area_type'].map( {'Urban' : 1, 'Rural' : 0} )
    dataset['sourcing_channel'] = dataset['sourcing_channel'].map( {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4} )

train_data['application_underwriting_score'] = train_data['application_underwriting_score']/100

upper_bound = 0.95
res = train_data['no_of_premiums_paid'].quantile([upper_bound])

true_index = train_data['no_of_premiums_paid'] < res.loc[upper_bound]
false_index = ~true_index

train_data['PremBand'] = pd.cut(train_data[true_index]['no_of_premiums_paid'], 4)
train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()

upper_bound = 0.90
res = train_data['premium'].quantile([upper_bound])

true_index = train_data['premium'] < res.loc[upper_bound]
false_index = ~true_index

train_data['PremBand'] = pd.cut(train_data[true_index]['premium'], 4)
train_data[['PremBand', 'target']].groupby('PremBand').count()

combine = [train_data]
for dataset in combine: 
    dataset.loc[ dataset['premium'] <= 5925.0, 'premium'] = 0
    dataset.loc[(dataset['premium'] > 5925.00) & (dataset['premium'] <= 10650.0), 'premium'] = 1
    dataset.loc[(dataset['premium'] > 10650.0) & (dataset['premium'] <= 15375.0), 'premium'] = 2
    dataset.loc[(dataset['premium'] > 15375.0) & (dataset['premium'] <= 201200.0), 'premium'] = 3
    dataset.loc[ dataset['premium'] > 201200.0, 'premium'] = 4
train_data.drop('PremBand', axis = 1, inplace = True)
combine = [train_data, test_data]

train_data['PremBand'] = pd.cut(train_data['perc_premium_paid_by_cash_credit'], 4)
train_data[['PremBand', 'target']].groupby('PremBand').mean()

combine = [train_data, test_data]
for dataset in combine: 
    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] <= 0.25, 'perc_premium_paid_by_cash_credit'] = 0
    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.25) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.5), 'perc_premium_paid_by_cash_credit'] = 1
    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.5) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.75), 'perc_premium_paid_by_cash_credit'] = 2
    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] > 0.75, 'perc_premium_paid_by_cash_credit'] = 3
train_data.drop('PremBand', axis = 1, inplace = True)
train_data.head()

train_data[['perc_premium_paid_by_cash_credit', 'late_premium']] = train_data[['perc_premium_paid_by_cash_credit', 'late_premium']].astype(int)
test_data[['perc_premium_paid_by_cash_credit']] = test_data[['perc_premium_paid_by_cash_credit']].astype(int)

X_train = train_data.drop(['id', 'target', 'premium', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()
y_train = train_data['target']
X_test = test_data.drop(['id', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()

ydata=train_data.groupby("target")
data1=ydata.get_group(1)
data0 = ydata.get_group(0)
data1 = data1.sample(n=4998, replace = False, random_state = 2)
data = pd.concat([data1,data0], ignore_index = True)
data['target'].value_counts()
data = data.drop(['id', 'premium', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis=1), data['target'], test_size = 0.25, random_state = 0)


classifier = DecisionTreeClassifier(max_depth = 7)
classifier.fit(X_train, y_train)

import pickle
pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
a=model.predict(X_test)
y_pred = classifier.predict(X_test)
print(classifier.feature_importances_)
from sklearn.metrics import classification_report
result = classification_report(y_test, a)
