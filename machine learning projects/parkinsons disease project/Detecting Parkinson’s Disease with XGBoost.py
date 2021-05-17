import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Masaüstü\\sifirdan ileri seviyeye python programlama ornek uygulamalar\\parkinsons disease\\parkinsons_data.csv')
# print(df.head())

# get the features and labels
features = df.loc[:, df.columns != 'status'].values[:,1:]
labels = df.loc[:, 'status'].values

#get the count of each label (0 and 1) in labels
# print("1'lerin sayısı:", labels[labels == 1].shape[0], "0'ların sayisi:", labels[labels == 0].shape[0])

#scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

#split the dataset into training and testing data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)


#train the model
model = XGBClassifier(use_label_encoder=False)
model.fit(x_train, y_train)

#calculate the accuracy
y_pred = model.predict(x_test)
print("Modelimizin doğruluk puani:", accuracy_score(y_test, y_pred) * 100)









