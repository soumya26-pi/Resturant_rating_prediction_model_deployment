import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Zomato_data1.csv')
df=df.head(3700)

df.drop('Unnamed: 0',axis=1,inplace=True)
#print(df.head())
x=df.drop('rate',axis=1)
y=df['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)


#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ETR_Model=ExtraTreesRegressor(n_estimators = 120)
ETR_Model.fit(x_train,y_train)
y_predict=ETR_Model.predict(x_test)
from sklearn.ensemble import  ExtraTreesRegressor
ETR_Model=ExtraTreesRegressor(n_estimators = 120)
ETR_Model.fit(x_train,y_train)
#y_predict=ET_Model.predict(x_test)



import pickle
# # Saving model to disk
pickle.dump(ETR_Model, open('ETR_Model.pkl','wb'))
model=pickle.load(open('ETR_Model.pkl','rb'))
#print(y_predict)
#from sklearn.externals import joblib
#joblib.dump(model,"model_joblib")
#mj=joblib.load("model_joblib")



