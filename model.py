import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomato_df.csv')
df=df.head(3700)

df.drop('Unnamed: 0',axis=1,inplace=True)
#print(df.head())
x=df.drop('rate',axis=1)
y=df['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)


#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)
from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(x_train,y_train)
#y_predict=ET_Model.predict(x_test)


def generate_pred_etr(modelname,model,x_train,x_test,y_train,y_test):

    ET_Model=ExtraTreesRegressor(n_estimators = 120)
    ET_Model.fit(x_train,y_train)
    y_train_pred=ET_Model.predict(x_train)
    
    print("------Evaluation metrics for training data set--------")
    rmse_train=np.sqrt(mean_squared_error(y_train,y_train_pred))
    Rsqr_train=round(r2_score(y_train,y_train_pred)*100,2)
    print("modelname-",modelname)
    print("rmse is",rmse_train)
    print("Rsqr is ",Rsqr_train)
    
    print("-------Evaluation metrics for test dataset--------")
    y_test_pred=ET_Model.predict(x_test)
    rmse_test=np.sqrt(mean_squared_error(y_test,y_test_pred))
    Rsqr_test=round(r2_score(y_test,y_test_pred)*100,2)
    print("modelname-",modelname)
    print("rmse is",rmse_test)
    print("Rsqr is ",Rsqr_test)


generate_pred_etr("Extraa tree regressor",ET_Model,x_train,x_test,y_train,y_test)
import pickle
# # Saving model to disk
pickle.dump(ET_Model, open('model4.pkl','wb'))
model=pickle.load(open('model4.pkl','rb'))
#print(y_predict)
#from sklearn.externals import joblib
#joblib.dump(model,"model_joblib")
#mj=joblib.load("model_joblib")



