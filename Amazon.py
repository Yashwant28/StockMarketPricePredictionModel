
# coding: utf-8

# In[24]:


import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm


# In[25]:


df = quandl.get("WIKI/AMZN")


# In[26]:


df.tail()


# In[27]:


df = df[['Adj. Close']]


# In[28]:


forecast_out = int(10)


# In[29]:


df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)


# In[30]:


X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)


# In[31]:


X


# In[32]:


X_forecast = X[-forecast_out:]


# In[33]:


X = X[:-forecast_out]


# In[34]:


y = np.array(df['Prediction'])
y = y[:-forecast_out]


# In[36]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


# In[38]:


clf = LinearRegression()
clf.fit(X_train,y_train)


# In[41]:


confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)


# In[43]:


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


# In[45]:


import matplotlib.pyplot as plt

