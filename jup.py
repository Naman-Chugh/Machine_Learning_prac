#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


# In[3]:


df = pd.read_csv('C:/Users/MSI1/OneDrive/Desktop/Machine Learning/02Linear Regression Multiple Variables/homeprices.csv')


# In[15]:


df.head()


# In[9]:


median_bedrooms = df['bedrooms'].median()


# In[14]:


df.bedrooms.fillna(median_bedrooms, inplace=True)


# In[16]:


reg = linear_model.LinearRegression()


# In[17]:


reg.fit(df[['area','bedrooms','age']],df['price'])


# In[18]:


reg.coef_


# In[19]:


reg.intercept_


# In[21]:


reg.predict([[33000,4,2]])


# In[ ]:




