#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[5]:


df = pd.read_csv('C:/Users/MSI1/OneDrive/Desktop/Prac.csv')


# In[6]:


df.head()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Area in Sq Ft")
plt.ylabel("Price")
plt.title("Price v/s Area")
plt.ticklabel_format(style='plain')
plt.scatter(df.Area, df.Price, color="green")


# In[32]:


reg = linear_model.LinearRegression()
reg.fit(df[["Area"]], df.Price)


# In[46]:


reg.predict([[2250]])


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Area in Sq Ft")
plt.ylabel("Price")
plt.title("Price v/s Area")
plt.ticklabel_format(style='plain')
plt.scatter(df.Area, df.Price, color="green")
plt.plot(df.Area, reg.predict(df[["Area"]]))
plt.grid()

