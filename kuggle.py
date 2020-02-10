#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[30]:


# Load the data
df_train = (pd.read_csv('train.csv').drop(columns=['id', 'opened_position_qty ', 'closed_position_qty']).to_numpy(), pd.read_csv('train.csv', usecols=['y']).to_numpy())
df_test = pd.read_csv('test.csv').drop(columns=['id', 'opened_position_qty ', 'closed_position_qty']).to_numpy()


# In[ ]:


data, labels = df_train
print(labels[0])
logreg = linear_model.LogisticRegression()
logreg.fit(data, labels)

