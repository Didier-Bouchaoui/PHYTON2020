#!/usr/bin/env python
# coding: utf-8

# TP pipline_Classification
# 

# Etape 1 import et clean
# 

# In[15]:


import pandas as pds
import numpy as np
import re
import pickle
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import sklearn as skl
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


# ## Mise en place de la classification
# 

# In[16]:


clf = svm.SVC()
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)


# In[17]:


clf=SVC()


# In[18]:


df = pds.read_csv("dataset/labels.csv", usecols =["class", "tweet"])


# In[19]:


df.head(10)


# In[20]:



df['tweet'] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))


# In[21]:


df


# In[ ]:





# In[ ]:





# In[22]:


clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True)))


# In[27]:


X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.80, random_state=20)
clf.fit(X=X_train, y=y_train)


# In[28]:


res = pickle.dumps(clf)


# In[29]:


clf2 = pickle.loads(res)


# In[30]:


clf2.predict(df['tweet'][0:15])


# In[ ]:





# In[ ]:




