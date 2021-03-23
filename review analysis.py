#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train= pd.read_csv("train.csv")


# In[5]:


test=pd.read_csv("test.csv")


# In[6]:


train.head()


# In[36]:


train['Star Rating'].value_counts()


# In[41]:


train['Star Rating'].isnull().sum()


# In[67]:


train.tail()


# In[37]:


train.groupby('Star Rating').describe()


# In[38]:


train.groupby('Review Text').describe()


# In[44]:


train['Review Text'].isnull().sum()


# In[45]:


train['Review Text'].dropna()


# In[51]:


sample_review = train["Review Text"].iloc[60]
print(sample_review)


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet


# In[ ]:


train['Star Rating'].value_counts().plot.bar(color = 'magenta')
plt.title('Visualizing the Ratings dist.')
plt.xlabel('Ratings')
plt.ylabel('count')
plt.show()


# In[26]:


data['Star Rating'].value_counts()

labels = '5', '4', '3', '2', '1'
sizes = [2286, 455, 161, 152, 96]
colors = ['blue', 'magenta', 'green', 'yellow', 'red']
explode = [0.001, 0.001, 0.001, 0.001, 0.001]


# In[27]:


plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title(' pie chart representing ratings ')
plt.show()


# In[30]:


# cleaning the texts
# importing the libraries for Natural Language Processing

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[59]:


corpus = []

for i in range(0, 3150):
  review = re.sub('[^a-zA-Z]', ' ', str(train['Review Text'][i]))
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)


# In[81]:


# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x = cv.fit_transform(corpus)

y =train['Star Rating']

print(x.shape)
print(y.shape)


# In[82]:


X_train=x
y_train=train['Star Rating']


# In[83]:


X_test=test[:-1]


# In[84]:


test.head()


# In[85]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[86]:


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training Accuracy :", model.score(x_train, y_train))
#print("Testing Accuracy :", model.score(x_test, y_test))


# In[ ]:




