
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skcmeans.algorithms import Probabilistic
from sklearn.cluster import KMeans
from tqdm import tqdm


# In[2]:

data = pd.read_table("20NG/20ng-train-stemmed.txt") 
data.columns=["Y","X"]


# # Returns a Dataframe with five random classes

# In[3]:

def ran5class(data):
    classes = ['alt.atheism',
                 'comp.graphics',
                 'comp.os.ms-windows.misc',
                 'comp.sys.ibm.pc.hardware',
                 'comp.sys.mac.hardware',
                 'comp.windows.x',
                 'misc.forsale',
                 'rec.autos',
                 'rec.motorcycles',
                 'rec.sport.baseball',
                 'rec.sport.hockey',
                 'sci.crypt',
                 'sci.electronics',
                 'sci.med',
                 'sci.space',
                 'soc.religion.christian',
                 'talk.politics.guns',
                 'talk.politics.mideast',
                 'talk.politics.misc',
                 'talk.religion.misc']
    #np.random.shuffle(classes)

    group5 = data.loc[data['Y'] == classes[1]]
    group5 = group5.append(data.loc[data['Y'] == classes[7]], ignore_index=True)
    group5 = group5.append(data.loc[data['Y'] == classes[12]], ignore_index=True)
    group5 = group5.append(data.loc[data['Y'] == classes[16]], ignore_index=True)
    return group5.append(data.loc[data['Y'] == classes[0]], ignore_index=True)


# In[4]:

group5 = ran5class(data)

Y = group5["Y"]
X = group5["X"]

vectorizer = TfidfVectorizer()
# counter = CountVectorizer()
# Xc = counter.fit_transform(X).toarray()
X = vectorizer.fit_transform(X).toarray()

x_train = X
y_train = Y


# In[5]:

print(x_train.shape)
print(y_train.shape)


# # Clustering

# In[6]:

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x_train.T, 5, 15, error=0.00001, maxiter=10000, init=None)
metrics.adjusted_rand_score(y_train,np.argmax(u,axis=0))


# In[7]:

# kmeans = KMeans(n_clusters=3,n_jobs=-1).fit(x_train)
# print('Score:',metrics.adjusted_rand_score(y_train,kmeans.predict(x_train)))


# In[10]:

docinc = [ [] for x in range(5)]
for k in range(x_train.shape[0]):
    docinc[np.argmax(u[:,k])].append(k)
clu = 0
CF = np.zeros((x_train.shape[1],x_train.shape[1]))
for i in tqdm(range(x_train.shape[1])):
    for j in range(x_train.shape[1]):
        summ=0
        for k in docinc[clu]:
            summ+=X[k][i]*X[k][j]
        CF[i,j]=summ




