import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# This is for the WebKB dataset. This can be extended to other datasets like NG20 and R8 easily

def selecttop(CF, k):
    """
        Finds cosine similarity between SC and Wi and returns index of top features
    """
    NCF = np.zeros((CF.shape[1],CF.shape[1]))
    for i in range(CF.shape[1]):
        for j in range(CF.shape[1]):
            if (CF[i,j]+CF[j,j]-CF[i,j]) !=0:
                NCF[i,j]=CF[i,j]/(CF[i,j]+CF[j,j]-CF[i,j])
            else:
                NCF[i,j]=0
            
    SC = np.zeros(CF.shape[1])
    for i in range(CF.shape[1]):
        SC[i] = np.sum(NCF[i,:])
    
    print(np.isnan(SC).any())
    print(np.isnan(CF).any())
    cosim = cosine_similarity(SC,CF)
    return (-cosim).argsort()[0][:int(k*CF.shape[1])]

#Loading CF matrix for each cluster
CF0 = np.load("webKB/cf_0_webKB.npy")
CF1 = np.load("webKB/cf_1_webKB.npy")
CF2 = np.load("webKB/cf_2_webKB.npy")
CF3 = np.load("webKB/cf_3_webKB.npy")
print(CF0.shape)

#Retrieving indexes of of top features from first cluster
l=[]
percent_features = 0.05
l.extend(selecttop(CF0,percent_features))
print(len(l))

#Retrieving indexes of of top features from second cluster
l.extend(selecttop(CF1,percent_features))
print(len(l))

#Retrieving indexes of of top features from third cluster
l.extend(selecttop(CF2,percent_features))
print(len(l))

#Retrieving indexes of of top features from fourth cluster
l.extend(selecttop(CF3,percent_features))
print(len(l))

# Removing duplicates
l = list(set(l))
print(len(l))

df = pd.read_table("data/WebKB/webkb-train-stemmed.txt",header=None)
df.columns=["Y","X"]
Y = df["Y"]
X = df["X"]

# Loading the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X.values.astype('U')).toarray()

# Selecting top features based on the indexes and seving them
np.save("webkb_selected_rfs",X[:,l])




