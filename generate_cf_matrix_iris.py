from sklearn.datasets import load_iris
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

data = load_iris()
X = data.data

# Adding random features to X
X = np.hstack((X,np.random.rand(len(X),1)))
Y = data.target
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, 3, 60, error=0.0001, maxiter=100000, init=None)
metrics.adjusted_rand_score(Y,np.argmax(u,axis=0))


# Creating the CF Matrix
clu = 1
CF = np.zeros((X.shape[1],X.shape[1]))
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        summ=0
        for d in range(X.shape[0]):
            if np.argmax(u[:,d]) == clu:
                summ+=X[d][i]*X[d][j]
        CF[i,j]=summ

# Normalization:
NCF = np.zeros((X.shape[1],X.shape[1]))
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        NCF[i,j]=CF[i,j]/(CF[i,j]+CF[j,j]-CF[i,j])

# Semantic Centroid
SC = np.zeros(X.shape[1])
for i in range(X.shape[1]):
    SC[i] = np.sum(NCF[i,:])

 # Cosine Similarity to select top features
cosim = cosine_similarity(SC,CF)
print(cosim)
top = 4
print("Top features are: ",(-cosim).argsort()[0][:top])


