import numpy as np
import skfuzzy as fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from tqdm import tqdm
from sklearn.cluster import KMeans
from multiprocessing import Pool

def random_class_selection(data):
    classes = ['acq',
               'crude',
               'earn',
               'grain',
               'interest',
               'money-fx',
               'ship',
               'trade']
    group5 = data.loc[data['Y'] == classes[0]]
    group5 = group5.append(data.loc[data['Y'] == classes[1]], ignore_index=True)
    group5 = group5.append(data.loc[data['Y'] == classes[2]], ignore_index=True)
    return group5.append(data.loc[data['Y'] == classes[3]], ignore_index=True)

def generate_cf_matrix(clu):
    CF = np.zeros((x_train.shape[1],x_train.shape[1]))
    for i in tqdm(range(x_train.shape[1])):
        for j in range(x_train.shape[1]):
            summ=0
            for k in docinc[clu]:
                summ+=X[k][i]*X[k][j]
            CF[i,j]=summ

    np.save("cf_%d_webKB.npy"%clu, CF)

    # CF = np.load("cf_webKB.npy")
    NCF = np.zeros((X.shape[1],X.shape[1]))
    for i in range(X.shape[1]):
        break
        for j in range(X.shape[1]):
            # print(CF[i,j], (CF[i,j]+CF[j,j]-CF[i,j]))
            NCF[i,j]=(0.001 + CF[i,j])/(0.001 +(CF[i,j]+CF[j,j]-CF[i,j]))
            break
    SC = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        SC[i] = np.sum(NCF[i,:])

    cosim = cosine_similarity(SC, CF)
    print(cosim)
    # top = 10
    # print("Top features are: ", (-cosim).argsort()[0][:top])
    np.save("sc_%d_webKB.npy"%clu, SC)

data = pd.read_table("data/WebKB/webkb-train-stemmed.txt")
data.columns=["Y","X"]
# data = random_class_selection(data)
Y = data["Y"]
X = data["X"]
vectorizer = TfidfVectorizer()
# counter = CountVectorizer()
# Xc = counter.fit_transform(X).toarray()
X = vectorizer.fit_transform(X.values.astype('U')).toarray()

x_train = X
y_train = Y

# kmeans = KMeans(n_clusters=4,n_jobs=-1).fit(x_train)
# print('Score:',metrics.adjusted_rand_score(y_train,kmeans.predict(x_train)))

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x_train.T, 4, 12, error=0.00001, maxiter=100000, init=None)
print('Score:',metrics.adjusted_rand_score(y_train,np.argmax(u,axis=0)))

docinc = [[] for x in range(4)]
for k in range(x_train.shape[0]):
    docinc[np.argmax(u[:,k])].append(k)

clus = [0, 1, 2, 3]
mpool = Pool(4)
mapped_data = mpool.map(generate_cf_matrix, clus)
mpool.close()