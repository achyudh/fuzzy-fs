import numpy as np
import skfuzzy as fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from tqdm import tqdm
from sklearn.cluster import KMeans

def random_class_selection(data):
    classes = ['project', 'faculty', 'course', 'student']
    #np.random.shuffle(classes)
    group5 = data.loc[data['Y'] == classes[0]]
    group5 = group5.append(data.loc[data['Y'] == classes[1]], ignore_index=True)
    group5 = group5.append(data.loc[data['Y'] == classes[2]], ignore_index=True)
    return group5.append(data.loc[data['Y'] == classes[3]], ignore_index=True)


data = pd.read_table("data/WebKB/webkb-train-stemmed.txt")
data.columns=["Y","X"]
data = random_class_selection(data)
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

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x_train.T, 16, 4, error=0.0001, maxiter=100000, init=None)
print('Score:',metrics.adjusted_rand_score(y_train,np.argmax(u,axis=0)))

docinc = [[] for x in range(5)]

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