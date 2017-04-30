import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_table("data/WebKB/webkb-train-stemmed.txt",header=None)
df.columns=["Y","X"]
classes = {'project':0, 'faculty':1, 'course':2, 'student':3}
Y = np.array([classes[i0] for i0 in df["Y"]])
Xshort = np.load("webkb_selected_rfs.npy")
print(Xshort.shape)

# Score with RFC using the novel feature selection
r2 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
scores = cross_val_score(r2, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Score with 5-KNN using the novel feature selection
k2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
scores = cross_val_score(k2, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Score with 3-KNN using the novel feature selection
k2 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
scores = cross_val_score(k2, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Score with Support Vector Classifiers using the novel feature selection
s2 = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
scores = cross_val_score(s2, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

s2 = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
scores = cross_val_score(s2, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Score with Naive Bayes classifiers using the novel feature selection
gnb = GaussianNB()
scores = cross_val_score(gnb, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

bnb = BernoulliNB()
scores = cross_val_score(bnb, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

mnb = MultinomialNB()
scores = cross_val_score(mnb, Xshort, Y, cv=5, scoring='f1_weighted')
print("Fmeasure: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

