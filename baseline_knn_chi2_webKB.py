import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

data = pd.read_table("data/WebKB/webkb-train-stemmed.txt")
data.columns=["Y","X"]
classes = {'project':0, 'faculty':1, 'course':2, 'student':3}
Y = np.array([classes[i0] for i0 in data["Y"]])
X = data["X"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X.values.astype('U')).toarray()

x_train = SelectKBest(chi2, k=0.1*len(X[1])).fit_transform(X, Y)
y_train = Y

# data = pd.read_table("data/WebKB/webkb-test-stemmed.txt")
# data.columns=["Y","X"]
# classes = {'project':0, 'faculty':1, 'course':2, 'student':3}
# Y = np.array([classes[i0] for i0 in data["Y"]])
# X = data["X"]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(X.values.astype('U')).toarray()

# x_test = SelectKBest(chi2, k=0.1*len(X[1]).fit_transform(X, y))
# y_test = Y

knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=1) # Minkowski distance with p=2
psum = rsum = fsum = 0
# scoring=metrics.make_scorer(metrics.precision_recall_fscore_support)
cv_scores = cross_val_score(knn_model, x_train, y_train, scoring='f1_weighted', cv=5, n_jobs=5)
print(sum(cv_scores)/5)