import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from util.custom_stopwords import custom_stopword_set
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

stopword_set = set(stopwords.words('english')) | custom_stopword_set
stemmer = SnowballStemmer("english")

data = pd.read_table("data/20NG/training.txt") 
data.columns = ["Y", "X"]
Y = data["Y"]
X = data["X"]
X_post = list()

# Stopword Removal and Stemming
for xi in X:
    X_post.append([stemmer.stem(xij) for xij in xi.split() if xij not in stopword_set])
for index in range(len(X_post)):
    X_post[index] = " ".join([xij for xij in X_post[index] if xij not in stopword_set])

x_train, x_test, y_train, y_test = train_test_split(X_post, Y, test_size=0.2)

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)

print(x_train.shape)
print(y_train.shape)

# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         x_train, 20, 2, error=0.005, maxiter=1000, init=None)
