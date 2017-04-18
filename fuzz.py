import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skcmeans.algorithms import Probabilistic
from sklearn.datasets import make_blobs

data = pd.read_table("20NG/training.txt") 
data.columns=["Y","X"]
Y = data["Y"]
X = data["X"]

x_train, X, y_train, Y = train_test_split(
    X, Y, test_size=0.1)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2)
    
print(x_train.shape)
print(y_train.shape)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        x_train, 20, 2, error=0.005, maxiter=1000, init=None)
