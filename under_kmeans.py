import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from imblearn.under_sampling import ClusterCentroids

df = pd.read_csv("data/creditcard.csv")


y = df["Class"]
X = df.drop("Class", 1).drop("Time", 1)

scaler = preprocessing.StandardScaler()
scaler.fit(X)

train, test = train_test_split(df, test_size=0.20, stratify=y)

x_train_0 = train[train["Class"] == 0].drop("Class", axis=1).drop("Time", axis=1)
x_train_1 = train[train["Class"] == 1].drop("Class", axis=1).drop("Time", axis=1)
x_train_0 = scaler.transform(x_train_0)
x_train_1 = scaler.transform(x_train_1)

km = KMeans(n_clusters=1000, n_jobs=10, n_init=5, max_iter=100, verbose=1).fit(x_train_0)
np.save("resultat/km_centroid", km.cluster_centers_)