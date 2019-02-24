import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from imblearn.under_sampling import ClusterCentroids

df = pd.read_csv("data/creditcard.csv")

y = df["Class"]
X = df.drop("Class", 1).drop("Time", 1)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=1)

cc = ClusterCentroids()

X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

np.save("resultat/X_resampled", X_resampled)
np.save("resultat/y_resampled", y_resampled)