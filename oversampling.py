import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import preprocessing
import time


df = pd.read_csv("data/creditcard.csv")
y = df["Class"]
X = df.drop("Class", 1).drop("Time", 1)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

models = (
    LinearDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=(100, 2)),
    GaussianNB(),
    xgb.XGBClassifier(n_estimators=50, n_jobs=-1),
    RandomForestClassifier(max_depth=15, n_estimators=50, n_jobs=-1),
    svm.LinearSVC(),
    LogisticRegression(n_jobs=-1)  
)


print(" -------- RANDOM OVER SAMPLING --------------")

ros = RandomOverSampler(random_state=0)
X_rdm, y_rdm = ros.fit_resample(X, y)


for m in models :
    str(type(m))
    start_time = time.time()
    cv = cross_val_score(m, X_rdm, y_rdm, cv=10, scoring=make_scorer(roc_auc_score))
    print("--- > Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")


    
models = (
    LinearDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=(100, 2)),
    GaussianNB(),
    xgb.XGBClassifier(n_estimators=50, n_jobs=-1),
    RandomForestClassifier(max_depth=15, n_estimators=50, n_jobs=-1),
    svm.LinearSVC(),
    LogisticRegression(n_jobs=-1)  
)


print(" -------- SMOTE OVER SAMPLING --------------")

X_resampler_SMOTE, y_resampled_SMOTE = SMOTE().fit_resample(X, y)

for m in models :
    str(type(m))
    start_time = time.time()
    cv = cross_val_score(m, X_resampler_SMOTE, y_resampled_SMOTE, cv=10, scoring=make_scorer(roc_auc_score))
    print("--- > Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")





models = (
    LinearDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=(100, 2)),
    GaussianNB(),
    xgb.XGBClassifier(n_estimators=50, n_jobs=-1),
    RandomForestClassifier(max_depth=15, n_estimators=50, n_jobs=-1),
    svm.LinearSVC(),
    LogisticRegression(n_jobs=-1)  
)


print(" -------- ADASYN OVER SAMPLING --------------")

X_resampler_ADASYN, y_resampled_ADASYN = ADASYN().fit_resample(X, y)


for m in models :
    str(type(m))
    start_time = time.time()
    cv = cross_val_score(m, X_resampler_ADASYN, y_resampled_ADASYN, cv=10, scoring=make_scorer(roc_auc_score))
    print("--- > Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")
