from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import make_scorer
from tqdm import tqdm
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids


def calcul_metrics(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return (roc_auc_score(y_true, y_pred), auc(recall, precision))

def cacul_score(scores):
    print(" ROC auc : ", np.round(np.mean(scores["roc_auc"]), decimals=3), " (+/-", np.round(np.std(scores["roc_auc"]), decimals=2), ")")
    print(" P-R auc : ", np.round(np.mean(scores["pr_auc"]), decimals=3), " (+/-", np.round(np.std(scores["pr_auc"]), decimals=2), ")")
    print(" Time :", np.round(np.mean(scores["time"]), decimals=2), " sec")
    print(" Time TT:", np.round(np.mean(scores["time_tt"]), decimals=2), " sec")
    
    return (
        np.round(np.mean(scores["roc_auc"]), decimals=3),
        np.round(np.std(scores["roc_auc"]), decimals=2),
        np.round(np.mean(scores["pr_auc"]), decimals=3),
        np.round(np.std(scores["pr_auc"]), decimals=2),
        np.round(np.mean(scores["time"]), decimals=2),
        np.round(np.mean(scores["time_tt"]), decimals=2)
    )


# def run_model(name, model):
#     skf=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
#     scores = {"time" : [], "roc_auc": [], "pr_auc" : [], "time_tt": []}
#     print(" ---------- ", name, " ------- ")
#     for train_index, test_index in tqdm(skf.split(X, y)):
#         x_train, y_train = X[train_index], y[train_index]
#         x_test, y_test = X[test_index], y[test_index]
#         start_time_tt = time.time()
#         rus = RandomUnderSampler(random_state=0)
#         X_res, y_res = rus.fit_resample(x_train, y_train)
#         start_time = time.time()
#         m = model
#         m.fit(X_res, y_res)
#         pred = m.predict(x_test)
#         roc_auc, pr_auc = calcul_metrics(y_test, pred)
#         scores["time"].append(time.time() - start_time)
#         scores["time_tt"].append(time.time() - start_time_tt)
#         scores["roc_auc"].append(roc_auc)
#         scores["pr_auc"].append(pr_auc)
#     score_final = cacul_score(scores)
#     np.save("scores/under/random_"+name, score_final)


def run_model(name, model):
    skf=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
    scores = {"time" : [], "roc_auc": [], "pr_auc" : [], "time_tt": []}
    print(" ---------- ", name, " ------- ")
    for x in range(1,11):

        X_train = np.load("scores/under/centroids/k-"+str(x)+"-X_train.npy")
        y_train = np.load("scores/under/centroids/k-"+str(x)+"-y_train.npy")
        
        X_test = np.load("scores/under/centroids/k-"+str(x)+"-X_test.npy")
        y_test = np.load("scores/under/centroids/k-"+str(x)+"-y_test.npy")

        start_time = time.time()
        m = model
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        roc_auc, pr_auc = calcul_metrics(y_test, pred)
        scores["time"].append(time.time() - start_time)
        scores["time_tt"].append(80)
        scores["roc_auc"].append(roc_auc)
        scores["pr_auc"].append(pr_auc)
    score_final = cacul_score(scores)
    np.save("scores/under/centroids_"+name, score_final)


df = pd.read_csv("data/creditcard.csv")
y = df["Class"]
X = df.drop("Class", 1).drop("Time", 1)
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)


# models = (
#     (QuadraticDiscriminantAnalysis(), "QLA"),
#     (MLPClassifier(hidden_layer_sizes=(100,)), "MLP"),
#     (GaussianNB(), "Naive Bayes"),
#     (xgb.XGBClassifier(n_estimators=50, n_jobs=10), "XGBoost"),
#     (RandomForestClassifier(max_depth=15, n_estimators=50, n_jobs=10), "Random Forest"),
#     (svm.LinearSVC(), "SVM"),
#     (LogisticRegression(n_jobs=10, solver="lbfgs"), "Logistic Regression")
# )

# for (m, name) in models:
# 	run_model(name, m)




i=1
skf=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

for train_index, test_index in tqdm(skf.split(X, y)):
    x_train, y_train = X[train_index], y[train_index]
    x_test, y_test = X[test_index], y[test_index]

    start_time_tt = time.time()
    cc = ClusterCentroids()
    X_res, y_res = cc.fit_resample(x_train, y_train)
    
    
    np.save("scores/under/centroids/k-"+str(i)+"-X_train", X_res)
    np.save("scores/under/centroids/k-"+str(i)+"-y_train", y_res)
        
    np.save("scores/under/centroids/k-"+str(i)+"-X_test", x_test)
    np.save("scores/under/centroids/k-"+str(i)+"-y_test", y_test)
    
    print(time.time() - start_time_tt)  

# import os
# os.system('shutdown -s')