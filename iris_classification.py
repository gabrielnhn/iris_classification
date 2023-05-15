from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as KNN

import numpy as np
from matplotlib import pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import math

iris = datasets.load_iris() #Loading the dataset
iris.keys()

species = iris["target_names"]
feats = iris["feature_names"]
data = iris["data"]
targets = iris["target"]

x_train,x_test,y_train,y_test=train_test_split(data,targets,test_size=0.2,random_state=42)

k = int(math.sqrt(len(data)))
knn = KNN(k)

print("Fitting...")
knn.fit(x_train, y_train)


out = knn.score(x_test, y_test)
print(f"Accuracy for test: {out}")

y_pred = knn.predict(x_test)


out = classification_report(y_test, y_pred)
print(out)


