from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import statistics



iris = datasets.load_iris() #Loading the dataset
iris.keys()

species = iris["target_names"]
feats = iris["feature_names"]
data = iris["data"]
targets = iris["target"]



# General Summary
# for feat in range(len(feats)):
#     feat_list = data[:, feat]

#     # print(feat_list)
#     print(f"{feats[feat]}:")
#     print(f"\tAverage:{statistics.mean(feat_list)}")
#     print(f"\tMode:{statistics.mode(feat_list)}")
#     print(f"\tVariance:{statistics.variance(feat_list)}")
#     print()