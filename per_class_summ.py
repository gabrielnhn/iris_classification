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


data_per_class = {species[i]:[] for i in range(3)}
for i, a in enumerate(data):
    data_per_class[species[targets[i]]].append(a)

for x in data_per_class.keys():
    l = data_per_class[x] 
    data_per_class[x] = {}
    data_per_class[x]["data"] = np.array(l)
        
for i, specie in enumerate(species):
    print(f"Species {specie}:")
    this_species = data_per_class[specie]["data"]

    for feat in range(len(feats)):
        # print(feat)
        feat_list = this_species[:, feat]
        # print(feat_list)
        print(f"\t{feats[feat]}:")
        print(f"\t\tAverage:{statistics.mean(feat_list)}")
        print(f"\t\tMode:{statistics.mode(feat_list)}")
        print(f"\t\tVariance:{statistics.variance(feat_list)}")
        print()

        data_per_class[specie]["avg"] = statistics.mean(feat_list)
        data_per_class[specie]["mode"] = statistics.mode(feat_list)
        data_per_class[specie]["variance"] = statistics.variance(feat_list)
