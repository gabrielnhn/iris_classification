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
        print(feat)
        feat_list = this_species[:, feat]
        # print(feat_list)
        print(f"\t{feats[feat]}:")
        print(f"\t\tAverage:{statistics.mean(feat_list)}")
        print(f"\t\tMode:{statistics.mode(feat_list)}")
        print(f"\t\tVariance:{statistics.variance(feat_list)}")
        print()


for i, s in enumerate(species):
    this_species = data_per_class[s]["data"]

    sepal_width = this_species[:, 1]
    sepal_len = this_species[:, 0]

    plt.title(f"Sepal size")

    plt.xlabel("Sepal Width")
    plt.ylabel("Sepal Length")

    colors = "red", "green", "blue"
    plt.scatter(sepal_width, sepal_len, color=colors[i], label=s)

    plt.xlim(1.5, 4.5)
    plt.ylim(4, 8)

    # plt.savefig(f"{s}_petal.png", dpi=150)
    # plt.show()
plt.legend()
plt.show()



for i, s in enumerate(species):
    this_species = data_per_class[s]["data"]

    petal_width = this_species[:, 3]
    petal_len = this_species[:, 2]

    plt.title(f"Petal size")

    plt.xlabel("Petal Width")
    plt.ylabel("Petal Length")

    colors = "red", "green", "blue"
    plt.scatter(petal_width, petal_len, color=colors[i], label=s)

    plt.xlim(0, 3)
    plt.ylim(0, 8)

    # plt.savefig(f"{s}_petal.png", dpi=150)
    # plt.show()
plt.legend()
plt.show()