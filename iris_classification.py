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
    # Create a bar plot
    # fig = plt.figure()

    plt.title(s.title())

    plt.xlabel("Sepal Width")
    plt.ylabel("Sepal Length")

    colors = "red", "green", "blue"
    plt.scatter(sepal_width, sepal_len, color=colors[i])

    # Show the plot
    plt.show()

    # fig = plt.figure()
    # plt.title('Gaze angular error')
#     plt.plot(epoch_list, avg_MAE_test, color='b', label='test')
#     plt.plot(epoch_list, avg_MAE_val, color='g', label='val')

#     plt.legend()
#     # plt.locator_params(axis='x', nbins=30)

#     fig.savefig(os.path.join(evalpath,data_set+".png"), format='png')
