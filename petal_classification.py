from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as KNN

import numpy as np
from matplotlib import pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
import math

iris = datasets.load_iris() #Loading the dataset
iris.keys()

species = iris["target_names"]
feats = iris["feature_names"]
data = iris["data"]
targets = iris["target"]


# data = data[:, 0:2]
data = data[:, 2:]
# print(data)

x_train,x_test,y_train,y_test=train_test_split(data,targets,test_size=0.2,random_state=42)

# k = int(math.sqrt(len(data)))
k = 2
knn = KNN(k)

print("Fitting...")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

out = classification_report(y_test, y_pred)
print(out)


# Define the range of feature values for creating a grid of points
# x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x1_min, x1_max = 0,10
x2_min, x2_max = 0, 4

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))

# Predict the class labels for the grid points
Z = knn.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

# Plot the decision boundary
colors = "red", "green", "blue"

cmap = ListedColormap(colors)
plt.contourf(xx2, xx1, Z, alpha=0.5, cmap=cmap)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
plt.title('KNN Decision Boundary')
# plt.show()


data_per_class = {species[i]:[] for i in range(3)}
for i, a in enumerate(data):
    data_per_class[species[targets[i]]].append(a)

for x in data_per_class.keys():
    l = data_per_class[x] 
    data_per_class[x] = {}
    data_per_class[x]["data"] = np.array(l)

for i, s in enumerate(species):
    this_species = data_per_class[s]["data"]

    petal_width = this_species[:, 1]
    petal_len = this_species[:, 0]

    # plt.title(f"Petal size")

    plt.xlabel("Petal Width")
    plt.ylabel("Petal Length")

    plt.scatter(petal_width, petal_len, color=colors[i], label=s)

    plt.xlim(0, 3)
    plt.ylim(0, 8)

    # plt.savefig(f"{s}_petal.png", dpi=150)
    # plt.show()
plt.legend()
# plt.show()
plt.savefig("PETAL_KNN.png", dpi=200)
# In this example, we generate a synthetic classification dataset using make_classification and split it into training and testing sets. We train a support vector classifier (SVC) with a linear kernel on the training data. We define a range of feature values and create a grid of points using np.meshgrid. The model is then used to predict the class labels for the grid points, and a contour plot is created using plt.contourf to visualize the decision boundary. Finally, we scatter plot the original data points and display the plot using plt.show(). You can customize the code according to your specific dataset and model.


