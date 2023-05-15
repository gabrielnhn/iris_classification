from sklearn import datasets
import numpy as np
from matplotlib import pyplot



iris = datasets.load_iris() #Loading the dataset
iris.keys()

species = iris["target_names"]
feats = iris["feature_names"]

for feat in range(len(feats)):





# for feat in range(len(feats)):
    # fig = plt.figure()
    # plt.title('Gaze angular error')
#     plt.plot(epoch_list, avg_MAE_test, color='b', label='test')
#     plt.plot(epoch_list, avg_MAE_val, color='g', label='val')

#     plt.legend()
#     # plt.locator_params(axis='x', nbins=30)

#     fig.savefig(os.path.join(evalpath,data_set+".png"), format='png')
#     # plt.show()
