from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime

data = load_breast_cancer()
X = data.data
y = data.target
np.unique(y)
plt.scatter(X[:, 0], X[:, 1], c=y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
Kernel = ["linear", "rbf", "sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf = SVC(kernel=kernel, gamma="auto", cache_size=5000).fit(Xtrain, Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    # print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
