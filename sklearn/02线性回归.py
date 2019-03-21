from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
import pandas as pd
from sklearn.metrics import  mean_squared_error as MSE
import matplotlib.pyplot as plt
housevalue = fch() #会需要下载，大家可以提前运行试试看
X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# 恢复索引
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)
print(reg.coef_)
print([*zip(Xtrain.columns, reg.coef_)])
print(MSE(yhat, Ytest))
print(cross_val_score(reg, X, y, cv=10,scoring="neg_mean_squared_error"))
from sklearn.metrics import r2_score
print(r2_score(yhat,Ytest))
print(r2_score(Ytest,yhat))
r2 = reg.score(Xtest,Ytest)
print(r2)

plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
plt.legend()
plt.show()