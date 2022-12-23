import pandas as pd
import numpy as np
import winreg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

house_price = pd.read_csv('HousingData.csv')

house_price = house_price.dropna().reset_index()

del house_price["index"]

train = house_price.drop(["MEDV"],axis=1)
target = house_price["MEDV"]

X_train, X_test, y_train, y_test=train_test_split(
    train,target,random_state=23)

X_train,X_test,y_train,y_test = train_test_split(train,target)
lasso = Lasso(alpha=0.5,max_iter=1000)
lasso.fit(X_train,y_train)

# print("Lasso训练模型得分："+str(r2_score(y_train,lasso.predict(X_train))))
# print("Lasso待测模型得分："+str(r2_score(y_test,lasso.predict(X_test))))

ridge = Ridge(alpha=0.5)
ridge.fit(X_train,y_train)
# print("Lasso训练模型得分："+str(r2_score(y_train,ridge.predict(X_train))))
# print("Lasso待测模型得分："+str(r2_score(y_test,ridge.predict(X_test))))

result=pd.DataFrame(columns=["参数","lasso训练模型得分","lasso待测模型得分","岭回归训练模型得分","岭回归待测模型得分"])
for i in range(1,100):
    alpha=i/10
    ridge=Ridge(alpha=alpha)
    lasso=Lasso(alpha=alpha,max_iter=10000)
    ridge.fit(X_train,y_train)
    lasso.fit(X_train,y_train)
    result=result.append(
        [{"参数":alpha,
          "lasso训练模型得分":r2_score(y_train,lasso.predict(X_train)),
          "lasso待测模型得分":r2_score(y_test,lasso.predict(X_test)),
          "岭回归训练模型得分":r2_score(y_train,ridge.predict(X_train)),
          "岭回归待测模型得分":r2_score(y_test,ridge.predict(X_test))}
         ])

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['SimHei','Arial']})#设定汉字字体，防止出现方框
# matplotlib inline
#在jupyter notebook上直接显示图表
fig= plt.subplots(figsize=(15,5))
plt.plot(result["参数"],result["lasso训练模型得分"],label="lasso训练模型得分")#画折线图
plt.plot(result["参数"],result["lasso待测模型得分"],label="lasso待测模型得分")
plt.plot(result["参数"],result["岭回归训练模型得分"],label="岭回归训练模型得分")
plt.plot(result["参数"],result["岭回归待测模型得分"],label="岭回归待测模型得分")
plt.rcParams.update({'font.size': 15})
plt.legend()
plt.xticks(fontsize=15)#设置坐标轴上的刻度字体大小
plt.yticks(fontsize=15)
plt.xlabel("参数",fontsize=15)#设置坐标轴上的标签内容和字体
plt.ylabel("得分",fontsize=15)
plt.show()