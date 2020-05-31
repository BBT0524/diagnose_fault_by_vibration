#-*- coding:utf-8 -*-
"""
    创建于2020-05-31 14:40:32
    作者：lbb
    描述：对提取的特征进行分类器训练和模型的保存
    其他：相关软件版本
          matplotlib:  3.1.3  
          numpy: 1.18.1  
          scipy: 1.4.1 
          pandas: 1.0.3
          sklearn: 0.22.1 
          joblib: 0.14.1
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import joblib         # -> 用来保存模型

from utils.augment import preprocess
from utils.feature import extract_feature

# -1- 载入数据
path = r"./data/0HP"
data_mark = "FE"
len_data = 1024
overlap_rate = 50      # -> 50%
random_seed = 1 
fs = 12000

X, y = preprocess(path, 
                    data_mark, 
                    fs, 
                    len_data/fs, 
                    overlap_rate, 
                    random_seed ) 

# -2- 提取特征
FX, Fy = extract_feature(X, y, fs)

# -3- 数据集划分
x_train, x_test, y_train, y_test = train_test_split(FX, 
                                                    Fy, 
                                                    test_size=0.10, 
                                                    random_state=2)

# -4- 模型训练和保存

# K近邻
knn = make_pipeline(StandardScaler(),  
                     KNeighborsClassifier(3))
knn.fit(x_train, y_train)
# 保存Model(models 文件夹要预先建立，否则会报错)
joblib.dump(knn, 'models/knn.pkl')
print("KNN model saved in ./models")

score = knn.score(x_test, y_test) * 100
print("KNN score is: %.3f"%score, "in test dataset")

# 高斯分布的贝叶斯
nbg = make_pipeline(StandardScaler(),  
                    GaussianNB())
nbg.fit(x_train, y_train)

joblib.dump(nbg, 'models/GaussianNB.pkl')
print("GaussianNB model saved in ./models")

score = nbg.score(x_test, y_test) * 100
print("GaussianNB score is: %.3f"%score, "in test dataset")
# 随机森林
rfc = make_pipeline(StandardScaler(),
                    RandomForestClassifier(max_depth=6, random_state=0))
rfc.fit(x_train, y_train)

joblib.dump(rfc, 'models/RandomForest.pkl')
print("RandomForest model saved in ./models")

score = rfc.score(x_test, y_test) * 100
print("RandomForest score is: %.3f"%score,  "in test dataset")
