# diagnose_fault_by_vibration
## 介绍
毕设研究课题，根据轴承的振动数据信息来诊断轴承故障的位置和故障严重等级。方法思路走的是数据驱动，使用传统机器学习方法以及深度学习方法。\
这个开源项目做的是将其中传统机器学习的诊断方法进行整理。\
内容为三个部分： 
- 数据集预处理：数据集增强
- 特征工程
- 分类器训练和保存

## 在0HP上测试集score：
KNN score is: 90.295% in test dataset \
GaussianNB score is: 91.561% in test dataset \
RandomForest score is: 94.515% in test dataset
