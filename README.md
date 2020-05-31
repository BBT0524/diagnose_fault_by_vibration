# 基于振动信号的滚动轴承故障诊断
## 1.介绍
毕设研究课题，根据轴承的振动数据信息来诊断轴承故障的位置和故障严重等级。方法思路走的是数据驱动，使用传统机器学习方法以及深度学习方法。这个开源项目做的是整理基于传统机器学习的轴承故障诊断的内容。\
主要分为三个部分： 
- 数据集预处理：数据集增强(utils.augment)
- 特征工程(utils.feature):均值(mean), 均方差(rms), 标准差(std), 偏度(skewness), 峭度(kurtosis), 包络谱最大幅值处频率(maxf), 信号熵(signal_entropy), 信号幅值中位数处概率密度值(am_median_pdf)
- 分类器训练和保存

## 2.在0HP上测试集score：
+ KNN score is: 90.295% in test dataset 
+ GaussianNB score is: 91.561% in test dataset 
+ RandomForest score is: 94.515% in test dataset
