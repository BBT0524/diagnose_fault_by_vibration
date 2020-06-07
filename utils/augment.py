#-*- coding:utf-8 -*-
"""
    创建于2020-05-20 15:48:32
    作者：lbb
    描述：对原始轴承数据进行数据增强。
         参考的论文哈工大张伟硕士论文《基于卷积神经网络的轴承故障诊断算法研究》
"""

import os
from scipy.io  import loadmat
import numpy as np
import random
from sklearn.preprocessing  import scale, StandardScaler, MinMaxScaler
from collections import Counter

def iteror_raw_data(data_path,data_mark):
    """ 
       读取.mat文件，返回数据的生成器：标签，样本数据。
  
       :param data_path：.mat文件所在路径
       :param data_mark："FE" 或 "DE"                                                   
       :return iteror：（标签，样本数据）
    """  

    # 标签数字编码
    labels = {"normal":0, "IR007":1, "IR014":2, "IR021":3, "OR007":4,
         "OR014":5, "OR021":6, "B007":7, "B014":8, "B021":9}

    # 列出所有文件
    filenams = os.listdir(data_path)

    # 逐个对mat文件进行打标签和数据提取
    for single_mat in filenams: 
        
        single_mat_path = os.path.join(data_path, single_mat)
        # 打标签
        for key, _ in labels.items():
            if key in single_mat:
                label = labels[key]

        # 数据提取
        file = loadmat(single_mat_path)
        for key, _ in file.items():
            if data_mark in key:
                data = file[key]

        yield label, data

def data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs):
    """
        :param win_tlen: 滑动窗口的时间长度
        :param overlap_rate: 重叠部分比例, [0-100]，百分数；
                             overlap_rate*win_tlen*fs//100 是论文中的重叠量。
        :param fs: 原始数据的采样频率
        :param data_iteror: 原始数据的生成器格式
        :param kargs: {"norm"}
                norm  数据标准化的方式,三种选择：
                    1："min-max"；
                    2："Z-score", mean = 0, std = 1;
                    3： sklearn中的StandardScaler；
        :return (X, y): X, 切分好的数据， y数据标签
    """
    overlap_rate = int(overlap_rate)
    # 重合部分的时间长度，单位s
    overlap_tlen = win_tlen * overlap_rate / 100     
    # 步长，单位s
    step_tlen = win_tlen - overlap_tlen   
    # 滑窗采样增强数据           
    X = []
    y = []
    for iraw_data in data_iteror:   
        single_raw_data = iraw_data[1] 
        lab = iraw_data[0]  
        number_of_win = np.floor((len(single_raw_data) - overlap_tlen * fs)
                                  / (fs * step_tlen))
        for iwin in range(1,int(number_of_win)+1):
            # 滑窗的首尾点和其更新策略
            start_id = int((iwin - 1) * fs * step_tlen + 1)
            end_id = int(start_id + win_tlen * fs) 
            current_data = single_raw_data[start_id:end_id]
            current_label = lab
            X.append(current_data)							
            y.append(np.array(current_label))

    # 转换为np数组
    # X[0].shape == (win_tlen*fs, 1)
    # X.shape == (len(X), win_tlen*fs, 1)
    X = np.array(X)    
    y = np.array(y)
    
    for key, val in kargs.items():
        # 数据标准化方式选择
        if key == "norm" and val == "1":
            X = MinMaxScaler().fit_transform(X)
        if key == "norm" and val == "2":
            X = scale(X)
        if key == "norm" and val == "3":
            X = StandardScaler().fit_transform(X)

    return X, y

def under_sample_for_c0(X, y, low_c0, high_c0, random_seed):
    """ 使用非0类别数据的数目，来对0类别数据进行降采样。
        :param X: 增强后的振动序列
        :param y: 类别标签0-9
        :param low_c0: 第一个类别0样本的索引下标
        :param high_c0: 最后一个类别0样本的索引下标
        :param random_seed: 随机种子
        :return X,y
    """

    np.random.seed(random_seed)
    to_drop_ind = random.sample(range(low_c0, high_c0), (high_c0 - low_c0 + 1) - len(y[y==3]))    
    # 按照行删除    
    X = np.delete(X,to_drop_ind,0)
    y = np.delete(y,to_drop_ind,0)
    return X, y
    

def preprocess(path, data_mark, fs, win_tlen, 
               overlap_rate, random_seed, **kargs):

    data_iteror = iteror_raw_data(path, data_mark)
    X, y = data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs)
    # print(len(y[y==0]))

    # 降采样，随机删除类别0中一半的数据
    low_c0 = np.min(np.argwhere(y==0))
    high_c0 = np.max(np.argwhere(y==0))
    X, y = under_sample_for_c0(X, y, low_c0, high_c0, random_seed)
    # print(len(y[y==0]))
    
    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强和0类数据降采样后共有：{1}条,"           
           .format(fs,  X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
           .format(X.shape[1], int(overlap_rate*win_tlen*fs // 100)) )
    print("-> 类别数据数目:",  sorted(Counter(y).items()))
    return X, y

if __name__ == "__main__":    
    path = r"../data/0HP"
    data_mark = "FE"
    fs = 12000
    win_tlen = 2048/12000
    overlap_rate = (2047/2048)*100
    random_seed = 1
    X, y = preprocess(path,
                      data_mark, 
                      fs, 
                      win_tlen, 
                      overlap_rate, 
                      random_seed, 
                      norm=3)
