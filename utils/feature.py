#-*- coding:utf-8 -*-
"""
    创建于2020-05-31 14:34:32
    作者：lbb
    描述：对增强后的轴承数据进行特征提取
    其他：相关软件版本
          matplotlib:  3.1.3  
          numpy: 1.18.1  
          scipy: 1.4.1 
          pandas: 1.0.3
          sklearn: 0.22.1 
"""
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert
from scipy.fftpack import fft
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

def extract_feature(X, y,fs):
    """ 特征提取
    
        @param X: 数据样本
        @param y：数据标签
        @param fs：原始数据采样频率
        @return FX: 特征向量
        @return Fy: 标签
    
    example：

    from utils.augment import preprocess
    from utils.feature import extract_feature
    # -1- 载入数据
    path = r"./data/0HP"
    data_mark = "FE"
    len_data = 1024
    overlap_rate = 50 # 50%
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

    """
    
    def skewness(s) -> float:
        """ 
        偏度计算
        """
        N = len(s)
        s = np.ravel(s)
        mean = np.mean(s)
        rms = np.sqrt(np.dot(s,s)/ N)
        return np.sum(np.power(np.abs(s)-mean,3))/(N*rms**3)

    def maxf_in_env_spectrum(data, fs) -> float:
        """ 
        包络谱最大幅值处的频率
        """
        data = np.ravel(data)
        N = len(data)
        T = 1/fs
        analytic_signal = hilbert(data)
        am_enve = np.abs(analytic_signal).reshape(N,)              
        yf = fft(am_enve - np.mean(am_enve))
        y_envsp = 2.0/N * np.abs(yf[0:N//2]).reshape(N//2,1) 
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # 返回最大幅值的频率
        maxf = xf[np.argwhere(y_envsp == np.max(y_envsp))[0][0]]
        return  maxf
    
    def hist_for_entropy(s):
        """ 
        对信号的直方图计算
            
            @param s:一维序列数据
            @return res: 直方图每个组对应的高度
            @return s_min：s的最小值
            @return s_max：s的最大值
            @return ncell：直方图的分组数目
            
        """
        s = np.ravel(s)
        N = len(s)
        s_max = np.max(s)
        s_min = np.min(s)
        delt = (s_max - s_min) / N
        c_0 = s_min - delt / 2
        c_N = s_max + delt / 2
        ncell = int(np.ceil(np.sqrt(N)))

        # c = f(s)
        c = np.round((s - c_0) / (c_N - c_0) * ncell + 1/2)

        # 计算分组数组出现的频次
        res = np.zeros(ncell)
        for i in range(0, N):
            ind = int(c[i])
            if ind >= 1 and ind <= ncell:
                res[ind-1] = res[ind-1] + 1

        return res, s_min, s_max, ncell
    
    def shannom_entropy_for_hist(s) -> float:
        """ 
        一维序列的香农信号熵
        
            @param x: 一维序列数据
            @return estimate: 香农信号熵的无偏估计值
        """
        h,s_min, s_max, ncell = hist_for_entropy(s)
        # 无偏估计
        h = h[h!=0]
        N = np.sum(h)
        estimate = -np.sum(h*np.log(h)) / N
        sigma = np.sum(h*np.log2(h)**2)    
        sigma = np.sqrt((sigma/N - estimate**2) / (N - 1))
        estimate = estimate + np.log(N) + np.log((s_max-s_min)/ncell)
        nbias = -(ncell-1)/(2*N)
        estimate = estimate - nbias

        return estimate
    
    def pdf_for_median_am(s)  -> float: 
        """ 
        一维序列信号幅值中位数处的概率密度估计
            
            @param s: 一维序列信号
            @return 幅值中位数处的概率密度估计
        """
        N = len(s)
        res, s_min, s_max, ncell = hist_for_entropy(s)
        # 归一化的到概率密度
        pdf = res / N / (s_max - s_min) * ncell
        
        # 幅值中位数 映射 到直方图的组号
        delt = (s_max - s_min) / N
        c_min = s_min - delt / 2
        c_max = s_max + delt / 2
        
        s_median = np.median(s)
        s_median_icell = int(np.round((s_median - c_min) / (c_max - c_min) * ncell + 1/2))
        
        return  pdf[s_median_icell]
    
    feature = {}
    N = len(X[0])
    feature['mean'] = [np.mean(x) for x in X]
    feature['rms'] = [np.sqrt(np.dot(np.ravel(x),np.ravel(x))/ N) for x in X]
    feature['std'] = [np.std(x) for x in X]
    feature['skewness'] = [skewness(x) for x in X]
    feature['kurtosis'] = [kurtosis(x,fisher=False) for x in X]
    feature['maxf'] = [maxf_in_env_spectrum(x, fs) for x in X]
    feature['signal_entropy'] = [shannom_entropy_for_hist(x) for x in X]
    feature['am_median_pdf'] = [pdf_for_median_am(x) for x in X]
    feature['label'] = [int(la) for la in y]
    
    # 返回pandas.DataFrame类型的特征矩阵
    f_datafram = pd.DataFrame([feature[k] for k in feature.keys()], index=list(feature.keys())).T
    
    # 返回 FX，Fy 
    features = ['mean', 'rms', 'std', 'skewness', 'kurtosis', 'maxf', 'signal_entropy', 'am_median_pdf']
    FX, Fy = f_datafram[features], f_datafram['label'] 
    
    return FX, Fy

if __name__ == "__main__":
    from augment import preprocess
    from feature import extract_feature
    # -1- 载入数据
    path = r"./data/0HP"
    data_mark = "FE"
    len_data = 1024
    overlap_rate = 50 # 50%
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
    
