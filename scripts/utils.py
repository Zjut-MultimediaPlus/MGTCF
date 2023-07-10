#*************************###
"""                       ###
    本文件用于定义训练使用的
    一些辅助函数
"""                       ###
#*************************###
import os

import numpy as np
from pandas import Series
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def lr_decay(updater,epoch,lr_init):
    """
    按照当前训练的代数对lr进行衰减处理
    :param updater:输入的优化器
    :param epoch:当前训练的代数
    :param lr_init:初始的学习率
    :return:当前的lr
    """
    lr = lr_init*0.90**(epoch/10)
    for param_group in updater.param_groups:
        param_group['lr']=lr
    return lr

def kl (data_p,data_q):
    """
    计算两组数据之间的KL散度,log使用2为底数
    :param data_p: 原始数据，numpy数组
    :param data_q: 近似数据，numpy数组
    :return:KL散度 numpy 数组
    """
    return np.sum(data_p*np.log2(data_p/data_q))

def euclidean(data_p,data_q):
    """
    计算两组数据之间的欧式距离
    :param data_p:数据1，numpy数组
    :param data_q:数据2，numpy数组
    :return:
    """
    return np.sum((data_p-data_q)**2)

def draw_scatter_graph(x,y,title):
    """
    绘散点图并计算偏差相关数据
    :param x: x数据
    :param y: y数据
    :param title: 图表标题
    :return: NOne
    """

    '''线性回归求拟合直线k、b'''
    reg = LinearRegression().fit(x,y)
    k = reg.coef_
    b = reg.intercept_

    '''根据k、b求对应的点'''
    x_fit = np.linspace(265,315,1000)
    y_fit = (x_fit*k+b).ravel()

    '''根据xy坐标位置求密度'''
    xy = np.vstack([x.reshape(1,-1),y.reshape(1,-1)])
    z = gaussian_kde(xy)(xy)
    z*=len(x)

    '''从低到高排列，确保密度高的后打印'''
    idx = np.argsort(z)
    x,y,z = x[idx],y[idx],z[idx]

    '''画直线和点'''

    plt.plot(x_fit,y_fit,c='black')
    #plt.scatter(x,y,c=z,cmap='Spectral',s=2)
    plt.scatter(x, y, c=z, cmap='rainbow', s=2)

    sample_size = len(x)
    corrcoef = np.corrcoef(x.T, y.T)[0, 1]
    bias = x - y
    bias_max = np.max(bias)
    bias_min = np.min(bias)
    bias_mean = np.mean(bias)
    bias_abs_mean = np.mean(np.abs(bias))
    bias_abs_min = np.min(np.abs(bias))
    bias_abs_max = np.max(np.abs(bias))
    std_bias = np.sqrt(np.square(bias).sum() / (sample_size - 1))
    rmse_bias = np.sqrt(np.square(bias).sum() / (sample_size))

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    #横纵坐标意义
    plt.xlabel('sst_obs')
    plt.ylabel('sst_sat/sst_corrected')

    # 计算相关指标
    line_dis = 1.6
    line_start = 308
    raw_start = 271
    plt.text(s='样本数量：%.5d' % sample_size,
             x=raw_start, y=line_start, fontsize=7, wrap=True, color='black')
    plt.text(s='样本斜率：%.3f, 样本截距：%.3f' % (k, b),
             x=raw_start, y=(line_start-1*line_dis), fontsize=7, wrap=True, color='black')
    plt.text(s='样本相关系数：%4f' % corrcoef,
             x=raw_start, y=(line_start - 2*line_dis), fontsize=7, wrap=True, color='black')
    plt.text(s='平均偏差：%.3f' % bias_mean,
             x=raw_start, y=(line_start-3*line_dis), fontsize=7, wrap=True, color='black')
    plt.text(s='最大偏差：%.3f, 最小偏差：%3f' % (bias_max,bias_min),
             x=raw_start, y=(line_start - 4 * line_dis),fontsize=7, wrap=True, color='black')
    plt.text(s='平均绝对偏差：%.3f,' % bias_abs_mean,
             x=raw_start, y=(line_start-5*line_dis), fontsize=7, wrap=True,color='black')
    plt.text(s='最大绝对偏差：%.3f, 最小绝对偏差：%.3f' % (bias_abs_max, bias_abs_min),
             x=raw_start,y=(line_start - 6 * line_dis), fontsize=7, wrap=True, color='black')
    plt.text(s='误差标准差：%.3f, 误差rmse：%.3f' % (std_bias, rmse_bias),
             x=raw_start, y=(line_start-7*line_dis), fontsize=7, wrap=True, color='black')

    plt.title(title)
    plt.xlim(270,310)
    plt.ylim(270,310)
    plt.colorbar()
    plt.savefig('./img/%s.png'%title, dpi=600)
    plt.show()

def draw_distr_graph(y_data,y_pred,y_sat):
    """
    用于绘制数据分布图
    :param y_data: 真值数据(ndarray)
    :param y_pred: 预测数据(ndarray)
    :param y_sat:  卫星数据(ndarray)
    :return: None
    """
    s1 = Series(y_data.ravel())
    s1.plot(kind='kde', c='deepskyblue', alpha=0.8, label='obs')
    s2 = Series(y_pred.ravel())
    s2.plot(kind='kde', c='darkcyan', alpha=0.8, label='pred')
    s3 = Series(y_sat.ravel())
    s3.plot(kind='kde', c='tomato', alpha=0.8, label='sat')
    plt.xlabel('sst')
    plt.ylabel('density')
    plt.title('Data Distribution', fontsize='large', fontweight='bold')
    plt.legend()
    plt.savefig('./img/distribution.png', dpi=600)
    plt.show()

def draw_training_change(train_data_set,eval_data_set,lable,out_path,ylim_min,ylim_max):
    """
    绘制损失值或误差值随训练过程的变化
    :param train_data_set: 训练集数据(ndarray)
    :param eval_data_set: 验证集数据(ndarray)
    :param lable: 图像标题(str)
    :param out_path: 图像保存路径和(str)
    :param ylim_min: y轴显示下限
    :param ylim_max: y轴显示上限
    :return: None
    """
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(range(int((len(train_data_set)))), train_data_set, 'r-', lw=0.5, alpha=0.5, label='训练集')
    plt.plot(range(int((len(eval_data_set)))), eval_data_set, 'b', lw=0.5, alpha=0.5, label='验证集')
    plt.title(label=lable)
    plt.ylim(ylim_min, ylim_max)
    plt.legend()
    plt.savefig(out_path, dpi=600)
    plt.show()

def cal_statistic(x,y):

    sample_size = len(x)
    '''线性回归求拟合直线k、b'''
    reg = LinearRegression().fit(x, y)
    k = reg.coef_
    b = reg.intercept_

    corrcoef = np.corrcoef(x.T,y.T)[0,1]
    bias = x-y
    bias_max = np.max(bias)
    bias_min = np.min(bias)
    bias_mean = np.mean(bias)
    bias_abs_mean = np.mean(np.abs(bias))
    std_bias = np.sqrt(np.square(bias).sum()/(sample_size-1))
    rmse_bias = np.sqrt(np.square(bias).sum()/(sample_size))

    print('样本数量：%5d'%sample_size)
    print('样本斜率：%4f, 样本截距：%4f, 样本相关系数：%4f'%(k,b,corrcoef))
    print('平均偏差：%4f, 最大偏差：%4f, 最小偏差：%f, 平均绝对偏差：%f'%(bias_mean,bias_max,bias_min,bias_abs_mean))
    print('误差标准差：%4f, 误差rmse：%4f'%(std_bias,rmse_bias))

def cal_time(stamp1,stamp2):
    """
    传入两个时间戳，并计算两者绝对偏差
    :param stamp1:时间戳1
    :param stamp2:时间戳2
    :return:
    """
    results = {}
    d_value = abs(stamp1-stamp2)
    results['sec'] = d_value % 60
    results['min'] = (d_value / 60) % 60
    results['hour'] = d_value/3600
    results['d_value'] = d_value
    return results
