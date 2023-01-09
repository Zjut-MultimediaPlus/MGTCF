import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
# from utils import draw_scatter_graph
# import matplotlib.pyplot as mpl
import os
import pandas as pd

def allTraplot(tra):
    ftime = 0
    tra_x = tra[:,ftime,0]
    tra_y = tra[:, ftime, 1]
    fig = plt.figure()
    distance = np.sqrt(tra_x**2+tra_y**2)
    distance = np.sort(distance)
    r50 = distance[distance.shape[0]//2]
    r90 = distance[distance.shape[0]*9//10]
    rave = distance.mean()
    print(r50,r90)




    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('departure')
    ax.scatter(tra_x, tra_y,marker='.',s=20)
    ax.axis('equal')

    x1 = np.arange(-r50, r50, 0.0001)  # 点的范围
    y1 = np.sqrt(r50**2 - np.power(x1, 2))  # 上半个圆的方程
    x2 = np.arange(-r50, r50, 0.0001)
    y2 = -1 * np.sqrt(r50**2 - np.power(x2, 2))  # 下半个圆的方程
    plt.plot(x1, y1, x2, y2, color='k',linewidth=2.5)  # 画圆，黑色虚线

    x1 = np.arange(-r90, r90, 0.0001)  # 点的范围
    y1 = np.sqrt(r90**2 - np.power(x1, 2))  # 上半个圆的方程
    x2 = np.arange(-r90, r90, 0.0001)
    y2 = -1 * np.sqrt(r90**2 - np.power(x2, 2))  # 下半个圆的方程
    plt.plot(x1, y1, x2, y2, color='k',linewidth=2.5)  # 画圆，黑色虚线

    # x1 = np.arange(-rave, rave, 0.0001)  # 点的范围
    # y1 = np.sqrt(rave ** 2 - np.power(x1, 2))  # 上半个圆的方程
    # x2 = np.arange(-rave, rave, 0.0001)
    # y2 = -1 * np.sqrt(rave ** 2 - np.power(x2, 2))  # 下半个圆的方程
    # plt.plot(x1, y1, x2, y2, color='k', linewidth=1)  # 画圆，黑色虚线


    plt.annotate(r'$%.1f(50$' % r50 + '%)', xy=(np.sqrt(r50**2/2), np.sqrt(r50**2/2)), xytext=(+60, -60), textcoords='offset points', fontsize=22,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',linewidth=2.5))
    plt.annotate(r'$%.1f(90$' % r90 + '%)', xy=(np.sqrt(r90 ** 2 / 2), np.sqrt(r90 ** 2 / 2)), xytext=(+30, -30),
                 textcoords='offset points', fontsize=22,
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',linewidth=2.5))

    # addfx(plt)

    # 左右下上4个轴

    # 设置轴的位置
    ax.spines['left'].set_position(('data',0))
    # 设置轴的颜色
    ax.spines['right'].set_color('none')
    # 设置轴的位置
    ax.spines['bottom'].set_position(('data',0))
    # 设置轴的颜色
    ax.spines['top'].set_color('none')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

    pass

def addfx(plt):
    plt.text(140, -1, 'E', ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='top', fontdict=dict(fontsize=20, color='r',
                                     family='monospace',
                                     # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                                     weight='bold',
                                     # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                                     ))  # 字体属性设置)
    plt.text(-223, 1, 'W', ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'#y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
             va='bottom', fontdict=dict(fontsize=20, color='r',
                                     family='monospace',
                                     # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                                     weight='bold',
                                     # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                                     ))  # 字体属性设置)
    plt.text(-1, -110, 'S', ha='right',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='top', fontdict=dict(fontsize=20, color='r',
                                     family='monospace',
                                     # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                                     weight='bold',
                                     # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                                     ))  # 字体属性设置)
    plt.text(1, 143, 'N', ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='top', fontdict=dict(fontsize=20, color='r',
                                     family='monospace',
                                     # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                                     weight='bold',
                                     # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                                     ))  # 字体属性设置)

def allboxplot(tra):
    dist = np.sqrt(tra[:,:,0]**2+tra[:,:,1]**2)
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
    labels = '6', '12', '18', '24'  # 图例
    data = [dist[:,i] for i in range(4) if i not in [4,6]]
    plt.boxplot(data, labels=labels)  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.show()  # 显示图像

def instanceplot(tra,v):
    instanceList = [10.8,17.2,24.5,32.7,41.5,51.0,100]
    tylist = []
    thattimeIns = v[:,0]
    for i in range(len(instanceList)-1):
        mask1 = thattimeIns>=instanceList[i]
        mask2 = thattimeIns<instanceList[i+1]
        tylist.append(tra[mask1*mask2])
        # allTraplot(tra[mask1*mask2])
    instenceboxplot(tylist)

        # print(tylist)

    pass

def topv(pred_Me):
    # 0  经度  1纬度
    # pred_traj[:, :,0] = pred_traj[:, :,0] / 10 * 500 + 1300
    # pred_traj[:,:,1] = pred_traj[:,:,1] / 6 * 300 + 300
    # 0 气压 1 风速
    p = pred_Me[:, :, 0]*50+960
    v = pred_Me[:, :, 1] * 25 + 40
    return p,v

def toNE(pred_traj):
    # 0  经度  1纬度
    pred_traj[:, :,0] = pred_traj[:, :,0] / 10 * 500 + 1300
    pred_traj[:,:,1] = pred_traj[:,:,1] / 6 * 300 + 300
    # 0 气压 1 风速
    # p = pred_Me[:, :, 0]*50+960
    # v = pred_Me[:, :, 1] * 25 + 40
    return pred_traj

def calculatmin_max(tylist):
    for catigary in tylist:
        box = np.sort(catigary)
        cmin = box.min()
        c25 = box[int(box.size/4)]
        c50 = box[(int(box.size / 2))]
        c75 = box[int(box.size*0.75)]
        cmax = c75+(c75-c25)*1.5
        print(cmin,c50,cmax)




def instenceboxplot(tylist):
    # tylist = np.array(tylist)
    tylist = [np.sqrt(x[:,0, 0] ** 2 + x[:,0, 1] ** 2) for x in tylist]
    calculatmin_max(tylist)
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    # plt.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
    labels = 'TD', 'TS', 'STS', 'TY', 'STY', 'SurperTY'  # 图例
    # tylist = [tylist[:, i] for i in range(6)]
    tylist[0] = tylist[0][tylist[0]<200]
    plt.boxplot(tylist, labels=labels)  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('TC Intensity',fontsize=16)
    plt.ylabel('Absolute Error (km)',fontsize=16)
    plt.show()  # 显示图像

def pvplot(v):
    data = [v[:,i] for i in range(4) if i not in [4,6]]
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
    labels = '6', '12', '18', '24'  # 图例
    # tylist = [tylist[:, i] for i in range(6)]
    plt.boxplot(data, labels=labels)  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.show()  # 显示图像
    pass

def pvsubplot(pv,v):
    pv = pv[:,0]
    instanceList = [10.8, 17.2, 24.5, 32.7, 41.5, 51.0, 100]
    tylist = []
    thattimeIns = v[:, 0]
    for i in range(len(instanceList) - 1):
        mask1 = thattimeIns >= instanceList[i]
        mask2 = thattimeIns < instanceList[i + 1]
        tylist.append(pv[mask1 * mask2])
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
    labels = 'TD', 'TS', 'STS', 'TY', 'STY', 'SurperTY'  # 图例
    # tylist = [tylist[:, i] for i in range(6)]
    plt.boxplot(tylist, labels=labels)  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.show()  # 显示图像
    pass

def pvScatterPlot(pvdif,pv):
    pvdif = np.abs(pvdif[:,0])
    pv = pv[:,0]
    ux,s ,index,w= np.unique(pv,return_counts=True,return_index=True,return_inverse=True)
    ymean = []
    delet = []
    deletindex = []
    for i in range(ux.shape[0]):
        if len((pvdif[index==i]))>5:
            ymean.append(pvdif[index==i].mean())
        else:
            delet.append(i)
            if len(np.argwhere(pv==ux[i])) > 0:
                for ind in np.argwhere(pv==ux[i]):
                    deletindex.append(ind)
    ux = np.delete(ux,delet)
    pvdif = np.delete(pvdif,deletindex)
    pv = np.delete(pv, deletindex)
    xnew = np.linspace(ux.min(), ux.max(), 300)

    func = interp1d(ux, ymean, kind='quadratic')

    ynew = func(xnew)




    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('departure')
    plt.plot(xnew, ynew,color='r')
    ax.scatter(pv, pvdif, marker='.', s=16)
    plt.xlabel('Best-Track (m/s)',fontsize=18)
    plt.ylabel('Absolute Error (m/s)',fontsize=18)
    # plt.xlabel('Best-Track (hPa)',fontsize=18)
    # plt.ylabel('Absolute Error (hPa)',fontsize=18)
    plt.tick_params(labelsize=16)
    # plt.show()
    plt.savefig('error_intensity.pdf')
    pass

def individual(tpre,tgt,):
    pass

def polar(tra):
    tra6 = tra[:,0,:]
    xd = tra6[:,0]>0
    xx = tra6[:, 0] <= 0
    yd = tra6[:, 1] > 0
    yx = tra6[:, 1] <= 0
    NE = (xd*yd*1).sum()
    WN = (xx*yd*1).sum()
    SW = (xx * yx * 1).sum()
    ES = (xd * yx * 1).sum()
    tra12 = tra[:, 1, :]
    xd = tra12[:, 0] > 0
    xx = tra12[:, 0] <= 0
    yd = tra12[:, 1] > 0
    yx = tra12[:, 1] <= 0
    NE = ((xd * yd * 1).sum()+NE)/2
    WN = ((xx * yd * 1).sum()+WN)/2
    SW = ((xx * yx * 1).sum()+SW)/2
    ES = ((xd * yx * 1).sum()+ES)/2
    theta = np.linspace(0.25*np.pi,2.25*np.pi,4,endpoint=False)
    radii = np.array([NE,WN,SW,ES])
    print(radii)
    width = [0.5*np.pi,0.5*np.pi,0.5*np.pi,0.5*np.pi]

    ax = plt.subplot(111,projection='polar')
    bars = ax.bar(theta,radii,width=width,bottom=0)
    color = ['#F0614A','#5233F0','#F0D61A','#26F095']
    for r,bar,c in zip(radii,bars,color):
        bar.set_facecolor(c)
        bar.set_alpha(0.5)
    plt.tick_params(labelsize=18)
    plt.show()


    pass


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

    rangemin = np.min(x)
    rangemax = np.max(x)

    '''根据k、b求对应的点'''
    x_fit = np.linspace(rangemin,rangemax,1046)
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
    plt.scatter(x, y, c=z, cmap='rainbow', s=8)

    sample_size = len(x)
    corrcoef = np.corrcoef(x.T, y.T)[0, 1]
    bias = x - y
    bias_max = np.max(bias)
    bias_min = np.min(bias)
    bias_mean = np.mean(bias)
    bias_abs_mean = np.mean(np.abs(bias))
    bias_abs_min = np.min(np.abs(bias))
    bias_abs_max = np.max(np.abs(bias))
    std_bias = np.sqrt(np.square(bias-bias_mean).sum() / (sample_size - 1))
    rmse_bias = np.sqrt(np.square(bias).sum() / (sample_size))

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    #横纵坐标意义
    # plt.xlabel('Longitude')
    # plt.ylabel('Longitude Prediction')

    # 计算相关指标
    line_dis = 50
    linexxxx = 10
    line_start = rangemax
    raw_start = rangemin
    # plt.text(s='样本数量：%.5d' % sample_size,
    #          x=raw_start, y=line_start, fontsize=7, wrap=True, color='black')
    # plt.text(s='样本斜率：%.3f, 样本截距：%.3f' % (k, b),
    #          x=raw_start, y=(line_start-1*line_dis), fontsize=7, wrap=True, color='black')
    plt.text(s='SCC:  %3f' % corrcoef,
             x=raw_start+linexxxx, y=(line_start - 1*line_dis), fontsize=28, wrap=True, color='black')
    # 样本相关系数
    # plt.text(s='平均偏差：%.3f' % bias_mean,
    #          x=raw_start, y=(line_start-3*line_dis), fontsize=7, wrap=True, color='black')
    # plt.text(s='最大偏差：%.3f, 最小偏差：%3f' % (bias_max,bias_min),
    #          x=raw_start, y=(line_start - 4 * line_dis),fontsize=7, wrap=True, color='black')
    # plt.text(s='平均绝对偏差：%.3f,' % bias_abs_mean,
    #          x=raw_start, y=(line_start-5*line_dis), fontsize=7, wrap=True,color='black')
    # plt.text(s='最大绝对偏差：%.3f, 最小绝对偏差：%.3f' % (bias_abs_max, bias_abs_min),
    #          x=raw_start,y=(line_start - 6 * line_dis), fontsize=7, wrap=True, color='black')
    font = {'weight':'bold'}
    plt.text(s='SDE： %.3f' % (std_bias),
             x=raw_start+linexxxx, y=(line_start-2*line_dis), fontsize=28, wrap=True, color='black')
    # 误差标准差
    plt.text(s='RMSE：%.3f' % (rmse_bias),
             x=raw_start+linexxxx, y=(line_start - 3 * line_dis), fontsize=28, wrap=True, color='black')
    # 均方根误差

    plt.title(title)
    plt.xlim(rangemin,rangemax)
    plt.ylim(rangemin,rangemax)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.colorbar(fontsize=18)
    plt.savefig('%s.png'%title, dpi=600)
    plt.show()




def lagcor(lag_len=1):
    from scipy.stats import pearsonr,spearmanr
    obs_len = 12
    LONG_r	,LAT_r	,PRES_r	,WND_r = [],[],[],[]
    root_path = r'G:\data\informer_dataset\TY_BST_72'
    for year in range(1949,2021):
        file_path = os.path.join(root_path, str(year))
        ty_file_list = os.listdir(file_path)
        for ty in ty_file_list:
            ty_raw = pd.read_excel(os.path.join(file_path, ty))
            lens = ty_raw.shape[0]
            if lens < (obs_len + lag_len*2):
                continue
            sample_num = lens - (obs_len + lag_len*2)
            for id in range(sample_num):
                # 获取轨迹强度数据
                cols_data = ty_raw.columns[1:]
                df_data1 = ty_raw[cols_data][id:id + obs_len + lag_len].values
                df_data2 = ty_raw[cols_data][id+lag_len:id + obs_len + lag_len*2].values
                LONG1	,LAT1	,PRES1	,WND1 = df_data1[:,0],df_data1[:,1],df_data1[:,2],df_data1[:,3]
                LONG2, LAT2, PRES2, WND2 = df_data2[:,0], df_data2[:,1], df_data2[:,2], df_data2[:,3]

                LONG_r.append(spearmanr(LONG1,LONG2)[0])
                LAT_r.append(spearmanr(LAT1,LAT2)[0])
                PRES_r.append(spearmanr(PRES1,PRES2)[0])
                WND_r.append(spearmanr(WND1,WND2)[0])

                # LONG_r.append(pearsonr(LONG2, LONG1)[0])
                # LAT_r.append(pearsonr(LAT1, LAT2)[0])
                # PRES_r.append(pearsonr(PRES1, PRES2)[0])
                # WND_r.append(pearsonr(WND1, WND2)[0])

                # LONG1 = ty_raw['LONG'][id:id + obs_len + lag_len]
                # LONG2 = ty_raw['LONG'][id + lag_len:id + obs_len + lag_len * 2]
                # LONG_r.append(LONG2.corr(LONG1,method='pearson'))
                # WND1 = ty_raw['LONG'][id:id + obs_len + lag_len]
                # WND2 = ty_raw['LONG'][id + lag_len:id + obs_len + lag_len * 2]
                # WND_r.append(WND2.corr(WND1, method='pearson'))

    LONG_r = np.mean(np.array(LONG_r))
    LAT_r = np.mean(np.array(LAT_r))
    PRES_r = np.mean(np.array(PRES_r)[~np.isnan(np.array(PRES_r))])
    WND_r = np.mean(np.array(WND_r)[~np.isnan(np.array(WND_r))])
    print('LONG_r:',LONG_r)
    print('LAT_r:', LAT_r)
    print('PRES_r:', PRES_r)
    print('WND_r:', WND_r)
    print(LONG_r,LAT_r,PRES_r,WND_r)

if __name__ == '__main__':
    # lagcor()
    root = r'model_save/mgchooser_pipre5lr1e4_evn_envshare_noclip_trainall_relu_tripchkecl_gph'
    traj_path = os.path.join(root, 'trajectory.npy')
    pv_path = os.path.join(root, 'pvdif.npy')
    gt_path = os.path.join(root, 'gt.npy')
    tra = np.load(traj_path)
    gt = np.load(gt_path)
    pvdif = np.load(pv_path )
    # ne = np.load('ne.npy')
    p_gt, v_gt = topv(gt[:, :, 2:])
    p_diff = np.squeeze(pvdif[:,:,0])
    v_diff = np.squeeze(pvdif[:,:,1])
    # abspv = np.load('absPV.npy')
    # p_pre = abspv[:,:,0]
    # v_pre = abspv[:,:, 1]
    p,v = topv(pvdif)


    allTraplot(tra)
    allboxplot(tra)


    instanceplot(tra,v_gt)

    pvplot(pvdif[:,:,0])
    pvsubplot(pvdif[:,:,1],v_gt)
    pvScatterPlot(pvdif[:,:,0],p_gt)


    polar(tra)

    # tra_pre = toNE(tra)
    # tra_gt = toNE(gt[:,:,:2])

    # print(tra)
    # draw_scatter_graph(tra_gt[:,0,0].reshape(-1,1),ne[:,0,0].reshape(-1,1),'')
    # draw_scatter_graph(p_gt[:,3].reshape(-1,1),p_pre[:,3].reshape(-1,1),'')
    # draw_scatter_graph(tra_gt[:, 1, 1].reshape(-1, 1), ne[:, 1, 1].reshape(-1, 1), '222')




