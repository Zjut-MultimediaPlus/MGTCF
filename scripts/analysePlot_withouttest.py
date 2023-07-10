import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
import cv2
from matplotlib import animation
import matplotlib.image as img
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

from scipy.spatial import ConvexHull
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from matplotlib.backends.backend_agg import FigureCanvasAgg
# 引入 Image
import PIL.Image as Image
from matplotlib.patches import Rectangle


def getclosetra(i,gt,pre):
    pres = np.stack(pre)
    pres = pres[:,:,i,:]
    x = pres[:,:,0]-gt[:,0]
    y = pres[:,:,1]-gt[:,1]
    dist = x**2+y**2
    sumdist = np.sum(dist,axis=1)
    mindist = np.min(sumdist)
    mask = (sumdist==mindist)*1
    return mask


def getPicName(tyid):
    tyname = tyid[0]['new'][1]
    date = tyid[0]['new'][0]
    root = r'/data/hc/TYDataset/TY2019_img'
    datef = date[:-2]+'_'+date[-2:]
    filelist = os.listdir(os.path.join(root,tyname))
    for filename in filelist :
        if datef in filename and '.xml' not in filename:
            return os.path.join(root,tyname,filename)
    return 0

def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca() #获取当前子图，如果当前子图不存在，那就创建新的子图(get current ax)
    p = np.c_[x,y] #.c_功能类似于zip，不过不是生成组合的元祖，而是生成拼接起来的数组array
    hull = ConvexHull(p) #将数据集输入到ConvexHull中，自动生成凸包类型的对象
    poly = plt.Polygon(p[hull.vertices,:], **kw)
        #使用属性vertices调用形成凸包的点的索引，进行切片后，利用绘制多边形的类plt.Polygon将形成凸包的点连起来
        #这里的**kw就是定义函数的时候输入的**kw，里面包含了一系列可以在绘制多边形的类中进行调节的内容
        #包括多边形的边框颜色，填充颜色，透明度等等
    ax.add_patch(poly) #使用add_patch，将生成的多边形作为一个补丁补充到图像上


def plot_ty(saveroot,plotdir,plot_all=False):
    plot_data = np.load('plot_data_evngen.npy',allow_pickle=True).item()
    batch_len = len(plot_data['tyid'])
    # plot_dic = {'tyid':[],'gt_data_all':[],'pred_data':[],'gt_data_all_pv':[],'pred_data_pv':[]}
    plotCount =0
    for ty_batch_i in range(batch_len):
        tyID = plot_data['tyid'][ty_batch_i]
        aa = plot_data['gt_data_all'][ty_batch_i]
        pred_list = plot_data['pred_data'][ty_batch_i]
        meaa = plot_data['gt_data_all_pv'][ty_batch_i]
        pred_list_Me = plot_data['pred_data_pv'][ty_batch_i]


        color = ['#47F8FC','#8352FF','#38FF39','#FF741F','#FFEF2B','steelblue']
        for i in range(aa.shape[1]):
            name = tyID[i][0]['new'][1] + str(tyID[i][0]['new'][0])
            # if name != 'HALONG2019110800':
            #     continue
            if getPicName(tyID[i])==0:
                print(i)
                continue

            bgimg = img.imread(getPicName(tyID[i]))
            plotCount += 1
            fig = plt.figure(figsize=(10, 10))
            fig.figimage(bgimg)
            ax = fig.add_axes([0, 0, 1, 1])
            # ax.axesPatch.set_alpha(0.05)
            # ax.set_axisbelow(True)

            ax.set_xlim(80, 200)
            ax.set_ylim(-60, 60)
            traplot = [1,1,1,1,1,1]

            traplot = getclosetra(i,aa[8:12,i, :],pred_list)
            color = ['#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39']

            all_point = [aa[8,i][np.newaxis,:]]
            for j,pred_traj_fakex in enumerate(pred_list):
                all_point.append(pred_traj_fakex[:,i,:])
                if not plot_all:
                    if traplot[j] == 0:
                        continue
                out_a=pred_traj_fakex[:,i,:]
                # bb=np.concatenate((input_a[:, i, :].cuda().data.cpu().numpy(),out_a.cuda().data.cpu().numpy()),axis=0)
                bb = out_a
                # global x1,y1
                x1=bb[:,0]
                y1=bb[:,1]
                ax.plot(x1, y1, '*',markersize=5,color=color[j])


            # plt.show()
            # ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5)
            ax.plot(aa[:12,i, 0], aa[:12,i, 1], '.', color='red',markersize=5)
            # 覆盖区域===============
            all_point = np.concatenate(all_point,axis=0)
            encircle(all_point[:,0],all_point[:,1],ax,fc='#F52100', alpha=.5)

            ax.set_zorder(100)
            ax.set_axis_off()

            savePath = os.path.join(saveroot,plotdir)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            name = tyID[i][0]['new'][1]+str(tyID[i][0]['new'][0])
            plt.savefig(os.path.join(savePath,name+'.png'))
            # plt.show()
            plt.close()

def plot_ty_comparison(saveroot,plotdir,plot_all=False,crop=False):
    plot_data_env = np.load('plot_data_evngen.npy',allow_pickle=True).item()
    plot_data_MMSTN = np.load('plot_data_MMSTN.npy', allow_pickle=True).item()
    plot_data = np.load('plot_data_base.npy',allow_pickle=True).item()
    batch_len = len(plot_data['tyid'])
    # plot_dic = {'tyid':[],'gt_data_all':[],'pred_data':[],'gt_data_all_pv':[],'pred_data_pv':[]}
    plotCount =0
    for ty_batch_i in range(batch_len):
        tyID = plot_data['tyid'][ty_batch_i]
        aa = plot_data['gt_data_all'][ty_batch_i]
        pred_list = plot_data['pred_data'][ty_batch_i]
        pred_list_evngen = plot_data_env['pred_data'][ty_batch_i]
        pred_list_MMSTN = plot_data_MMSTN['pred_data'][ty_batch_i]
        # meaa = plot_data['gt_data_all_pv'][ty_batch_i]
        # pred_list_Me = plot_data['pred_data_pv'][ty_batch_i]


        color = ['#47F8FC','#8352FF','#38FF39','#FF741F','#FFEF2B','steelblue']
        for i in range(aa.shape[1]):
            name = tyID[i][0]['new'][1] + str(tyID[i][0]['new'][0])
            if name != 'LEKIMA2019080706':
                continue
            if getPicName(tyID[i])==0:
                print(i)
                continue


            traplot = [1,1,1,1,1,1]

            traplot = getclosetra(i,aa[8:12,i, :],pred_list_evngen)
            color = ['#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39']

            all_point = [aa[7,i][np.newaxis,:]]
            all_point_base = [aa[7, i][np.newaxis, :]]
            all_point_MMSTN = [aa[7, i][np.newaxis, :]]
            # all_point_MMSTN = []
            for j,(pred_traj_fakex,pred_traj_fakex_base,pred_traj_fakex_gen) in enumerate(zip(pred_list_evngen,pred_list,pred_list_MMSTN)):
                all_point.append(pred_traj_fakex[:,i,:])
                all_point_base.append(pred_traj_fakex_base[:,i,:])
                all_point_MMSTN.append(pred_traj_fakex_gen[:,i,:])
                if not plot_all:
                    if traplot[j] == 0:
                        continue
                out_a=pred_traj_fakex[:,i,:]
                # bb=np.concatenate((input_a[:, i, :].cuda().data.cpu().numpy(),out_a.cuda().data.cpu().numpy()),axis=0)
                bb = out_a
                # global x1,y1



            # plt.show()
            # ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5)

            # draw
            bgimg = img.imread(getPicName(tyID[i]))

            plotCount += 1

            if crop == True:
                size = 100
                bigsize = 5
                crop_imag, (colr, coll), (rowr, rowl) = img_crop(bgimg, aa[7, i][np.newaxis, :],size=size)
                crop_imag = cv2.resize(crop_imag,(size*2*bigsize,size*2*bigsize))
                fig = plt.figure(figsize=(size*2/100*bigsize, size*2/100*bigsize))
                fig.figimage(crop_imag)
                markersize = bigsize*5
            else:
                fig = plt.figure(figsize=(10, 10))
                xy = aa[7, i][np.newaxis, :]
                col = int((xy[0, 0] - 80) / 120 * 1000) - 100
                row = int((60 - xy[0, 1]) / 120 * 1000) - 100
                # bgimg = cv2.rectangle(bgimg,(col,row),(col+200,row+200),(1,0,0),thickness=3)
                # bgimg = cv2.line(bgimg,(col+200,row),(1000,0),(1,0,0),thickness=3)
                # bgimg = cv2.line(bgimg, (col + 200, row+200), (1000, 500), (1, 0, 0), thickness=3)
                fig.figimage(bgimg)
                colr, coll = 80, 200
                rowr, rowl = -60, 60
                markersize = 5

                # col = int((xy[0, 0] - 80) / 120 * 1000)-100
                # row = int((60 - xy[0, 1]) / 120 * 1000)-100
                # plt.gca().add_patch(Rectangle((col, row), 200, 200, linewidth=markersize, edgecolor='r', facecolor='none'))

            ax = fig.add_axes([0, 0, 1, 1])
            # ax.axesPatch.set_alpha(0.05)
            # ax.set_axisbelow(True)

            ax.set_xlim(colr,coll)
            ax.set_ylim(rowr,rowl)


            # 覆盖区域===============
            all_point = np.concatenate(all_point,axis=0)
            all_point_base = np.concatenate(all_point_base, axis=0)
            all_point_MMSTN = np.concatenate(all_point_MMSTN, axis=0)

            # FFED18,0000FF,38FF39
            # encircle(all_point_MMSTN[:, 0], all_point_MMSTN[:, 1], ax, fc='#FF0000', alpha=.6)
            # encircle(all_point_base[:, 0], all_point_base[:, 1], ax, fc='#0000FF', alpha=.55)
            #
            # encircle(all_point[:, 0], all_point[:, 1], ax, fc='#38FF39', alpha=.5)

            x1 = bb[:, 0]
            y1 = bb[:, 1]
            # ax.plot(x1, y1, '*', markersize=markersize, color='#00FF00')
            ax.plot(aa[:8, i, 0], aa[:8, i, 1], '.', color='red', markersize=markersize)




            ax.set_zorder(100)
            ax.set_axis_off()




            savePath = os.path.join(saveroot,plotdir)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            name = tyID[i][0]['new'][1]+str(tyID[i][0]['new'][0])
            # cv2.imwrite(os.path.join(savePath,name+'.png'),region)
            plt.savefig(os.path.join(savePath,name+'.png'))
            # plt.show()
            plt.close()

def img_crop(aaa,xy,size):
    col = int((xy[0, 0] - 80) / 120 * 1000)
    row = int((60 - xy[0, 1]) / 120 * 1000)
    region = aaa[row - size:row + size, col - size:col + size]
    colr,coll = (col - size)/1000*120+80,(col + size)/1000*120+80
    rowd,rowu = 60-(row - size)/1000*120,60-(row + size)/1000*120
    return region, (colr,coll),(rowu,rowd)

def plt2image(aaa,xy):
    # ax.set_xlim(80, 200)
    # ax.set_ylim(-60, 60)
    col = int((xy[0,0]-80)/120*1000)
    row = int((60-xy[0,1])/120*1000)
    canvas = FigureCanvasAgg(aaa)
    print(type(canvas))
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    # rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
    region = rgb_image[row-60:row+60,col-60:col+60]
    return region,(col-60,col+60),(row-60,row+60)
    # cv2.imshow('1',region)
    # cv2.waitKey(0)


def get_video(root):
    img_root = root
    file_list = os.listdir(img_root)
    ty_list = [file_name[:-14] for file_name in file_list]
    file_list = np.array(file_list)
    ty_list = np.array(ty_list)
    ty_name_list = np.unique(ty_list)
    for ty_name in ty_name_list:
        out_root = os.path.join(img_root,'video',ty_name+'.avi')
        fps = 2  # 帧率
        size = (1000, 1000)
        typath = sorted([os.path.join(root,x) for x in file_list[ty_list==ty_name]])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # 支持jpg
        videoWriter = cv2.VideoWriter(out_root, fourcc, fps, size)

        print(len(typath))
        for im_name in typath:
            _,x = os.path.splitext(im_name)
            if x != '.png':
                continue
            # string = os.path.join(img_root,im_name)
            print(im_name)
            frame = cv2.imread(im_name)
            frame = cv2.resize(frame, size)  # 注意这里resize大小要和视频的一样
            videoWriter.write(frame)

def get_left_down(im_name,left_down):

    env_data_path = '/home/hc/Desktop/hca6000/TYDataset/evn_data/norm_2019/2019'
    file_name = im_name.split('/')[-1].split('.')[0]
    text_name = file_name[:-10]
    text_date_str = file_name[-10:]
    env_data_path = os.path.join(env_data_path, text_name,text_date_str+'.npy')
    text_date = text_date_str[:4]+'/'+text_date_str[4:6]+'/'+text_date_str[6:8]+' '+ text_date_str[8:10]+':00'
    # text_intensity_list = ['Tropical Depression','Tropical Storm', 'Severe Tropical Storm',
    #                        'Typhoon', 'Severe Typhoon', 'Super Typhoon']
    text_intensity_list = ['TD','TS','STS','TY','STY','SuperTY']
    env_data = np.load(env_data_path,allow_pickle=True).item()
    wind = int(env_data['wind']*110)
    intensity = '('+text_intensity_list[np.argmax(env_data['intensity_class'])]+')'
    location_lon = str(round(env_data['location'][0]/10,2))+'°E'
    location_lat = str(round(env_data['location'][1] / 10,2)) + '°N'


    fontpath = "times.ttf"
    # 32为字体大小
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(left_down)
    draw = ImageDraw.Draw(img_pil)
    # 绘制文字信息
    # (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
    draw.text((10, 10), "TC Name: "+text_name, font=font, fill=(0, 0, 0))
    draw.text((10, 60), "TC Date: " + text_date, font=font, fill=(0, 0, 0))
    draw.text((10, 110), "TC Intensity: "+str(wind)+' m/s '+intensity, font=font, fill=(0, 0, 0))
    draw.text((10, 160), "TC Longitude: " + location_lon, font=font, fill=(0, 0, 0))
    draw.text((10, 210), "TC Latitude: " + location_lat, font=font, fill=(0, 0, 0))
    # draw.text((100, 350), "你好", font=font, fill=(255, 255, 255))
    bk_img = np.array(img_pil)
    # cv2.putText(left_down, text_name, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    # cv2.imshow('1', bk_img)
    # cv2.waitKey(0)
    return bk_img

def get_video_two(root_img,root_corp):
    img_root = root_img
    file_list = os.listdir(img_root)
    ty_list = [file_name[:-14] for file_name in file_list]
    file_list = np.array(file_list)
    ty_list = np.array(ty_list)
    ty_name_list = np.unique(ty_list)
    for ty_name in ty_name_list:
        out_root = os.path.join(img_root,'video_half_lable',ty_name+'.avi')
        fps = 2  # 帧率
        size = (1500, 1000)
        typath = sorted([os.path.join(root_img,x) for x in file_list[ty_list==ty_name]])
        ty_corp_path = sorted([os.path.join(root_corp, x) for x in file_list[ty_list == ty_name]])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # 支持jpg
        videoWriter = cv2.VideoWriter(out_root, fourcc, fps, size)

        print(len(typath))
        for im_name,corp_name in zip(typath,ty_corp_path):
            _,x = os.path.splitext(im_name)
            if x != '.png':
                continue
            # string = os.path.join(img_root,im_name)
            print(im_name)
            frame = cv2.imread(im_name)
            frame_crop = cv2.imread(corp_name)
            frame_crop = cv2.resize(frame_crop,(500,500))
            left_down = np.ones_like(frame_crop)*255
            left_down = get_left_down(im_name,left_down)
            crop_seg = np.concatenate([frame_crop,left_down],axis=0)
            integ = np.concatenate([frame,crop_seg],axis=1)
            # frame = cv2.resize(frame, size)  # 注意这里resize大小要和视频的一样
            videoWriter.write(integ)

if __name__ == '__main__':
    saveroot = 'plot'
    # plotdir = 'plot_single'
    # plot_ty(saveroot,plotdir)
    plotdir = 'plot_comparison_history_png'
    # plot_ty_comparison(saveroot, plotdir,crop=False)
    # get_video('plot/plot_comparison_two_png_noline_crop_afterdraw')
    get_video_two('plot/plot_comparison_two_png_noline_Rectangle_linehalf',
                  'plot/plot_comparison_two_png_noline_crop_afterdraw')

