import matplotlib
matplotlib.use('Agg')
#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-07-03 14:25
#

"""

""" 
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os

def tif2png(tif):
    # rst_file = '/root/PycharmProjects/ANNRK/AT_2016_year_2016.tif'
    raster = gdal.OpenShared(tif)
    data = raster.ReadAsArray()

    # cmap = plt.cm.jet
    png = tif.replace('tif', 'png')
    # plt.imsave(png, data, cmap=cmap)

    min = np.nanmin(data)
    max = np.nanmax(data)
    print('当前处理数据：', tif)
    print('最小值和最大值：',min,max)
    if min < max:
        try:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00',
                                                                                '#FF0000',
                                                                                '#8B0000'], 256)
            plt.imshow(data, cmap=cmap)
            plt.colorbar(ticks=[min, max])
            plt.savefig(png)
            plt.close()
        except:
            print('绘图失败！')
    else:
        print('最小值大于等于最大值，数据不对，无法绘图')
        return


folder = '/root/fusion_products/fusion_images/AT'
year_dir = os.listdir(folder)
for year in year_dir:
    in_folder = os.path.join(folder,year)
    tif_list = os.listdir(in_folder)
    for tif in tif_list:
        if tif.endswith('.tif'):
            tif_path = os.path.join(in_folder,tif)
            tif2png(tif_path)

# rst_file = '/root/PycharmProjects/ANNRK/AT_2016_year_2016.tif'
# raster = gdal.OpenShared(rst_file)
# data = raster.ReadAsArray()
#
# min = np.nanmin(data)
# max = np.nanmax(data)
#
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)
#
# plt.imshow(data, cmap=cmap)
# plt.colorbar(ticks=[min, max])
# plt.savefig('/tmp/test1.png')
# # plt.show()
