import matplotlib
matplotlib.use('Agg')
#
# -*-coding:utf-8 -*-
#
# @Module:数据融合单元 S_BDA_PDD_SCI_002-002-003

# @Author: zhaojianghua
# @Date  : 2018-02-22 10:06
#

"""
根据多要素相关性分析单元选择相关性较高的特征数据（栅格数据格式）对气象监测数据进行融合处理
dependencies:
dnf install python3-matplotlib
pip install pykrige
"""

import numpy as np
from pykrige.ok import OrdinaryKriging
import sys
from sklearn import preprocessing
import gdal, ogr, osr
import sklearn.metrics.regression
from math import sqrt
import pandas as pd
from pandas import DataFrame
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

# from raster_reader import Monitordata, Relateddata, get_related_data

def point_geotransform(inproj, outproj, x, y):
    '''
    坐标投影变换
    :param inproj: 输入投影
    :param outproj: 输出投影
    :param x: 经度或者横坐标
    :param y: 纬度或者纵坐标
    :return: 投影变换后的坐标点
    '''
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.SetFromUserInput(inproj)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.SetFromUserInput(outproj)
    transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # point coordinate transformation
    geom = ogr.Geometry(ogr.wkbPoint)
    geom.AddPoint(x, y)
    geom.Transform(transform)
    x_c = geom.GetX()
    y_c = geom.GetY()
    return x_c, y_c


def prepare_data(obs_data, proj,feat):
    '''
    数据准备：将位置信息进行投影转换
    :param obs_data: 监测数据
    :param proj: 投影信息
    :return: 预处理后的监测数据
    '''
    daily_df = obs_data
    # daily_df = pd.read_csv(obs_data)
    station_num = daily_df.shape[0]

    data = np.array([])
    for i in range(station_num):
        station_id = daily_df.iloc[i].STATION
        at_mean = daily_df.iloc[i][feat]

        # use loc to ge series wichi satisfying condition, and then iloc to get first element
        latitude =daily_df.iloc[i]['LATITUDE']
        longitude = daily_df.iloc[i]['LONGITUDE']

        # transform wgs84 to projected coordinate system
        x_c, y_c = point_geotransform("EPSG:4326", proj, longitude, latitude)

        data = np.append(data, [x_c, y_c, at_mean])

    data = data.reshape(-1, 3)
    return data


def array_to_tif(array, dst_tif, dst_proj, ref_gt, pixel_width, pixel_height):
    '''
    将二维数组存储成tif格式
    :param array: 二维数组
    :param dst_tif: tif格式输出结果的文件路径
    :param dst_proj: 输出图像的投影
    :param ref_gt: 参考的geotransform参数
    :param pixel_width: 像素宽度
    :param pixel_height: 像素高度
    :return:无
    '''
    # array original is (xmin, ymin), should be transformed into original (xmin, ymax)
    # or the gt should be revised as [minx, pixel_width, 0.0, miny, 0.0, pixel_height
    row = array.shape[0]
    # print(row)
    for r in range(int(row / 2) - 1):
        array[[r, row - r - 1], :] = array[[row - r - 1, r], :]

    # output the array in geotiff format
    xsize = array.shape[0]
    ysize = array.shape[1]
    dst_format = 'GTiff'
    dst_nbands = 1
    dst_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_tif, ysize, xsize, dst_nbands, dst_datatype)
    gt = [ref_gt[0] - 0.5 * pixel_width, pixel_width, 0.0, ref_gt[3] + 0.5 * pixel_height, 0.0, (-1) * pixel_height]
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(dst_proj)
    # dst_ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    dst_ds.GetRasterBand(1).WriteArray(array)


def ordinary_kriging(data, gridx, gridy, variogram_model):
    '''
    普通克里金方法
    :param data: 待插值数据
    :param gridx: 插值后横坐标数组
    :param gridy: 插值后纵坐标数组
    :param variogram_model: 变异函数
    :return: 插值结果
    '''
    # import pykrige.kriging_tools as kt
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model=variogram_model, verbose=False,
                         enable_plotting=False)
    z, ss = OK.execute('grid', gridx, gridy)
    return z.data


def grid_generation(shp_studyarea, pixel_width, pixel_height):
    '''
    生成插值格网
    :param shp_studyarea: 结果范围
    :param pixel_width: 像素宽度
    :param pixel_height: 像素高度
    :return: 格网的横纵坐标数组
    '''
    ds = ogr.OpenShared(shp_studyarea)
    layer = ds.GetLayer()
    # feat = layer.GetFeature(0)
    # geom = feat.GetGeometryRef().ExportToWkt()
    # init_proj = layer.GetSpatialRef().ExportToWkt()

    # coordinate system transformation
    # geom_proj = GeomTrans(init_proj, proj).transform_geom(geom)
    # minx, maxx, miny, maxy = geom_proj.GetEnvelope()

    extent = layer.GetExtent()  # (364541.09976471704, 542955.54090274, 4365990.607469364, 4545397.762843872)minx, maxx, miny, maxy
    minx = extent[0]
    maxx = extent[1]
    miny = extent[2]
    maxy = extent[3]

    # Creates the kriged grid and the variance grid
    # grid of points, on a masked rectangular grid of points, or with arbitrary points
    gridx = np.arange(minx, maxx, pixel_width)
    gridy = np.arange(miny, maxy, pixel_height)
    return gridx, gridy


def train_ann(x_train, y_train, x_test, y_test):
    '''
    神经网络模型训练
    :param x_train: 训练样本X
    :param y_train: 训练样本Y
    :param x_test: 测试数据X
    :param y_test: 测试数据Y
    :return: 神经网络模型
    '''
    clf = MLPRegressor(hidden_layer_sizes=40, activation='logistic', solver='lbfgs', alpha=0.0001, max_iter=200)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    r2_score = sklearn.metrics.regression.r2_score(y_test, y_pred)
    explained_variance_score = sklearn.metrics.regression.explained_variance_score(y_test, y_pred)
    median_absolute_error = sklearn.metrics.regression.median_absolute_error(y_test, y_pred)
    mean_squared_error = sklearn.metrics.regression.mean_squared_error(y_test, y_pred)
    mean_absolute_error = sklearn.metrics.regression.mean_absolute_error(y_test, y_pred)

    # print "ann: r2_score :", r2_score
    # print "ann: explained_variance_score:", explained_variance_score
    # print "ann: median_absolute_error:", median_absolute_error
    # print "mean_squared_error:", mean_squared_error
    # print "mean_absolute_error:", mean_absolute_error

    return clf


def process_and_split_samples(obs_data, frac):
    '''
    样本数据处理与划分
    :param obs_data: 监测数据
    :param frac: 数据划分比例
    :return: 训练样本、测试样本
    '''
    # # 当obs_data是原始的csv监测数据文件时
    # daily_at = pd.read_csv(obs_data, usecols=["STATION", "ODATE", 'LONGITUDE', 'LATITUDE', "ELEVATION", "AT"])
    # grouped = daily_at.groupby("STATION")
    # result = grouped.agg({'AT': 'mean', 'LATITUDE': 'first', 'LONGITUDE': 'first', 'ELEVATION': 'first'})
    # obs_data直接是dataframe，而不是csv文件
    result = obs_data

    # remove rocords with nan feature values
    # result = result.dropna(axis=0, how='any')
    whole_count = result.shape[0]
    shuffle_df = shuffle(result)

    split_line = int(whole_count * frac)
    train_df = shuffle_df.iloc[:split_line, :]
    test_df = shuffle_df.iloc[split_line + 1:, :]
    # train_df_file = "/mnt/win/tmp/train_20100101.csv"
    # test_df_file = "/mnt/win/tmp/test_20100101.csv"
    #
    # train_df.to_csv(train_df_file)
    # test_df.to_csv(test_df_file)
    # return train_df_file, test_df_file
    return train_df, test_df

def split_dataset(x, y, frac):
    '''
    input x,y are ndarrays, output are ndarrays either, frac the percent of the training dataset, between 0 and 1, 0.8 for example
    :param x: x数组
    :param y: y数组
    :param frac: 训练样本的划分比例，介于0-1之间
    :return: 数组
    '''
    whole_count, x_col = x.shape
    x_df = DataFrame(x)
    y_df = DataFrame(y)
    whole_dt = pd.merge(x_df, y_df, left_index=True, right_index=True)
    shuffle_dt = shuffle(whole_dt)
    shuffle_dt = shuffle_dt.reset_index(drop=True)

    # split dataset into training and testing
    split_line = int(whole_count * frac)
    x_train = shuffle_dt.iloc[:split_line, :x_col]
    y_train = shuffle_dt.iloc[:split_line, -1]
    x_test = shuffle_dt.iloc[split_line + 1:, :x_col]
    y_test = shuffle_dt.iloc[split_line + 1:, -1]

    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test


def data_regulazition(x):
    '''
    数据归一化
    :param x: 数组数据
    :return: 归一化后的数据
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    ones_col = np.ones(x_scaled.shape[0])
    x_scaled = np.c_[ones_col, x_scaled]
    return x_scaled


def data_prepare_for_ann(obs_data,feat):
    '''
    神经网络训练数据准备
    :param obs_data: 监测数据
    :return:预处理数据及训练和测试样本数据集
    '''
    # daily_at = pd.read_csv(obs_data, usecols=["STATION", "ODATE", 'LONGITUDE', 'LATITUDE', "ELEVATION", "AT"])
    # grouped = daily_at.groupby("STATION")
    # result = grouped.agg({'AT': 'mean', 'LATITUDE': 'first', 'LONGITUDE': 'first', 'ELEVATION': 'first'})
    #
    # # remove rocords with nan feature values
    # result = result.dropna(axis=0, how='any')
    daily_at = obs_data
    # daily_at = pd.read_csv(obs_data)

    x_df = DataFrame(daily_at, columns=['LONGITUDE', 'LATITUDE', 'ELEVATION'])
    y_df = DataFrame(daily_at, columns=[feat])

    print('X data', x_df)
    # add a column of ones into the x
    x = x_df.values
    y = y_df.values
    print('x values', x)
    x_scaled = data_regulazition(x)
    print('x scaled', x_scaled)
    x_train, y_train, x_test, y_test = split_dataset(x_scaled, y, 0.8)
    return x_df, y_df, x_train, y_train, x_test, y_test


def read_value_fcoord(raster_file, x_c, y_c):
    '''
    根据坐标信息读取栅格数据对应坐标位置的数据
    :param raster_file: 栅格数据
    :param x_c: 横坐标
    :param y_c: 纵坐标
    :return: None或者数值
    '''
    raster = gdal.OpenShared(raster_file)
    if raster is None:
        print("Failed to open file:" + raster_file)
        sys.exit()
    gt = raster.GetGeoTransform()

    xsize = raster.RasterXSize
    ysize = raster.RasterYSize

    col = int((x_c - gt[0]) / gt[1])
    row = int((y_c - gt[3]) / gt[5])
    # the value at location col and row is the left-top corner value.
    if col < xsize and col > 0 and row < ysize and row > 0:
        value = raster.ReadAsArray(col, row, 1, 1)
        return value[0][0]
    else:
        return None


def grid_feature(gridx, gridy, grid_proj, feat_file):
    '''
    根据格网位置读取对应的栅格数据的数值
    :param gridx: 横坐标数组
    :param gridy: 纵坐标数组
    :param grid_proj: 格网投影
    :param feat_file: 栅格数据
    :return: 格网数据
    '''
    feat_rst = gdal.OpenShared(feat_file)
    feat_proj = feat_rst.GetProjectionRef()
    grid_X = []
    for x in gridx:
        for y in gridy:
            x_c, y_c = point_geotransform(grid_proj, feat_proj, x, y)
            feat = read_value_fcoord(feat_file, x_c, y_c)
            # print feat
            if feat is None or feat == 0:
                continue
            grid_X.append([x, y, feat])
    return np.array(grid_X)


def ann_predict(clf, grid_X, grid_proj):
    '''
    神经网络预测
    :param clf: 神经网络模型
    :param grid_X: X数据
    :param grid_proj: 投影
    :return: 神经网络回归数据
    '''
    grid_X_wgs = []
    for i in range(grid_X.shape[0]):
        x = grid_X[i][0]
        y = grid_X[i][1]
        # xycs_trans = GeomTrans(grid_proj, "EPSG:4326")
        # lon, lat = xycs_trans.transform_point([x, y])

        lon, lat = point_geotransform(grid_proj, "EPSG:4326", x, y)
        grid_X_wgs.append([lon, lat, grid_X[i][2]])
    grid_X_wgs = np.array(grid_X_wgs)

    # add a first column of 1s
    x_scaled = data_regulazition(grid_X_wgs)

    spatial_trend = clf.predict(x_scaled)
    # reformat spatial_trend list into grid array formate
    return spatial_trend


def records_to_grids(cs, records, gridx, gridy, pixel_width, pixel_height):
    '''
    一维数组转换为格网数据
    :param cs:格网特征数据
    :param records:神经网络计算后的趋势数据
    :param gridx:格网X坐标数组
    :param gridy:格网Y坐标数组
    :param pixel_width:像素宽度
    :param pixel_height:像素高度
    :return:格网数据
    '''
    grid_row = len(gridy)
    grid_col = len(gridx)
    # spatial_trend_grid = np.empty((grid_row, grid_col))
    # spatial_trend_grid[:] = np.nan
    spatial_trend_grid = np.empty((grid_row, grid_col)) * np.nan

    for i in range(len(cs)):
        col = int((cs[i][0] - gridx[0]) / pixel_width)
        row = int((cs[i][1] - gridy[0]) / pixel_height)
        # print(cs[i][0], cs[i][1], row, col, records[i])
        spatial_trend_grid[row][col] = records[i]

    return spatial_trend_grid


def accuracy_assessment(sum_array, obs_data, ref_proj, gridx, gridy, pixel_width, pixel_height, feat):
    '''
    精+度评估
    :param sum_array: 计算结果
    :param obs_data: 实际结果
    :param ref_proj: 参考数据的投影
    :param gridx:格网横坐标数组
    :param gridy:格网纵坐标数组
    :return:评价结果
    '''
    test_data = prepare_data(obs_data, ref_proj, feat)
    # y_test = list(sample_data[:,2])
    y_test = []
    y_pred = []
    for sample in test_data:
        # print(sample)
        row = int((sample[1] - gridy[0]) / pixel_height)
        col = int((sample[0] - gridx[0]) / pixel_width)
        pred = sum_array[row][col]
        # print(sample[2], pred)
        if np.isnan(pred):
            continue
        else:
            y_test.append(sample[2])
            y_pred.append(pred)

    # with open("/mnt/win/tmp/test.txt", "w") as output:
    #     output.write(str(y_test))
    #     output.write('/n%s' % str(y_pred))
    r2_score = sklearn.metrics.regression.r2_score(y_test, y_pred)
    explained_variance_score = sklearn.metrics.regression.explained_variance_score(y_test, y_pred)
    median_absolute_error = sklearn.metrics.regression.median_absolute_error(y_test, y_pred)
    mean_squared_error = sklearn.metrics.regression.mean_squared_error(y_test, y_pred)
    mean_absolute_error = sklearn.metrics.regression.mean_absolute_error(y_test, y_pred)

    return r2_score, explained_variance_score, median_absolute_error, sqrt(mean_squared_error), mean_absolute_error


def data_prepare_for_kriging(spatial_trend, data, gridx, gridy, pixel_width, pixel_height):
    '''
    克里金插值数据准备
    :param spatial_trend: 已有数据
    :param data: 插值数据
    :param gridx: 格网横坐标数组
    :param gridy: 格网纵坐标数组
    :return: 数组数据
    '''
    residual_data = np.array([])

    for sample in data:
        # print(sample)
        row = int((sample[1] - gridy[0]) / pixel_height)
        col = int((sample[0] - gridx[0]) / pixel_width)
        pred = spatial_trend[row][col]
        residual = sample[2] - pred
        # print(sample[2], pred)
        if np.isnan(pred):
            continue
        else:
            residual_data = np.append(residual_data, [sample[0], sample[1], residual])
    residual_data = residual_data.reshape(-1, 3)
    return residual_data


class Datafusion:
    def __init__(self, monitordata, relatedata):
        self.monitordata = monitordata
        self.relatedata = relatedata

    def annrk_datafusion(self, reference_map, china_boundary, pixel_width, pixel_height, dst_tif,feat):
        # reference image
        # ref_raster = "/mnt/win/image/eco10_forestquality_2010.img"
        if self.monitordata.shape[0]==0:
            print('无监测数据。。')
            return
        ref_rst = gdal.OpenShared(reference_map)
        if ref_rst is None:
            print("参考数据读取失败:" + reference_map)
            sys.exit()

        ref_proj = ref_rst.GetProjectionRef()
        ref_gt = ref_rst.GetGeoTransform()

        # step1, determine the final data boundary and resolution
        gridx, gridy = grid_generation(china_boundary, pixel_width, pixel_height)

        # step2, extract related features for each grid cell

        grid_X = grid_feature(gridx, gridy, ref_proj, self.relatedata)

        # step3, train ann model, and get the spatial trend for all grid cells
        # obs_data = "/mnt/win/tmp/2010-1-1.csv"

        variogram_model = 'linear'
        train_data, test_data = process_and_split_samples(self.monitordata, 0.9)
        print("训练数据如下：\n", train_data)
        x_df, y_df, x_train, y_train, x_test, y_test = data_prepare_for_ann(train_data, feat)
        clf = train_ann(x_train, y_train, x_test, y_test)

        spatial_trend = ann_predict(clf, grid_X, ref_proj)
        # spatial_trend should transform from records to grids, some cells do not have ann predicted values
        assert len(grid_X) == len(spatial_trend)
        spatial_trend_grid = records_to_grids(grid_X, spatial_trend, gridx, gridy, pixel_width, pixel_height)

        # step4, calculate residual data for spatial_trend grid cells which hav observation points
        data = prepare_data(train_data, ref_proj,feat)
        residual_data = data_prepare_for_kriging(spatial_trend_grid, data, gridx, gridy, pixel_width, pixel_height)

        # step5, interplate residual data by kriging and get residual results for each grid cell
        res_array = ordinary_kriging(residual_data, gridx, gridy, variogram_model)

        # step6, add two parts
        sum_array = spatial_trend_grid + res_array
        # r2_score, explained_variance_score, median_absolute_error, rmse, mean_absolute_error = accuracy_assessment(
        #     sum_array, test_data, ref_proj, gridx, gridy, pixel_width, pixel_height, feat)
        # print("r2_score :%s " % r2_score)
        # print("explained_variance_score:", explained_variance_score)
        # print("median_absolute_error:", median_absolute_error)
        # print("root mean_squared_error:", rmse)
        # print("mean_absolute_error:", mean_absolute_error)

        # array_to_tif(sum_array, dst_tif, ref_proj, ref_gt, pixel_width, pixel_height)

        # 使用中国区域的raster数据去mask插值结果，只保留中国区域范围内的值
        # cmd = 'gdal_rasterize -a provinceID -ot Float32 -of GTiff -te 1373816.07058 406842.510257 6206966.64131 5921576.51573 -tr 40000 40000 -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 -l cn_polygon_projected /mnt/win/shp/cn_polygon_projected.shp "[temporary file]"'
        ysize,xsize = sum_array.shape
        # 根据插值结果，对中国区域进行栅格化
        rasterized_bound = '/tmp/raster_boundary.tif'
        cmd = 'gdal_rasterize -a provinceID -ot Float32 -of GTiff -ts %s %s -co COMPRESS=DEFLATE -co PREDICTOR=1 -co ZLEVEL=6 -l cn_polygon_projected %s %s'%(xsize, ysize, china_boundary, rasterized_bound)
        os.system(cmd)
        # 读取栅格化后的中国边界，内部设为1
        raster_boundary = gdal.OpenShared(rasterized_bound)
        array_boundary = raster_boundary.ReadAsArray()
        array_boundary[array_boundary!=0]=1
        # 对插值结果进行mask，只保留边界内的数据

        # 插值结果array original is (xmin, ymin), should be transformed into original (xmin, ymax)
        # or the gt should be revised as [minx, pixel_width, 0.0, miny, 0.0, pixel_height
        row = array_boundary.shape[0]
        # print(row)
        for r in range(int(row / 2) - 1):
            array_boundary[[r, row - r - 1], :] = array_boundary[[row - r - 1, r], :]

        assert sum_array.shape==array_boundary.shape
        china_array = np.multiply(sum_array,array_boundary)


        # step7, output the final results into geotiff
        array_to_tif(china_array, dst_tif, ref_proj, ref_gt, pixel_width, pixel_height)

        # 处理array，存储为带颜色的png或者jpeg图片
        data = china_array
        min = np.nanmin(data)
        max = np.nanmax(data)
        print('当前处理数据：', dst_tif)
        print('最小值和最大值：', min, max)
        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # cmap = plt.cm.jet
        dst_png = dst_tif.replace('tif', 'png')
        # plt.imsave(dst_png, data, cmap=cmap)
        if min < max:
            try:
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap',
                                                            ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000',
                                                             '#8B0000'], 256)
                plt.imshow(data, cmap=cmap)
                plt.colorbar(ticks=[min, max])
                plt.savefig(dst_png)
                plt.close()
            except:
                print('绘图失败！')
        else:
            print('最小值大于等于最大值，数据不对，无法绘图')




def test():
    # 气象监测数据和相关数据
    dem_file = "/mnt/win/meterology/featuredata/china_dem.tif"
    obs_data = "/mnt/win/tmp/2010-1-1.csv"
    annrk = Datafusion(obs_data, dem_file)
    # 参考影像
    ref_raster = "/mnt/win/image/eco10_forestquality_2010.img"
    cn_studyarea_projected = "/mnt/win/shp/cn_polygon_projected.shp"
    # 融合结果的像素大小
    pixel_width = 40000
    pixel_height = 40000
    dst_tif = "/tmp/ok_china_40km.tif"
    annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_tif)


if __name__ == "__main__":
    test()
