#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-05-28 11:02
#

"""
融合数据产品生产和存储
"""
import datafusion
from datafusion import Datafusion
from metadata import metadata_insert
import pandas as pd
import os, sys
from pandas import DataFrame
import numpy as np

# data_file = "/home/zjh/tmp/surface2010-2015.csv"

# SAPRK读取数据
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkConf
import json
from datetime import date
from pyspark.sql.types import DoubleType
conf = (SparkConf()
        .set("spark.dynamicAllocation.enabled", True)
        .set("spark.executor.memory", '20g')
        .set("spark.cores.max",'16'))
spark = SparkSession.builder.master("spark://10.0.85.60:7077").appName("co_query").config(conf=conf).getOrCreate()
# df = spark.read.format('csv').option('header','true').option('mode','DROPMALFORMED').load("hdfs://192.168.40.203:9000/tmp/earth_surface.csv")

# 类型：日，侯，旬，季，年
type = ['day', 'hou', 'xun', 'season', 'year']


from PIL import Image
from scipy import misc
import gdal

def test():
    tif = '/root/PycharmProjects/ANNRK/data/ok_china_40km.tif'
    raster = gdal.OpenShared(tif)
    arr = raster.ReadAsArray()
    max = np.max(arr)
    min = np.min(arr)
    misc.toimage(arr, high=max, low=min ).save('/tmp/test.jpg')


def avg_compute_and_fusion():
    BASE_DIR = "/root/fusion_products"
    # 参考和辅助数据
    dem_file = os.path.join(BASE_DIR, 'ANNRK/data/cn_DEM.tif')
    ref_raster = os.path.join(BASE_DIR, 'ANNRK/data/ok_china_40km.tif')
    cn_studyarea_projected = os.path.join(BASE_DIR, 'ANNRK/data/cn_polygon_projected.shp')
    # 融合产品存储路径
    RESULT_DIR = os.path.join(BASE_DIR, 'fusion_images')
    # 设定融合参数
    pixel_width = 40000
    pixel_height = 40000

    # 读取数据
    df = spark.read.format('csv').option('header', 'true').option('mode', 'DROPMALFORMED').load(
        "hdfs://10.0.85.60:9000/tmp/surf.csv")
    # 气象要素：温度、降水
    feature = ['AT','RAIN24']
    for feat in feature:
        #读取待融合的 日，侯，旬，月，季，年 站点 温度 数据
        feat_df = df.select(df.STATION, df.YEAR, df.MONTH, df.DAY, feat, 'LONGITUDE', 'LATITUDE','ELEVATION')
        # 先读取年数据
        year_list = range(2010,2017)
        season_list = range(1,5)
        month_list = range(1,13)
        xun_list = range(1,4)
        hou_list = range(1,7)

        for year in year_list:
            DATA_DIR = os.path.join(RESULT_DIR, feat + '/' + str(year))
            if not os.path.exists(DATA_DIR):
                os.system('mkdir %s' % os.path.join(RESULT_DIR, feat))
                os.system('mkdir %s' % DATA_DIR)
            print('新融合数据存储路径为：',DATA_DIR)
            feat_AT_year = feat_df.filter(feat_df['YEAR'] == year)

            print('begin %s...'%year)
            # temperature/2010/year/2010.tif
            dst_year = os.path.join(DATA_DIR, '%s_%s_year_%s.tif'%(feat, year, year))
            if os.path.exists(dst_year):
                print('%s年的%s均值融合插值数据已存在。。' % (year, feat))
            else:
                # 计算各个站点的年均值
                feat_year_avg = feat_AT_year.groupBy(feat_AT_year.STATION).agg(F.avg(feat).alias(feat),
                                                                               F.first('LONGITUDE').alias('LONGITUDE'),
                                                                               F.first('LATITUDE').alias('LATITUDE'),
                                                                               F.first('ELEVATION').alias('ELEVATION'))
                pddf_AT_year = feat_year_avg.toPandas()
                # 去掉空值，便于归一化
                pddf_AT_year = pddf_AT_year.dropna(axis=0, how='any')
                pddf_AT_year = pddf_AT_year.astype('float64')
                annrk = Datafusion(pddf_AT_year, dem_file)
                annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_year,feat)
                print('%s年的%s均值融合插值已结束。。' % (year, feat))
            metadata = {'dataid': '%s_%s_year_%s' % (feat, year, year), "feature": feat, "type": 'year',
                        'year': year, "month": '', "number": ''}
            metadata_insert(metadata)
            feat_AT_year = feat_AT_year.withColumn("SEASON", F.when(
                    (feat_AT_year.MONTH == 3) | (feat_AT_year.MONTH == 4) | (feat_AT_year.MONTH == 5), 1)
                                                       .when(
                    (feat_AT_year.MONTH == 6) | (feat_AT_year.MONTH == 7) | (feat_AT_year.MONTH == 8), 2)
                                                       .when(
                    (feat_AT_year.MONTH == 9) | (feat_AT_year.MONTH == 10) | (feat_AT_year.MONTH == 11), 3)
                                                       .when(
                    (feat_AT_year.MONTH == 12) | (feat_AT_year.MONTH == 1) | (feat_AT_year.MONTH == 2), 4))
            for season in season_list:
                feat_AT_season = feat_AT_year.filter(feat_AT_year['SEASON'] == season)
                dst_season = os.path.join(DATA_DIR, '%s_%s_season_%s%02d.tif' % (feat, year, year, season))
                if os.path.exists(dst_season):
                    print('%s年的%s第%s季度的均值融合插值已存在。。' % (year, feat, season))
                else:
                    # 计算各个站点的季度均值
                    feat_season_avg = feat_AT_season.groupBy(feat_AT_season.STATION).agg(F.avg(feat).alias(feat),
                                                                                         F.first('LONGITUDE').alias('LONGITUDE'),
                                                                                         F.first('LATITUDE').alias('LATITUDE'),
                                                                                         F.first('ELEVATION').alias('ELEVATION'))
                    pddf_AT_seaon = feat_season_avg.toPandas()
                    pddf_AT_seaon = pddf_AT_seaon.astype('float64')
                    pddf_AT_seaon = pddf_AT_seaon.dropna(axis=0, how='any')
                    annrk = Datafusion(pddf_AT_seaon, dem_file)
                    # temperature/2010/season/201001.tif

                    annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_season,feat)
                    print('%s年的%s第%s季度的均值融合插值已结束。。' % (year, feat, season))
                metadata = {'dataid': '%s_%s_season_%s%02d' % (feat, year,year,season), "feature": feat, "type": 'season',
                            'year': year, "month": '', "number": '' }
                metadata_insert(metadata)

            for month in month_list:
                feat_AT_month = feat_AT_year.filter(feat_AT_year['MONTH'] == month)
                # temperature/2010/month/201001.tif
                dst_month = os.path.join(DATA_DIR, '%s_%s_month_%s%02d.tif' % (feat, year,year, month))
                if os.path.exists(dst_month):
                    print('%s年的%s第%s月均值融合插值已存在。。' % (year, feat, month))
                else:
                    # 计算各个站点的季度均值
                    feat_month_avg = feat_AT_month.groupBy(feat_AT_season.STATION).agg(F.avg(feat).alias(feat),
                                                                                       F.first('LONGITUDE').alias('LONGITUDE'),
                                                                                       F.first('LATITUDE').alias('LATITUDE'),
                                                                                       F.first('ELEVATION').alias('ELEVATION'))
                    pddf_AT_month = feat_month_avg.toPandas()
                    pddf_AT_month = pddf_AT_month.astype('float64')
                    pddf_AT_month = pddf_AT_month.dropna(axis=0, how='any')
                    annrk = Datafusion(pddf_AT_month, dem_file)
                    annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_month,feat)
                    print('%s年的%s第%s月均值融合插值已结束。。' % (year, feat, month))
                metadata = {'dataid': '%s_%s_month_%s%02d' % (feat, year,year, month), "feature": feat,
                            "type": 'month','year': year, "month": month, "number": '' }
                metadata_insert(metadata)

                feat_AT_month = feat_AT_month.withColumn("HOU", F.floor((feat_AT_month.DAY - 1) / 5 + 1))
                feat_AT_month = feat_AT_month.withColumn("HOU", F.floor((feat_AT_month.DAY - 1) / 5 + 1))
                feat_AT_month = feat_AT_month.withColumn("XUN", F.floor((feat_AT_month.DAY - 1) / 10 + 1))
                # otherwise is important, if not, None would be the value
                feat_AT_month = feat_AT_month.withColumn("XUN", F.when((feat_AT_month.DAY == 31), 3).otherwise(feat_AT_month.XUN))
                for xun in xun_list:
                    feat_AT_xun = feat_AT_month.filter(feat_AT_month["XUN"] == xun)
                    # temperature/2010/xun/20100103.tif
                    dst_xun = os.path.join(DATA_DIR, '%s_%s_xun_%s%02d%02d.tif' % (feat, year, year, month,xun))
                    if os.path.exists(dst_xun):
                        print('%s年的%s第%s月第%s旬均值融合插值已存在。。' % (year, feat, month,xun))
                    else:
                        # 计算当前月份各个站点的旬均值
                        feat_xun_avg = feat_AT_xun.groupBy(feat_AT_xun.STATION).agg(F.avg(feat).alias(feat), F.first('LONGITUDE').alias('LONGITUDE'),
                                                                                    F.first('LATITUDE').alias('LATITUDE'),
                                                                                    F.first('ELEVATION').alias('ELEVATION'))
                        pddf_AT_xun = feat_xun_avg.toPandas()
                        pddf_AT_xun = pddf_AT_xun.astype('float64')
                        pddf_AT_xun = pddf_AT_xun.dropna(axis=0, how='any')
                        annrk = Datafusion(pddf_AT_xun, dem_file)
                        annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_xun,feat)
                        print('%s年的%s第%s月第%s旬均值融合插值已结束。。' % (year, feat, month,xun))
                    metadata = {'dataid': '%s_%s_xun_%s%02d%02d' % (feat, year, year, month,xun), "feature": feat,
                                "type": 'xun', 'year': year, "month": month, "number": '' }
                    metadata_insert(metadata)

                for hou in hou_list:
                    feat_AT_hou = feat_AT_month.filter(feat_AT_month["HOU"] == hou)
                    # temperature/2010/hou/20100103.tif
                    dst_hou = os.path.join(DATA_DIR, '%s_%s_hou_%s%02d%02d.tif' % (feat, year, year, month, hou))
                    if os.path.exists(dst_hou):
                        print('%s年的%s第%s月第%s侯均值融合插值已存在。。' % (year, feat, month, hou))
                    else:
                        # 计算当前月份各个站点的旬均值
                        feat_hou_avg = feat_AT_hou.groupBy(feat_AT_hou.STATION).agg(F.avg(feat).alias(feat), F.first('LONGITUDE').alias('LONGITUDE'),
                                                                                    F.first('LATITUDE').alias('LATITUDE'),
                                                                                    F.first('ELEVATION').alias('ELEVATION'))
                        pddf_AT_hou = feat_hou_avg.toPandas()
                        pddf_AT_hou = pddf_AT_hou.astype('float64')
                        pddf_AT_hou = pddf_AT_hou.dropna(axis=0, how='any')
                        annrk = Datafusion(pddf_AT_hou, dem_file)
                        annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_hou,feat)
                        print('%s年的%s第%s月第%s侯均值融合插值已结束。。' % (year, feat, month, hou))
                    metadata = {'dataid': '%s_%s_hou_%s%02d%02d' % (feat, year, year, month, hou), "feature": feat,
                                "type": 'hou', 'year': year, "month": month, "number": '' }
                    metadata_insert(metadata)

                day_list = list(feat_AT_month.select('DAY').distinct().toPandas()['DAY'])
                for day in day_list:
                    day = int(day)
                    feat_AT_day = feat_AT_month.filter(feat_AT_month["DAY"] == day)
                    # temperature/2010/day/20100103.tif
                    dst_day = os.path.join(DATA_DIR, '%s_%s_day_%s%02d%02d.tif' % (feat, year, year, month, day))
                    if os.path.exists(dst_day):
                        print('%s年的%s第%s月第%s天均值融合插值已存在。。' % (year, feat, month, day))
                    else:
                        # 计算当前月份各个站点的旬均值
                        feat_day_avg = feat_AT_day.groupBy(feat_AT_day.STATION).agg(F.avg(feat).alias(feat), F.first('LONGITUDE').alias('LONGITUDE'),
                                                                                    F.first('LATITUDE').alias('LATITUDE'),
                                                                                    F.first('ELEVATION').alias('ELEVATION'))
                        pddf_AT_day = feat_day_avg.toPandas()
                        pddf_AT_day = pddf_AT_day.astype('float64')
                        pddf_AT_day = pddf_AT_day.dropna(axis=0, how='any')

                        annrk = Datafusion(pddf_AT_day, dem_file)
                        annrk.annrk_datafusion(ref_raster, cn_studyarea_projected, pixel_width, pixel_height, dst_day, feat)
                        print('%s年的%s第%s月第%s天均值融合插值已结束。。' % (year, feat, month, day))
                    metadata = {'dataid': '%s_%s_day_%s%02d%02d' % (feat, year, year, month, day), "feature": feat,
                                "type": 'day', 'year': year, "month": month, "number": day }
                    metadata_insert(metadata)

if __name__=='__main__':
    avg_compute_and_fusion()




