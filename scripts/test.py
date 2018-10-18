#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-04-09 14:34
#

"""
栅格数据读取，关联关系分析，数据融合单元测试
"""

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from datafusion import Datafusion
from pandas.util.testing import assert_frame_equal # <-- for testing dataframes
from raster_reader import get_related_data

from scripts.relation_analyze import Correlation


class TestDFMethods(unittest.TestCase):
    def setUp(self):
        '''
        define all the input for unit tests
        :return: void
        '''
        # the path of data files for data fusion process
        self.data_path = os.getcwd() + '/data'
        # the observation meteorology data from monitor station
        self.obs_data = os.path.join(self.data_path, '2010-1-1.csv')
        # relate data for fusion. elevation data is used here
        self.dem_file = os.path.join(self.data_path, 'cn_DEM.tif')

        # reference image offering projection and study area boundary
        self.ref_raster = os.path.join(self.data_path, 'reference.img')
        self.cn_studyarea_projected = os.path.join(self.data_path,'cn_polygon_projected.shp')

        # resolution of the output image
        self.pixel_width = 40000
        self.pixel_height = 40000

        # output image path
        self.dst_tif = os.path.join(self.data_path,'ok_china_40km.tif')


    def test_raster_reader(self):
        '''
        test the data preparation unit
        :return: void
        '''
        # read relate data according to the coordinates of the monitor stations
        relate_data_df = get_related_data(self.obs_data, self.dem_file)

        # read test data saved
        test_data = os.path.join(self.data_path, 'relate_data_20100101.csv')
        test_df = pd.read_csv(test_data, usecols=['STATION', 'LATITUDE', 'LONGITUDE', 'AT','dem' ])

        # compare two dataframe
        assert_frame_equal(relate_data_df, test_df)

    def test_relation_analyze(self):
        '''
        test the relation analysis unit
        :return: void
        '''
        relate_data_df = get_related_data(self.obs_data, self.dem_file)

        station_num = relate_data_df.shape[0]
        temperature_list = []
        relatedata_list = []

        # read elevation for each station that has temperature monitored value
        for i in range(station_num):
            temperature = relate_data_df.iloc[i].AT
            relatedata_dem = relate_data_df.iloc[i].dem
            temperature_list.append(temperature)
            relatedata_list.append(relatedata_dem)

        # compute the correlation coefficient
        corr = Correlation(np.nan, temperature_list, np.nan, np.nan, relatedata_list)
        coef = corr.correlateanalyze()
        print("The coefficient of  temperature and elevation is %s." % coef)
        self.assertIsInstance(coef,float)

    def test_data_fusion(self):
        '''
        test the data fusion unit
        :return:void
        '''
        annrk = Datafusion(self.obs_data, self.dem_file)

        if os.path.exists(self.dst_tif):
            os.remove(self.dst_tif)
        dst_path = Path(self.dst_tif)

        # before data fusion, the fused file not exists
        self.assertFalse(dst_path.is_file())

        # data fusion function
        annrk.annrk_datafusion(self.ref_raster, self.cn_studyarea_projected, self.pixel_width, self.pixel_height, self.dst_tif)
        # after data fusion, the fused file exists
        self.assertTrue(dst_path.is_file())

if __name__ == '__main__':
    unittest.main()
