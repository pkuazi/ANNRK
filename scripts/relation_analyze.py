#
# -*-coding:utf-8 -*-
#
# @Module:
#
# @Author: zhaojianghua
# @Date  : 2018-01-26 15:13
#


#!/usr/bin/env python


import os

import numpy as np

from scripts.raster_reader import get_related_data


# from sklearn import preprocessing

class Correlation:
    def __init__(self, otime, obsdata, latitude, longitude, datarelate):
        self.otime = otime
        self.obsdata = obsdata
        self.latitude = latitude
        self.longitude = longitude
        self.datarelate = datarelate

    def getTime(self):
        return self.otime

    def getData(self):
        return self.obsdata

    def getLongitude(self):
        return self.longitude

    def getLatitude(self):
        return self.latitude

    def getDatarelate(self):
        return self.datarelate

    def correlateanalyze(self):
        #first normalise, then calculate the coefficients
        obsdata = np.array(self.obsdata)
        relatedata = np.array(self.datarelate)
        norm1 = obsdata / np.linalg.norm(obsdata)
        norm2 = relatedata/np.linalg.norm(relatedata)

        return np.corrcoef(norm1,norm2)[0][1]
        # return np.corrcoef(self.obsdata, self.datarelate)[0, 1]

if __name__ == '__main__':
    # the path of data files for data fusion process
    data_path = os.getcwd() + '/data'
    # the observation meteorology data from monitor station
    monitor_file = os.path.join(data_path, '2010-1-1.csv')
    # relate data for fusion. elevation data is used here
    relate_file = os.path.join(data_path, 'cn_DEM.tif')

    relate_data_df = get_related_data(monitor_file, relate_file)

    station_num = relate_data_df.shape[0]
    temperature_list = []
    relatedata_list = []

    for i in range(station_num):
        temperature = relate_data_df.iloc[i].AT
        relatedata_dem = relate_data_df.iloc[i].dem
        temperature_list.append(temperature)
        relatedata_list.append(relatedata_dem)

    corr = Correlation(np.nan, temperature_list, np.nan, np.nan, relatedata_list)
    coef = corr.correlateanalyze()
    print("The coefficient of  temperature and elevation is %s." % coef)



