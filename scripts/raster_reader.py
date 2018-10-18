#
# -*-coding:utf-8 -*-
#
# @Module:
#
# @Author: zhaojianghua
# @Date  : 2018-01-26 15:13
#

"""

""" 

#!/usr/bin/env python
# -*-coding:utf-8 -*-
# created by 'root' on 18-1-25

import os, sys
import pandas as pd
import gdal, osr, ogr
import numpy as np


class Monitordata:
    def __init__(self, station, odate, obsdata, latitude, longitude):
        self.station = station
        self.odate = odate
        self.obsdata = obsdata
        self.latitude = latitude
        self.longitude = longitude

    def getStation(self):
        return self.station

    def getTime(self):
        return self.odate

    def getData(self):
        return self.obsdata

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude


class Relateddata:
    def __init__(self, otime, data, latitude, longitude):
        self.otime = otime
        self.data = data
        self.latitude = latitude
        self.longitude = longitude

    def getTime(self):
        return self.otime

    def getData(self):
        return self.data

    def getLatitude(self):
        return self.latitude

    def getLongitude(self):
        return self.longitude

def get_related_data(monitor_file, relate_file):
    '''

    :param monitor_file:
    :param relate_file:
    :return: related data in pandas DataFrame format
    '''
    # read the obeservation data of monitor stations
    daily_at = pd.read_csv(monitor_file, usecols=["STATION", "ODATE", 'LONGITUDE', 'LATITUDE', "AT"])
    grouped = daily_at.groupby("STATION")
    # calculate average temperature for each station
    result = grouped.agg({'AT': 'mean', 'LATITUDE': 'first', 'LONGITUDE': 'first'})
    # remove rocords with nan feature values
    daily_df = result.dropna(axis=0, how='any')
    daily_df = pd.DataFrame(daily_df)
    daily_df['STATION'] = daily_df.index

    station_num = daily_df.shape[0]
    # for each monitor point, read related data. DEM data is used here
    raster = gdal.OpenShared(relate_file)
    if raster is None:
        print("Failed to open file: " + relate_file)
        sys.exit()
    feat_proj = raster.GetProjectionRef()
    gt = raster.GetGeoTransform()

    relatedata = []
    for i in range(station_num):
        station_id = daily_df.iloc[i].STATION
        at_mean = daily_df.iloc[i].AT

        # use loc to ge series wichi satisfying condition, and then iloc to get first element
        latitude = daily_df.iloc[i]['LATITUDE']
        longitude = daily_df.iloc[i]['LONGITUDE']

        # transform geographic coordinates of monitor points into projected coordinate system
        inSpatialRef = osr.SpatialReference()
        inSpatialRef.SetFromUserInput("EPSG:4326")
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.SetFromUserInput(feat_proj)
        transform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

        # point coordinate transformation
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(longitude, latitude)
        geom.Transform(transform)
        x_c = geom.GetX()
        y_c = geom.GetY()

        # read feature value of the point from related data
        col = int((x_c - gt[0]) / gt[1])
        row = int((y_c - gt[3]) / gt[5])

        # print(col, row)
        if col <raster.RasterXSize and row < raster.RasterYSize:
            feat_value = raster.ReadAsArray(col, row, 1, 1)[0][0]
            # print(feat_value)
        else:
            continue

        relate_data = Relateddata(np.nan, feat_value, latitude, longitude)

        relatedata.append(
            [station_id, relate_data.getLatitude(), relate_data.getLongitude(), at_mean, relate_data.getData()])
    relatedata = np.array(relatedata)
    relatedata_df = pd.DataFrame(relatedata,columns=['STATION', 'LATITUDE', 'LONGITUDE', 'AT','dem' ])
    return relatedata_df

if __name__ == '__main__':
    # the path of data files for data fusion process
    data_path = os.getcwd()+'/data'
    # the observation meteorology data from monitor station
    monitor_file = os.path.join(data_path, '2010-1-1.csv')
    # relate data for fusion. elevation data is used here
    relate_file = os.path.join(data_path, 'cn_DEM.tif')

    # read relate data according to the coordinates of the monitor stations
    relate_data_df = get_related_data(monitor_file, relate_file)
    print(relate_data_df)


    # output the relate data into a csv file
    relate_data_path = os.path.join(data_path,'relate_data.csv')
    relate_data_df.to_csv(relate_data_path)
    print('the data has been save into %s.'% relate_data_path)




