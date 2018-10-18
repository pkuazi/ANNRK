#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-05-28 10:27
#

"""
存储融合图像的元数据信息
""" 
import pymysql
import pymysql.cursors

host = '192.168.40.203'
user = 'root'
passwd = ''
db = 'METEO'

connection = pymysql.connect(host=host, user=user, password=passwd, db=db, charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

# 融合数据的元数据信息存储dataid, feature, type, year, month, number
def metadata_insert(metadata):
    connection = pymysql.connect(host=host, user=user, password=passwd, db=db, charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    dataId = metadata['dataid']
    feature = metadata["feature"]
    ctype = (metadata["type"])
    date_year = metadata['year']
    date_month = metadata["month"]
    date_num = metadata["number"]

    insert_sql = """INSERT INTO `metadata_fusion` (`dataID`, `feature`, `ctype`, `date_year`, `date_month`, `date_num`) VALUES (%s, %s, %s, %s, %s, %s);"""

    update_sql = """UPDATE `metadata_fusion` SET  `feature`=%s, `ctype`=%s, `date_year`=%s, `date_month`=%s, `date_num`=%s WHERE `dataID`=%s """

    sql = "SELECT * FROM `metadata_fusion` WHERE `dataID`='%s' " % (dataId)
    c = connection.cursor()
    c.execute(sql)
    datas = c.fetchall()
    c.close()


    if len(datas) == 0:
        try:
            with connection.cursor() as cursor:
                # Create a new record
                print(insert_sql)
                cursor.execute(insert_sql, (dataId, feature, ctype, date_year, date_month, date_num))
                cursor.close()

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            connection.commit()
            print("insert ", dataId)
        finally:
            connection.close()

    else:
        try:
            with connection.cursor() as cursor:
                # Create a new record
                print(update_sql)
                cursor.execute(update_sql, (feature, ctype, date_year, date_month, date_num, dataId))
                cursor.close()
            connection.commit()
            print("update ", dataId)
        finally:
            connection.close()



if __name__ == "__main__":

#     create_table = '''CREATE TABLE `metadata_fusion` (
#     `id` int(11) NOT NULL AUTO_INCREMENT,
#     `dataID` varchar(255) COLLATE utf8_bin NOT NULL,
#     `feature` varchar(111) COLLATE utf8_bin NOT NULL,
#     `ctype` varchar(111) COLLATE utf8_bin NOT NULL,
#     `date_year` varchar(111),
#     `date_month` varchar(111),
#     `date_num` varchar(111),
#     PRIMARY KEY (`id`)
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
# AUTO_INCREMENT=1 ;'''
    metadata = {'dataid': 'AT_2010_year_2010', 'number': '', 'month': '', 'year': 2010, 'feature': 'AT', 'type': 'year'}
    metadata_insert(metadata)
