#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
#
# Welcome to Assignment 4. This will be the most fun. Now we will prepare data for plotting.
#
# Just make sure you hit the play button on each cell from top to down. There are three functions
# you have to implement. Please also make sure than on each change on a function you hit the play
# button again on the corresponding cell to make it available to the rest of this notebook.
#
#

# Sampling is one of the most important things when it comes to visualization because often the data
# set gets so huge that you simply
#
# - can't copy all data to a local Spark driver (Watson Studio is using a "local" Spark driver) -
# can't throw all data at the plotting library
#
# Please implement a function which returns a 10% sample of a given data frame:

from pyspark.sql import SparkSession
# initialise sparkContext
spark = SparkSession.builder     .master('local')     .appName('myAppName')     .config('spark.executor.memory', '1gb')     .config("spark.cores.max", "2")     .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

import pyspark.sql.functions as F


def getSample(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #https://spark.apache.org/docs/latest/api/sql/
    return df.sample(fraction=0.1).count()

# Now we want to create a histogram and boxplot. Please ignore the sampling for now and return a python list containing all temperature values from the data set

def getListForHistogramAndBoxPlot(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #https://spark.apache.org/docs/latest/api/sql/
    my_list = df.select(F.collect_list('temperature')).first()[0]
    if not type(my_list)==list:
        raise Exception('return type not a list')
    return my_list


# Finally we want to create a run chart. Please return two lists (encapsulated in a python tuple
# object) containing temperature and timestamp (ts) ordered by timestamp. Please refer to the
# following link to learn more about tuples in python:
# https://www.tutorialspoint.com/python/python_tuples.htm


#should return a tuple containing the two lists for timestamp and temperature
#please make sure you take only 10% of the data by sampling
#please also ensure that you sample in a way that the timestamp samples and temperature samples correspond (=> call sample on an object still containing both dimensions)
def getListsForRunChart(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #https://spark.apache.org/docs/latest/api/sql/
    double_tuple_rdd = spark.sql("""
        select temperature, ts from washing where temperature is not null order by ts asc
    """).sample(False,0.1).rdd.map(lambda row : (row.ts,row.temperature))

    result_array_ts = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[0]).collect()
    result_array_temperature = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[1]).collect()
    return (result_array_ts,result_array_temperature)

    result_array_ts = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[0]).collect()
    result_array_temperature = double_tuple_rdd.map(lambda ts_temperature: ts_temperature[1]).collect()
    return (result_array_ts,result_array_temperature)

# Now it is time to grab a PARQUET file and create a dataframe out of it. Using SparkSQL you can handle it like a database.

df = spark.read.parquet('washing.parquet')
df.createOrReplaceTempView('washing')
df.show()

# Now we gonna test the functions you've completed and visualize the data.

import matplotlib.pyplot as plt

plt.hist(getListForHistogramAndBoxPlot(df,spark))
plt.show()

plt.boxplot(getListForHistogramAndBoxPlot(df,spark))
plt.show()

lists = getListsForRunChart(df,spark)

plt.plot(lists[0],lists[1])
plt.xlabel("time")
plt.ylabel("temperature")
plt.show()

# Congratulations, you are done! The following code submits your solution to the grader. Again,
# please update your token from the grader's submission page on Coursera

# from rklib import submitAll
# import json

# key = "S5PNoSHNEeisnA6YLL5C0g"
# email = ###_YOUR_CODE_GOES_HERE_###
# token = ###_YOUR_CODE_GOES_HERE_### #you can obtain it from the grader page on Coursera (have a look here if you need more information on how to obtain the token https://youtu.be/GcDo0Rwe06U?t=276)

# parts_data = {}
# parts_data["iLdHs"] = json.dumps(str(type(getListForHistogramAndBoxPlot(df,spark))))
# parts_data["xucEM"] = json.dumps(len(getListForHistogramAndBoxPlot(df,spark)))
# parts_data["IyH7U"] = json.dumps(str(type(getListsForRunChart(df,spark))))
# parts_data["MsMHO"] = json.dumps(len(getListsForRunChart(df,spark)[0]))

# submitAll(email, token, key, parts_data)
