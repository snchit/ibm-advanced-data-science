#!/usr/bin/env python
# coding: utf-8

# # Assignment 3
#
# Welcome to Assignment 3. This will be even more fun. Now we will calculate statistical measures.
#
# ## You only have to pass 4 out of 7 functions
#
# Just make sure you hit the play button on each cell from top to down. There are seven functions
# you have to implement. Please also make sure than on each change on a function you hit the play
# button again on the corresponding cell to make it available to the rest of this notebook.

# All functions can be implemented using DataFrames, ApacheSparkSQL or RDDs. We are only interested
# in the result. You are given the reference to the data frame in the "df" parameter and in case you
# want to use SQL just use the "spark" parameter which is a reference to the global SparkSession
# object. Finally if you want to use RDDs just use "df.rdd" for obtaining a reference to the
# underlying RDD object. But we discurage using RDD at this point in time.
#
# Let's start with the first function. Please calculate the minimal temperature for the test data
# set you have created. We've provided a little skeleton for you in case you want to use SQL.
# Everything can be implemented using SQL only if you like.


from pyspark.sql import SparkSession
# initialise sparkContext
spark = SparkSession.builder     \
            .master('local')     \
            .appName('myAppName')     \
            .config('spark.executor.memory', '1gb')     \
            .config("spark.cores.max", "2")     \
            .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


import pyspark.sql.functions as F


def minTemperature(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    return df.agg({"temperature":"min"}).collect()[0][0]

# Please now do the same for the mean of the temperature

def meanTemperature(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    return df.agg({"temperature":"mean"}).collect()[0][0]

# Please now do the same for the maximum of the temperature

def maxTemperature(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    return df.agg({"temperature":"max"}).collect()[0][0]

# Please now do the same for the standard deviation of the temperature

def sdTemperature(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #https://spark.apache.org/docs/2.3.0/api/sql/
    return df.select(F.stddev('temperature')).first()[0]

# Please now do the same for the skew of the temperature. Since the SQL statement for this is a bit
# more complicated we've provided a skeleton for you. You have to insert custom code at four
# positions in order to make the function work. Alternatively you can also remove everything and
# implement if on your own. Note that we are making use of two previously defined functions, so
# please make sure they are correct. Also note that we are making use of python's string formatting
# capabilitis where the results of the two function calls to "meanTemperature" and "sdTemperature"
# are inserted at the "%s" symbols in the SQL string.


def skewTemperature(df,spark):
    return df.select(F.skewness('temperature')).first()[0]

# Kurtosis is the 4th statistical moment, so if you are smart you can make use of the code for skew
# which is the 3rd statistical moment. Actually only two things are different.

def kurtosisTemperature(df,spark):
    return df.select(F.kurtosis('temperature')).first()[0]

# Just a hint. This can be solved easily using SQL as well, but as shown in the lecture also using RDDs.

def correlationTemperatureHardness(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #https://spark.apache.org/docs/2.3.0/api/sql/
    return df.select(F.corr('temperature', 'hardness')).first()[0]

# Now it is time to grab a PARQUET file and create a dataframe out of it. Using SparkSQL you can handle it like a database.

df = spark.read.parquet('washing.parquet')
df.createOrReplaceTempView('washing')
df.show()

# Now let's test the functions you've implemented


min_temperature = minTemperature(df,spark)
print(min_temperature)


mean_temperature = meanTemperature(df,spark)
print(mean_temperature)

max_temperature = maxTemperature(df,spark)
print(max_temperature)

sd_temperature = sdTemperature(df,spark)
print(sd_temperature)

skew_temperature = skewTemperature(df,spark)
print(skew_temperature)

kurtosis_temperature = kurtosisTemperature(df,spark)
print(kurtosis_temperature)

correlation_temperature = correlationTemperatureHardness(df,spark)
print(correlation_temperature)


# Congratulations, you are done, please submit this notebook to the grader.  We have to install a
# little library in order to submit to coursera first.
#
# Then, please provide your email address and obtain a submission token on the grader’s submission
# page in coursera, then execute the subsequent cells
#
# ### Note: We've changed the grader in this assignment and will do so for the others soon since it
# gives less errors This means you can directly submit your solutions from this notebook

# from rklib import submitAll
# import json

# key = "Suy4biHNEeimFQ479R3GjA"
# email = ###_YOUR_CODE_GOES_HERE_###
# token = ###_YOUR_CODE_GOES_HERE_### #you can obtain it from the grader page on Coursera


# parts_data = {}
# parts_data["FWMEL"] = json.dumps(min_temperature)
# parts_data["3n3TK"] = json.dumps(mean_temperature)
# parts_data["KD3By"] = json.dumps(max_temperature)
# parts_data["06Zie"] = json.dumps(sd_temperature)
# parts_data["Qc8bI"] = json.dumps(skew_temperature)
# parts_data["LoqQi"] = json.dumps(kurtosis_temperature)
# parts_data["ehNGV"] = json.dumps(correlation_temperature)

# submitAll(email, token, key, parts_data)
