#!/usr/bin/env python
# coding: utf-8

# ### Assignment 2 Welcome to Assignment 2. This will be fun. It is the first time you actually
# access external data from ApacheSpark.
#
# #### You can also submit partial solutions
#
# Just make sure you hit the play button on each cell from top to down. There are three functions
# you have to implement. Please also make sure than on each change on a function you hit the play
# button again on the corresponding cell to make it available to the rest of this notebook.
#

# So the function below is used to make it easy for you to create a data frame from a Cloud Object
# Store data frame using the so called "DataSource" which is some sort of a plugin which allows
# ApacheSpark to use different data sources.

# This is the first function you have to implement. You are passed a dataframe object. We've also
# registered the dataframe in the ApacheSparkSQL catalog - so you can also issue queries against the
# "washing" table using "spark.sql()". Hint: To get an idea about the contents of the catalog you
# can use: spark.catalog.listTables().  So now it's time to implement your first function. You are
# free to use the dataframe API, SQL or RDD API. In case you want to use the RDD API just obtain the
# encapsulated RDD using "df.rdd". You can test the function by running one of the three last cells
# of this notebook, but please make sure you run the cells from top to down since some are dependant
# of each other...


from pyspark.sql import SparkSession
# initialise sparkContext
spark = SparkSession.builder \
            .master('local').appName('myAppName') \
            .config('spark.executor.memory', '1gb') \
            .config("spark.cores.max", "2") \
            .getOrCreate()

sc = spark.sparkContext

# using SQLContext to read parquet file
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


#Please implement a function returning the number of rows in the dataframe
def count(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    #some more help: https://www.w3schools.com/sql/sql_count_avg_sum.asp
    #return spark.sql('select ### as cnt from washing').first().cnt
    return df.count()


# Now it's time to implement the second function. Please return an integer containing the number of fields. The most easy way to get this is using the dataframe API. Hint: You might find the dataframe API documentation useful: http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame

# In[10]:


def getNumberOfFields(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    return len(df.columns)


# Finally, please implement a function which returns a (python) list of string values of the field names in this data frame. Hint: Just copy&past doesn't work because the auto-grader will create a random data frame for testing, so please use the data frame API as well. Again, this is the link to the documentation: http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame

# In[11]:


def getFieldNames(df,spark):
    #TODO Please enter your code here, you are not required to use the template code below
    #some reference: https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
    return df.columns


# Now it is time to grab a PARQUET file and create a dataframe out of it. Using SparkSQL you can
# handle it like a database.
df = sqlContext.read.parquet('washing.parquet')
df.createOrReplaceTempView('washing')
df.show()

# The following cell can be used to test your count function

cnt = None
nof = None
fn = None

cnt = count(df,spark)
print(cnt)

# The following cell can be used to test your getNumberOfFields function

nof = getNumberOfFields(df,spark)
print(nof)

# The following cell can be used to test your getFieldNames function

fn = getFieldNames(df,spark)
print(fn)


# Congratulations, you are done. So please submit your solutions to the grader now.
#
# # Start of Assignment-Submission
#
#
# Now it’s time to submit first solution. Please make sure that the token variable contains a valid
# submission token. You can obtain it from the coursera web page of the course using the grader
# section of this assignment.
#
# Please specify you email address you are using with cousera as well.
#


# from rklib import submit, submitAll import json

# key = "SVDiVSHNEeiDqw70MIp2vA"

# if type(23) != type(cnt): raise ValueError('Please make sure that "cnt" is a number')

# if type(23) != type(nof): raise ValueError('Please make sure that "nof" is a number')

# if type([]) != type(fn): raise ValueError('Please make sure that "fn" is a list')

# email = #### your code here ### token = #### your code here ### (have a look here if you need more
# information on how to obtain the token https://youtu.be/GcDo0Rwe06U?t=276)

# parts_data = {} parts_data["2FjQw"] = json.dumps(cnt) parts_data["j8gMs"] = json.dumps(nof)
# parts_data["xaauC"] = json.dumps(fn)

# submitAll(email, token, key, parts_data)
