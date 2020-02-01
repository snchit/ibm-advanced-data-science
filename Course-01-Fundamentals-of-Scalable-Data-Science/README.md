# Fundamentals of Scalable Data Science

# Week 1
Key Concepts
Describe different methods used for IoT data analysis
Describe the challenges of data analytics
Summarize the key information about the course
Identify the tools used throughout the course that enable the data science experience

## Introduction to Apache Spark

We've been reported that some of the material in this course is too advanced. So in case you feel
the same, please have a look at the following materials first before starting this course, we've
been reported that this really helps.

Of course, you can give this course a try first and then in case you need, take the following
courses / materials. It's free...

https://cognitiveclass.ai/learn/spark

https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/f8982db1-5e55-46d6-a272-fd11b670be38/view?access_token=533a1925cd1c4c362aabe7b3336b3eae2a99e0dc923ec0775d891c31c5bbbc68

## Assessment and Setup

This is a course on Apache Spark. Please make sure you follow the videos below exactly. We see a lot
of learners ending up in a jupyter notebook environment without Apache Spark. Common errors like
“sc” or “spark” is not defined are the result.

The link below takes you to a set of video links which we constantly maintain and update:

https://github.com/IBM/coursera/wiki/Environment-Setup

## Submitting Solutions

Submitting solutions of programming assignments to the Coursera auto-grader is simple and
straightforward.

The first thing you need to do is importing the given jupyter notebook into a notebook environment
like IBM Watson Studio. Then you follow the steps inside the notebook and fill in the missing code
parts.

Finally, you obtain your submission token from the Coursera’s grader page and copy it to the
notebook. Your solution will be sent to the auto-grader directly from the notebook.

Once done, you can see your grades in the Coursera’s grader page.

Since this was a bit confusing in the past, I’ve created a short video on the topic:

https://www.youtube.com/watch?v=GcDo0Rwe06U

# Week 2
Key Concepts
Apply a solution that captures and stores IoT data from connected devices with Node-Red and Apache
ChouchDB NoSQL
Show how to process large data with ApacheSpark
Create and deploy a test data generator capable of simulating IoT sensor data coming from a
hypothetical washing machine
Compare and contrast common languages used in developing solutions for IoT data analysis ( R, Scala
and python for parallel programming on ApacheSpark )
Develop a python solution on ApacheSpark


# Week 3

Key Concepts
Illustrate transformation and basic visualization of data
Describe how to process large amount of data arriving in high velocity by using ApacheSpark and SQL
Explain the concept of multi-dimensional vector spaces and how any type of data corpus can be
understood as points in that space
Describe different statistical measures (moments) used in summarizing data

## Assignment

Recommendation: You can use the DataFrame API and SQL exclusively

More on SQL: https://www.w3schools.com/Sql/sql_intro.asp

More on the DF API:
https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame

In this assignment you will

Create an python based jupyter notebook using ApacheSpark on DataScience Experience
Access the prepared data stored in Cloud Object Storage (COS)
Calculate the minimum and maximum temperature
Calculate the first four statistical moments on temperature (mean, standard deviation, skew,
kurtosis)
Calculate the correlation between temperature and hardness
Please import the following notebook into Watson Studio and work from there (Note: it's ok to work
with Spark 2.1 for now)

https://raw.githubusercontent.com/IBM/coursera/master/coursera_ds/assignment3.1_spark2.3_python3.5_cos.ipynb

Once you are done you can export your work as python script using the name "assignment3.1.py" and
submit it to the grader which you can see at the top under "My submission"

# Week 4
Key Concepts
Analyze and draw conclusions out of the diagrams you’ve plotted
Analyze and reduce dimensions of your data set
Demonstrate how to plot Diagrams of low dimensional data sets like Box Plot, Run Chart, Scatter Plot
and Histogram

## Assignment

Recommendation: You can use the DataFrame API and SQL exclusively

More on SQL: https://www.w3schools.com/Sql/sql_intro.asp

More on the DF API:
https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame

In this assignment you will

create all necessary steps in order to prepare data for visualizing
ensure you don't crash your plotting library by sampling subsets of your data
prepare data for a histogram
prepare data for a run chart
Please import the following notebook into Watson Studio and work from there. All steps (including
grader submission) are explained and done through this notebook.

https://raw.githubusercontent.com/IBM/coursera/master/coursera_ds/assignment4.1_spark2.3_python3.6.ipynb

In case you want to have a look at the notebook first, you can use the following link:

https://github.com/IBM/coursera/blob/master/coursera_ds/assignment4.1_spark2.3_python3.6.ipynb


