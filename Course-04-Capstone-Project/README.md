# Capstone Project

By taking the previous courses you’ve gained sufficient knowledge do complete any data science
project on your own. This project is meant that you prove this ability. In the following four weeks,
we’ll guide you through different generic steps involved in a data science project to give you a
little framework.

First, you’ll identify a data source and use case for your project. Then, you’ll investigate your
data set to know it better. You’ll also prepare it for downstream machine learning and deep
learning.

After that, you’ll test different models, to maximise your model’s performance.

Finally, you’ll create a data product and the final presentation to stakeholders.

Although this project is organized in independents components on a weekly basis we highly recommend
that you take an iterative approach. This means you’ll cycle through the whole loop multiple times
and gradually improve your deliverables.

As a final step, you’ll create a data product – something a stakeholder can use and a presentation
where you present to the stake holders and also to your peers justifying your decisions.

## Open Data

Open data is a very big movement and we want to encourage you to use open data for your project. But
you are of course also allowed to use data from another source, including your company’s data.
Finally, you are also allowed (although we don’t really encourage you to do so) to create a test
data generator / simulator in case you want to support an interesting use case but can’t get hold of
relevant data.

Please take a moment and search for an open data set of your interest. Have a brief look at the data
and decide on the use-case you want to implement.

Here are some examples of open data sets

https://opendata.cityofnewyork.us/

https://www.kaggle.com/datasets

And there a very nice and maintained list

https://github.com/awesomedata/awesome-public-datasets

Here are some examples of Use-Cases categories

Examples from the IBM Call for Code Challenge

https://developer.ibm.com/callforcode/

More generic examples:

- Predicting the Best Retail Location

- Detecting insurance fraud

- Predicting crowd movement on public events

- Predict heart rate based on activity

- Optimize irrigation based on moisture sensor values and weather forecast

- Predict production machine failure based on vibration sensor data

Once you’ve come up with an interesting use-case and data set please move on the next week.

## Use Cases

Once you've identified a Use Case and Data Set it is time to get familiar with data. In the process
model this task is called Initial Data Exploration. Please take a minute or two to (re)visit the
following lecture

https://www.coursera.org/learn/data-science-methodology Module 2 - Data Understanding

Please also revisit http://coursera.org/learn/ds/ Module 3 - Mathematical Foundations and Module 4 -
Visualizations

Given the lectures above, please create statistics and visualization on your Data Set to identify
good columns for modeling, potential data quality issues and anticipate potential feature
transformations necessary.

Create a jupyter notebook where you document your code and include visualizations as first
deliverable. Please also stick to the naming conventions explained in the the process model manual.

So, the most important reasons / steps are:

- Identify quality issues (e.g. missing values, wrong measurements, …)

- Assess feature quality – how relevant is a certain measurement (e.g. use correlation matrix)

- Get an idea on the value distribution of your data using statistical measures and
visualizations

## Design Document

As the process model is paired with architectural decision guidelines in an iterative fashion one of
the deliverables is an architectural decisions document containing the technology mapping between
architectural components and concrete technologies. In addition to the mapping, a justification for
the decision of the mapping is required so that resources entering the project at a later stage can
retrace the thinking threads of current project decision makers.

Please use the template provided below and start filling the gaps in the document. As the whole
process model is iterative, it is favored behavior if this document evolves during the creation of
the capstone deliverables.

## Guidelines

Each step (stop) in the process model comes with a set of guidelines. Please make sure you check
them out on the link below and also document any decisions you've made (using the guidelines or your
experience or both) as comments/documentation in your deliverables (e.g. jupyther notebooks).

Keeping source and documentation as tight as possible is highly recommended. So please make use of
comments in the code directly and in sections above and below the code.

And please keep in mind, good software projects have as much documentation in the code as actual
code itself.

Following the guidelines will lead to decisions and hopefully to some good comments in and above the
code so that a follow-up data scientists understand what you've done.

https://github.com/IBM/coursera/tree/master/coursera_capstone/guidelines
