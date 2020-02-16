# AI Workflow: Deployment

# DATA AT SCALE

Our Story
The DevOps and engineering side of AAVAIL has a limited number of data engineers. The agreed-upon
approach to model deployment is that the members of the data science team provide their models as
portable containers complete with both prediction and training endpoints. To do this in scalable way
the team makes use of Spark whenever it is appropriate.

When iterating on the AI enterprise workflow it is important to recall the guiding phrase: make it
work, make it better, then make it faster. By first making it work, we are able to reduce
opportunity cost if a model shows little promise. Making it better refers to code polish, code
optimizations, docstrings, and readability. When you deploy a model, you take on a significant
responsibility and are obligated to make the model run as fast as possible. This course focuses on
speed improvements that can be made before a model is deployed, but it should be kept in mind that
ongoing improvements are part of the commitment that is made when deciding to deploy a model.


THE DESIGN THINKING PROCESS
The Prototype phase of the design thinking process is where ideas are tested in real-life to see if
they work in solving business challenges. The Prototype phase is about making a model work. At this
stage, you’ll be developing and tweaking many different models, all in an attempt to select the best
model for the job.

You’ll also learn about the non-linear aspects of design thinking. As you create and test prototype
models you will probably discover that you don’t have all of the data or user insights that you
need. Recall that design thinking is user centric. As your users engage with your models they will
come up with more suggestions and ideas for making your models more useful. During the earliest
stages of prototyping you’ll be building new data pipelines and new models, testing you abilities
every day.

Connecting the Dots
Some business opportunities can be solved by running a tuned model once on all available data. For
example, if AAVAIL was interested in whether or not to continue investing in a particular market, a
time-series forecasting model could be created for both the new market and a baseline. This model,
along with some visualizations and additional analysis, would likely be compiled into a report and
there would be no need for it to exist in a persistent state with callable endpoints. Ideally, the
forecasting model would be written in way that would allow it to be easily adopted to other market
analyses.

Depending on the needs of your business, it is possible that a sizable percentage of models exist as
standalone projects that would provide little added value if they were deployed. It is also possible
that nearly all models need to be deployed in production to be of benefit to the business. Either
way it is likely that model deployment will become a necessity at some point and optimization is a
key consideration for any model or service that is to be deployed.

## Optimization in Python

Today data scientists have more tooling than ever before to create model-driven or algorithmic
solutions. Because of tooling availability and an increasing popularity in the field of data
science, many practicing data scientists only have a few years of experience. This trend has
resulted in a lack of awareness when it comes to code optimization. There are many ways to increase
the speed of code and for some business applications speed is directly related to revenue. Before we
jump into the strategies and tools to help optimize code, it is important to know when to take the
time to make code optimizations.

An example of a situation where it is difficult to make speed improvements through code is when we
have a model that takes several days on multiple GPUs to train. A speech-to-text engine is a good
example, but almost any large neural network with a reasonable amount of data falls into this
category. You can profile TensorFlow models using TensorBoard but if you have already optimized the
model for performance, it becomes tricky to make improvements from a code perspective. You can
always use more GPUs or other computational resources, but beyond saving model checkpoints in the
event that there is a failure, little can be done to improve training time.

With scikit-learn model training it is also difficult to optimize apart from using the
\verb|njobs|njobs argument and trying several optimization algorithms when appropriate. In general,
unless you write your own inference portion of the code, as is sometimes the case with
expectation-maximization, improving on the efficiency of available inference algorithms is either
difficult or unrealistic. For model prediction there are several best practices, such as keeping the
trained model in memory, that we will discuss later that will help ensure optimized performance.

There are plenty of examples where a well-written script addresses a business opportunity even
without the use of machine learning. Perhaps the AAVAIL sales team wanted to optimize quarterly
travel schedules. This would be an example of the traveling salesman problem where you could write a
brute-force algorithm or use some variant on a minimal spanning tree to solve it. Either way, these
are useful tools that do not use machine learning algorithms.

Two important areas of data science where machine learning algorithms may not be the best solution
are:

optimization
graph theory
The first rule to ensuring that you are optimizing your code in a smart way is to look around for
implementations before creating one from from scratch. The scipy.optimize submodule has a number of
optimizers and algorithms (some of them general purpose) already implemented. If your problem is in
graph space like customer journeys or social networks then check out the algorithms implemented by
NetworkX before you set off building your own.

Finally, we come to the scripts or blocks of code that need speed improvements, but you have come to
the conclusion that there is no optimized code readily available. The task of optimizing the code
then falls to you. The first step is to identify which parts of your code are bottlenecks. This is
done using profiling or more specifically Python profilers. Once the specific pieces of code that
need to be optimized are identified, then there are a number of common tools that may be used to
improve the speed of programs. Several of these tools make use of the fact that modern computers
have multiple available processor cores on a machine. To see how many processor cores are available
on your machine or compute resource try the following code.


```python
from multiprocessing import Pool, cpu_count
total_cores = cpu_count()
print('total cores: ', total_cores)
```

```bash
total cores: 8
```


A list of commonly used techniques and tools to optimize code:
Use appropriate data containers - For example, a Python set has a shorter look-up time than a Python
list. Similarly, use dictionaries and NumPy arrays whenever possible.

Multiprocessing - This is a package in the standard Python library and it supports spawning
processes (for each core) using an API similar to the threading module. The multiprocessing package
offers both local and remote concurrency.

Threading - Another package in the standard library that allows separate flows of execution at a
lower level than multiprocessing.

Subprocessing - A module that allows you to spawn new processes, connect to their input/output/error
pipes, and obtain their return codes. You may run and control non-Python processes like Bash or R
with the subprocessing module.

mpi4py - MPI for Python provides bindings of the Message Passing Interface (MPI) standard for the
Python programming language, allowing any Python program to exploit multiple processors.

ipyparallel - Parallel computing tools for use with Jupyter notebooks and IPython. Can be used with
mpi4py.

Cython - An optimizing static compiler for both the Python programming language and the extended
Cython programming language. It is generally used to write C extensions for slow portions of code.

CUDA (Compute Unified Device Architecture) - Parallel computing platform and API created by Nvidia
for use with CUDA-enabled GPUs. CUDA in the Python environment is often run using the package
PyCUDA.

Additional materials
scipy-lectures tutorial for optimizing code
mpi4py tutorial
ipyparallel demos
Cython tutorial

## High Performance Python

We mentioned in a previous section that inference can be difficult to optimize and that one way
around this is to add more GPUs. The general idea of using an aggregation of compute resources to
dramatically increase available compute resources is known as high-performance computing (HPC) or
supercomputing. Within this field there is the important concept of parallel computing, which is
exactly what we enable by adding multiple GPUs to compuation tasks.

Supercomputers and parallel computing can help with model training, prediction and other related
tasks, but it is worth noting that there are two laws that constrain the maximum speed-up of
computing: Amdahl’s law and Gustafson’s law. Listed below is some of the important terminology in
this space.

Symmetric multiprocessing - Two or more identical processors connected to a single unit of memory.

Distributed computing - Processing elements are connected by a network.

Cluster computing - Group of loosely (or tightly) coupled computers that work together in a way that
they can be viewed as a single system.

Massive parallel processing - Many networked processors usually > 100 used to perform computations
in parallel.

Grid computing - distributed computing making use of a middle layer to create a virtual super
computer.

An important part of this course is dealing with data at scale, which is closely related to both
code optimization and parallel computing. In this course will be using Apache Spark, a
cluster-computing framework, to enable parallel computing.

If we talk about scale in the context of a program or model, we may be referring to any of the
following questions. Let the word service in this context be both the deployed model and the
infrastructure.

Does my service train in a reasonable amount of time given a lot more data?
Does my service predict in a reasonable amount of time given a lot more data?
Is my service ready to support additional request load?
It is important to think about what kind of scale is required by your model and business application
in terms of which bottleneck is most likely going to be the problem associated with scale. These
bottlenecks will depend heavily on available infrastructure, code optimizations, choice of model and
type of business opportunity. The three questions above can serve as a guide to help put scale into
perspective for a given situation.

Additional resources
High performance computing at IBM

## Spark-Submit

A Spark cluster can be managed in several ways. Some of the different managers include Apache Mesos,
Hadoop YARN, and Spark on Kubernetes. Applications can be submitted to any of these types of Spark
cluster environments using the \verb|spark-submit|spark-submit command and a script. Spark clusters
are often used by some combination of users and applications so the management layer is critical to
ensure that jobs can be monitored and properly scheduled.

One example of how you would run \verb|spark-submit|spark-submit would be to save the following file
as example-spark-submit.sh.


```bash
#!/bin/bash
${SPARK_HOME}/bin/spark-submit \
--master local[4] \
--executor-memory 1G \
--driver-memory 1G \
$@
```

The `$@` indicates that the argument that follows will be appended. This means that you could
submit a file, for example example-script.py using:


```bash
~$ ./example-spark-submit.sh example-script.py
```


Important:
The permissions will need to be modified to ensure you can run the script.


```bash
~$ chmod 711 example-spark-submit.sh
```

See the Spark application submission guide for more details on \verb|spark-submit|spark-submit
submissions. If you are new to Spark or you would like a refresher check out IBM’s introduction to
Spark tutorial.

## QUIZ

1. Question 1
True/False. If we continue to add GPUs or other computational resources the time it takes to train a
model will always continue to decrease.

    True

Incorrect
Amdahl’s law says that the speed-up from parallelization is bounded by the ratio of parallelizable
to irreducibly serial code in the algorithm. However, for big data analysis, Gustafson’s law is more
relevant. This says that we are nearly always interested in increasing the size of the
parallelizable bits, and the ratio of parallelizable to irreducibly serial code is not a static
quantity but depends on data size.

In short, we are nearly always interested in adding more resources, if possible, but there is a
theoretical point at which speed-up is no longer improved with more resources.

# DOCKER

Our Story
AAVAIL has engineers and data scientists with diverse backgrounds, which encourages the blending of
ideas, but it also means that there are many different operating systems in use, along with numerous
opinions on what makes an ideal development environment. Fully aware of this, the technical
leadership has mandated that software engineers, data scientists, data engineers, and generally
everyone else under their purview ship services in Docker containers. By doing this, all services
exist under a common language. For example, a machine learning model can be passed around and used
in the same general way as a microservice developed for use by mobile application developers. The
implementation of the Docker rule has resulted in an increased awareness across teams at AAVAIL. It
has also decreased the workload of DevOps personnel and data engineers.


THE DESIGN THINKING PROCESS
Enterprise-scale data science will necessarily involve the eventual deployment of machine learning
models for others to use. Instead of simply building models and then running them on behalf of
others, you’ll be responsible for making your model available.

This will likely happen during the Prototype or Test phases of the design thinking process. A model
that you have developed will need to be made available to others for testing purposes, and you’ll be
responsible for assessing not only the performance of the model, but also for responding to user
feedback.

Remember that design thinking is user-centric. Also remember the design thinking cycle of Observe,
Reflect, and Make. This cycle takes place rapidly, and is based on user feedback. What this means
for the data scientist is that you’ll need to make fast changes to your models in response to user
feedback during the Prototype and Test cycles.

An easy way to quickly deploy models with a publically available endpoint is via Docker containers.
Knowing how to quickly deploy Docker containers and make changes to the contents of those containers
are essential skills for the data scientist working in a large enterprise.

## Docker and Containers

Docker is an open-source application that performs operating-system level containerization. These
containers can hold multiple independently running applications. It is an easier method to create a
portable and reusable environment than running explicit servers or virtual environments. In data
science today, Docker is the industry standard for containerization of machine learning models and
AI services.

The community package of the Docker Engine is now called \verb|docker-ce|docker-ce. Fundamentally, a
docker container can be thought of as a running script with all the necessary components bundled
within. The containers themselves sit on top of an operating system as shown in the following image.

![]("./docker.png")

The Docker container is a running process that is kept isolated from the host and from other
containers. One of the important consequence of this isolation is that each container interacts with
its own private filesystem. A Docker image includes everything needed to run an application: code,
runtime libraries, and a private filesystem.

To install the Docker Engine use the appropriate guide from the following:

Docker Ubuntu install guide
Docker macOS install guide
Docker Windows install guide

When you have finished the install you should be able to run the “hello world” example.

```bash
~$ docker run hello-world
```

or if root privileges are required

```bash
sudo docker run hello-world
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

Congratulations! You have just run a container. It could have been anything from a fully functional
spark environment to a simple model that a colleague has recently deployed. There are a number of
arguments for the docker run command that we will get into, but this is the basis for running
containers.


To use TensorFlow with a GPU you need to ensure that the NVIDIA driver, CUDA and additional required
libraries are set up and versioned appropriately. Then you can install
\verb|tensroflow-gpu|tensroflow-gpu. There is some amount of overhead involved in getting this
ecosystem running smoothly and additionally there are maintenance requirements as the stable version
changes over time. TensorFlow can also be installed via Docker with the use of a GPU.

The process is similar for PyTorch, Caffe and in general any deep-learning framework that makes use
of GPUs. The NVIDIA container toolkit or simply nvidia-docker is an incredibly convenient way to
build and run GPU-accelerated Docker containers. Once this is done, you can pull down the latest GPU
version of tensorflow with Jupyter support with:

```bash
~$ docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

**NOTE: You do NOT need to install nvidia-docker for any of the the exercises or case studies in
these materials. The mention here is so that you are aware it is a best practice when deploying
deep-learning systems that make use of GPUs**

NVIDIA Docker and GPU computing are not required for this course or any in this specialization, but
knowledge of Dockerized versions of TensorFlow and similar tools can save significant amounts of
time. You will need to ensure that \verb|docker-ce|docker-ce is installed.

For more information on TensorFlow Docker Images and other images supported by
\verb|nvidia-docker|nvidia-docker see the following links.

nvidia-docker
nvidia-docker documentation
TensorFlow Docker Images
PyTorch NVIDIA docker

First, let’s go through the motions of running a container using the Docker tutorial example. If you
have already done this, feel free to skip ahead.

First, download the examples:

```bash
~$ git clone https://github.com/docker/doodle.git
~$ cd doodle/cheers2019/
```

Then, you can simply run the container:

```bash
~$ docker run -it --rm docker/doodle:cheers
```

Alternatively, if you had made changes to the Dockerfile you could rebuild it and then run

```bash
~$ sudo docker build .
~$ docker run -it --rm docker/doodle:cheers
```

If you look inside the repository you will see a \verb|`Dockerfile`|‘Dockerfile‘ that is used to
make it happen. Have a look at the contents of that file to get an idea for what is happening when
that file is used.

```bash
~$ ls ~/sandbox/doodle/cheers2019
~$ Dockerfile              Dockerfile.cross        README.md               cheers.go
```

Dockerfiles describe how to assemble a private filesystem for a container. Remember that a Docker
image includes everything needed to run an application: code, runtime libraries, and a private
filesystem. It is inside the Dockerfile that these components are specified.

See the getting started docs for Docker and this post on the importance of containers to get more
perspective. You do not have to make all of your containers using a Dockerfile. Docker Hub is a
service provided by Docker for finding and sharing container images. It is likely that people you
know are running Docker already, which will probably become an additional source of container images
and best practices.

More materials
[An introduction to Docker essentials](https://developer.ibm.com/courses/docker-essentials-a-developer-introduction/)
[Creating a database in a Docker Container](https://developer.ibm.com/tutorials/docker-dev-db/)
[Tweaking Docker for macOS](https://docs.docker.com/docker-for-mac/#preferences-menu)


# FLASK

The Python package Flask is a micro framework. We will use it during the tutorial to construct an
API that has \verb|train|train and \verb|predict|predict end points. We will deploy our Flask app as
part of a Docker container to reaffirm a production best practice. There are alternatives to Flask
like Django, web2py, and FlashAPI, and they too would be containerized using the methods discussed
here if they were the preferred tool to generate an API.

If you have not installed Flask yet

```bash
~$ pip install -U flask
```
The \verb|hello world!|helloworld! for flask is the following.

```python
#!/usr/bin/env python

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)
```

1. save the code as a file for example main.py

2. run the file

```bash
~$ python main.py
```
3. navigate to http://127.0.0.1:5000/

Use \verb|control + c|control+c to stop the server.

Please go through the Flask tutorial if you are not already familiar with Flask or if you need a
refresher. Flask is a core tool for many data scientists when it comes to model deployment.


Before you get started. If you feel like you still need some practice getting a feel for docker try
the tutorial for beginners in the Docker tutorial, before starting this tutorial. Docker is
intuitive so going through a few examples will be all that it takes to get comfortable.

Docker Tutorials
This tutorial is loosely based on the second tutorial, Webapps with Docker, so going through both of
those tutorials along with this one will provide a lot of context for how to use Docker in a number
of different ways.

You will need to run through this tutorial with access to a terminal. Jupyter lab or an open
terminal will work. We will create some of the files you need from within this notebook, but Docker
is a command line tool.

Docker generally exists and works together with other tools, some of which we will cover in the next
course. As companies began embracing containers, like AAVAIL who uses containers for nearly all of
their services, there are challenges that arise in managing hundreds (or even thousands) of
containers across a distributed system. Container orchestration technologies like Kubernetes and
Docker Compose have emerged as front-runners to address this unique challenge. Additional tools like
Jenkins and Ansible can be used alongside container orchestration to automate deployment-related
tasks. Here are some of the important technologies that are commonly used in an environment with
numerous Docker containers:

Docker Compose - Compose is a tool for defining and running multi-container Docker applications.
With Compose, you use a YAML file to configure your application’s services.

Kubernetes - An open-source system for automating deployment, scaling, and management of
containerized applications.

Jenkins - A tool to help automate the model deployment process. Often used for Continuous
Integration/Continuous Delivery (CI/CD).

Ansible - A tool for automation of the provisioning of the target environment and to then deploy the
application

Additional resources
[Guide to the IBM Kubernetes service](https://www.ibm.com/cloud/kubernetes-service)
[IBM video explaining containerization](https://www.youtube.com/watch?v=0qotVMX-J5s)
[IBM tutorial on building Docker images](https://developer.ibm.com/tutorials/building-docker-images-locally-and-in-cloud/)

# QUIZ

.Question 1
True/False. It is reasonable to think of the commands in a Dockerfile as a step-by-step recipe on
how to build up a Docker image.

    False

Incorrect
Dockerfiles describe how to assemble a private filesystem for a container. The Docker container is a
running process that is kept isolated from the host and from other containers. A Docker image
includes everything needed to run an application: code, runtime libraries, and the private
filesystem.

https://docs.docker.com/get-started/part2/

Data at scale

When it comes to preparing a model for deployment there is a guiding set of steps that may be
useful.

Make it work Make it better Then make it faster Before moving into a high-performance computing
(HPC) environment like Apache Spark there are some optimizations that might improve performance of
models once deployed. In some cases the optimizations are enough to avoid the HPC environment
entirely. Some of the important Python packages for code optimization are:

Multiprocessing - This is a package in the standard Python library and it supports spawning
processes (for each core) using an API similar to the threading module. The multiprocessing package
offers both local and remote concurrency

Threading - Another package in the standard library that allows separate flows flow of execution at
a lower level than multiprocessing.

Subprocessing - Module allows you to spawn new processes, connect to their input/output/error pipes,
and obtain their return codes. You may run and control non-Python processes like Bash or R with the
subprocessing module.

mpi4py - MPI for Python provides bindings of the Message Passing Interface (MPI) standard for the
Python programming language, allowing any Python program to exploit multiple processors.

ipyparallel - Parallel computing tools for use with Jupyter notebooks and IPython. Can be used with
mpi4py.

Cython - An optimizing static compiler for both the Python programming language and the extended
Cython programming language It is generally used to write C extensions for slow portions of code.

PyCUDA. - Python package that allows parallel computing on GPUs via CUDA (Compute Unified Device
Architecture)

Supercomputers and parallel computing can help with model training, prediction and other related
tasks, but it is worth noting that there are two laws that constrain the maximum speed-up of
computing: Amdahl’s law and Gustafson’s law.

Dealing with data at scale, which is closely related to both code optimization and parallel
computing. Apache Spark, is an example of a cluster computing framework that enables to enable
parallel computing.

If we talk about scale in the context of a program or model we may be referring to any of the
following questions. Let the word service in this context be both the deployed model and the
infrastructure.

Does my service train in a reasonable amount of time given a lot more data?  Does my service predict
in a reasonable amount of time given a lot more data?  Is my service ready to support additional
request load?  Docker and containers Technologies that are commonly used in an environment with
numerous Docker containers.

Docker Compose - Compose is a tool for defining and running multi-container Docker applications.
With Compose, you use a YAML file to configure your application’s services.

Kubernetes - An open-source system for automating deployment, scaling, and management of
containerized applications.

Jenkins - Tool to help automate the model deployment process. Often used for Continuous
Integration/Continuous Delivery (CI/CD).

Ansible - A tool for automation to the provision of the target environment and to then deploy the
application

Watson Machine Learning Tutorial Watson Machine Learning (WML) is an IBM service that makes
deploying a model for prediction and/or training relatively easy. The Watson Machine Learning Python
client was used in this tutorial to connect to the service. You may train, test and deploy your
models as APIs for application development, then share the models with colleagues. In this tutorial
you saw how you could match your local environment to the requirements of the available runtime
environments in WML. You also have the option of iterating on models in Watson Studio and then using
nearly the same code those same models could be deployed.

# WEEK 2

This week is primarily focused on deploying models using Spark. The rationale to move to Spark
almost always has to do with scale, either at the level of model training or at the level of
prediction. Although the resources available to build Spark applications are fewer than those for
scikit-learn, Spark gives us the ability to build in an entirely scaleable environment. We will also
look at recommendation systems. Most recommender systems today are able to leverage both explicit
(e.g. numerical ratings) and implicit (e.g. likes, purchases, skipped, bookmarked) patterns in a
ratings matrix. The majority of modern recommender systems embrace either a collaborative filtering
or a content-based approach. A number of other approaches and hybrids exist making some implemented
systems difficult to categorize. We wrap the week up with our hands-on case study on Model
Deployment.

Key Concepts
Build a data ingestion pipeline using Apache Spark
Tune hyperparameters in machine learning models on Apache Spark
Deploy a ML algorithm with both training and prediction endpoints
Connect Apache Spark Streaming to a recommendation engine
Explain how collaborative filtering and content-based filtering work

Our Story
The AAVAIL team deploys services and models that belong mostly to three categories: scikit-learn,
Spark-MLlib, and TensorFlow. This unit focuses on machine learning models in Spark. Of the three
options, this is the easiest to set up when scale is an important consideration when deploying the
model as a service. Models that are used infrequently generally stand to benefit little by existing
in a Spark environment, unless the underlying data are already stored in a distributed file system.
Here you will see how the AAVAIL team creates machine learning services using Spark.


THE DESIGN THINKING PROCESS
How do you figure out what “scale” means in a design thinking project?

Your first clue is to look at the users and personas that are referenced in the hill statements that
direct the work of each design thinking team. If the users and/or personas mentioned in the hills
refer to huge populations of millions of people, then you’ll need to work with your technical teams
on ensuring that your machine learning solutions will have enough processing power to handle a high
number of concurrent users.

Obviously, another factor to consider is the type of model you are deploying. During the Prototype
and Test phases of the design thinking process you will have decided upon the algorithms you’ll be
deploying to production. A neural network-based approach will obviously require more resources than
a logistic regression approach. These are the considerations you’ll be bringing up during playback
sessions with architects and engineers.

Model complexity and having millions of users will obviously affect model performance. What are the
end users expecting in terms of response times? Note that during prototyping and testing, you’ll be
required to deploy your test models to scalable platforms in order to assess user expectations about
response time for predictions and results from your models. A platform such as Apache Spark allows
you to easily scale your models to acheive the performance expectations of your end users.

MLlib is Spark’s machine learning (ML) library. Spark ML is not an official name, but we will use it
to refer to the MLlib DataFrame-based API that embraces ML pipelines.

Official Spark MLlib documentation
Official Spark ML pipelines
There should be a logical reason to use machine learning models in Spark if that is the technology
to be used for model deployment. There is some amount of additional work required when implementing
a model in the Spark environment when compared to scikit-learn. The rationale to move to Spark
almost always has to do with scale, either at the level of model training or at the level of
prediction. For model training, you generally have access to a distributed resource for data
storage. Because the computation is carried out in parallel, you can often expect faster train times
when compared to a single-node server. There is some amount of overhead in coordinating across
distributed resources, so a speed improvement is not guaranteed. The resources available to build
Spark applications are fewer than those for scikit-learn, but the ability to build in a entirely
scaleable environment is very powerful.

Spark Pipelines
DataFrame - This ML API uses DataFrame from Spark SQL as an ML dataset which can hold a variety of
data types. For example, a DataFrame could have a mixture of continuous and categorical data types.

Transformer - A Transformer is an algorithm which can transform one DataFrame into another
DataFrame. For example, a ML model is a Transformer which transforms a DataFrame with features into
a DataFrame with predictions.

Estimator - An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer
For example, a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

Pipeline - A Pipeline chains multiple Transformers and Estimators together to specify an ML
workflow.

Parameter - All Transformers and Estimators now share a common API for specifying parameters.

Feature extraction and transformations are part of the AI work flow. These topics were covered in
module 3. The feature extraction Spark docs can provide insight into the many available
transformations. Many of these will be familiar from working with them in scikit-learn like: a ChiSq
feature selection and a Spark StandardScaler. There are also available transformations specific to
natural language processing applications like Term frequency-inverse document frequency (TF-IDF) and
Spark Word2Vec.

Spark MLlib has a number of available supervised learning algorithms—specifically those used for
classification and regression. Many of the commonly used algorithms have been implemented including:
random forests, gradient boosted trees, linear support vector machines and even basic multilayer
perceptrons.

Spark MLlib - supervised learning
Spark Mlib has fewer models and algorithms to choose from compared to scikit-learn’s supervised
learning, but many of the most popular methods are present. Both random forests and gradient boosted
trees are models used in production and should be on your radar when comparing models. They both use
decision trees as a base model.

A minimal example for a random forest would look something like this:

```python
import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = (ps.sql.SparkSession.builder
        .appName("aavail-churn")
        .getOrCreate()
        )

sc = spark.sparkContext

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("./work/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

```

```python
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[121,122,123...|
|           0.0|  0.0|(692,[122,123,124...|
|           0.0|  0.0|(692,[122,123,148...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[124,125,126...|
+--------------+-----+--------------------+
only showing top 5 rows

Test Error = 0
RandomForestClassificationModel (uid=RandomForestClassifier_c6e91457acc5) with 10 trees

```

Spark MLlib has several available tools for unsupervised learning—namely dimension reduction and
clustering. For clustering, K-means and Gaussian Mixture Models (GMMs) are the main tools. Latent
Dirichlet Allocation (LDA) is available as a tool for clustering over documents of natural language.
This is a particularly important tool since the size of NLP datasets can often make single-node
computation challenging.

Spark clustering documentation
For dimension reduction, two of the most frequently used tools are PCA and the Chi-Squared Feature
Selector. All of the tools in the unsupervised learning category take the form of a transformer or
an estimator and, in keeping with the scikit-learn API, they too can be assembled in pipelines.

MLlib Estimators and Transformers use the same API for specifying parameters. There are two basic
methods to pass parameters.

Param - A named parameter with self-contained documentation.

ParamMap - Is a set of (parameter, value) pairs.

Each of these are provided in a slightly different way to an algorithm:

1. Set parameters for an instance. For example, for the gradient boosted tree classifier.

```python
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
```
or

```python
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
gbt.setMaxIter(20)
```
2. Pass a ParamMap to \verb|.fit()|.fit() or \verb|.transform()|.transform(). Any parameters in the
   ParamMap will override parameters previously specified via setter methods.

```python
model <- spark.gbt(training, label ~ features, "classification")
model.setMaxIter(10)

paramMap = {model.maxIter: 20}

## update a paramMap
paramMap[model.maxIter] = 30

model_fit = model.fit(training, paramMap)
```

Model tuning in Spark MLlib
A train-test split can be carried out with TrainValidationSplit. Cross Validation is accomplished in
Spark MLlib using the CrossValidator() object. A data set is split into a set of folds which are
used as separate training and test datasets. The CrossValidator computes the average evaluation
metric for the k models produced by fitting the Estimator on the k different (training, test)
dataset pairs. This helps identify the best ParamMap, which is then used to re-fit the Estimator
with the entire dataset.

Spark MLlib - model tuning
Additional resources
Example notebook that uses Watson Machine Learning and Spark MLlib
IBM Spark MLlib Modeler

# QUIZ
1.Question 1
True/False. The Spark ML API uses DataFrames from Spark SQL. They can hold a variety of data types
with different columns for storing text including: feature vectors, truth labels, and predictions.

    False

Incorrect
Spark ML, the DataFrame based API, is often referred to uses Spark DataFrames. The DataFrames, like
those in R and scikit-learn, can hold mixed data types.

Our Story
AAVAIL’s goal is to attract and retain video feed subscribers. AAVAIL plans to massively expand its
catalog of videos. This wider selection should help to retain customers because there will always be
new content for them to watch, but it comes at the risk of frustrating customers if they are
overwhelmed with choices that don’t interest them. Making good recommendations is key to retaining
customers, but the difficulty and complexity of making good recommendations increases dramatically
as the size of the catalog increases. It’s relatively easy to make good recommendations if everyone
has seen (and rated) the same ten videos. But the task is much more complicated if everyone has seen
a different set of ten videos out of a catalog of ten thousand.

You are expected to lead the research and development that will result in one or more deployed
recommender systems. With this in mind, you are expected to refresh your knowledge of recommender
systems and prepare a code base in preparation for the near future when the larger catalog of videos
and accompanying ratings become available.


THE DESIGN THINKING PROCESS
Design Thinking is ideally suited for coming up with a solution to the challenge of deciding what a
good recommender would look like for AAVAIL’s end users. Creating and defining persona descriptions
for each type of end user would have been part of the early design thinking process. Workshops with
actual AAVAIL subscribers would have included activities such as “As-Is” exercises to determine what
subscribers currently do. Needs statements, Empathy Maps, and Hopes and Fears would have been
captured and documented as well, to build a more complete picture of the subscriber base.

Those data would have informed your initial models for recommender systems, and those initial models
(along with their user interfaces) would be presented to actual subscribers during the Prototype
phase. Their feedback would have informed your design of the next iteration of models, and you would
have continued on this cycle of prototyping and testing until the end users were satisfied with the
results.

This process emphasizes how important it is to be able to make rapid changes to models for
prototyping purposes. If you end up having to use a neural network-based model, you’ll need to
inform the rest of the team of the time and effort required to re-train new iterations of the model.

In terms of a business opportunity we can consider the use of a recommender system when any of the
following questions are relevant:

What would a user like?
What would a user buy?
What would a user click?
There are other situations where recommendations might be appropriate outside of the scope of these
questions. One example would be if the AAVAIL team wanted to recommend words or phrases for an
autofill feature that is part of the company’s website or app. To consider a recommender system, we
need appropriate data. This most often comes in the form of a ratings matrix, sometimes known as a
utility matrix. Here is what a piece of a ratings matrix might look like for AAVAIL data.

User    Feed 1    Feed 2    Feed 3    Feed 4    Feed 5    Feed 6    Feed 7    Feed 8    Feed 9
1    ?    ?    4    ?    ?    ?    1    ?    ?
2    ?    ?    4    ?    ?    ?    1    ?    2
3    ?    ?    4    5    ?    1    1    3    ?
4    3    2    ?    ?    ?    1    ?    ?    ?
5    1    4    4    ?    ?    1    1    ?    ?
Notice that the majority of entries are missing, as is typical with utility matrices. We can’t
expect that every user has watched every feed, or even a signficant portion of them. Most User/Feed
intersections will be unrated, or blank, resulting in a sparse matrix.

Ratings come in two types: explicit and implicit. The above utility matrix contains explicit ratings
because the users rated feeds directly. Implicit data is derived from a user’s behaviors or actions
for example likes, shares, page visits or amount of time watched. These can also be used to
construct a utility matrix. Keeping with our AAVAIL feed example, we can engineer a measure based on
indirect evidence. For example the score for \verb|Feed 1|Feed1 could be based on a user’s location,
comment history, preferred type of feed, specified topic preferences and more. Each element that
contributes to the overall score could have a maximal value of 1.0 and the final number could be
scaled to a range of 1-5. Explict and implicit data can be combined using this type of approach as
well and naturally you would want to have a solid understanding of user stories before engineering a
score.

Most recommender systems today are able to leverage both explicit (e.g. numerical ratings) and
implicit (e.g. likes, purchases, skipped, bookmarked) patterns in a ratings matrix. The SVD++
algorithm is an example of a method that exploits both patterns [1].

Recommendation systems
The majority of modern recommender systems embrace either a collaborative filtering or a
content-based approach. A number of other approaches and hybrids exist making some implemented
systems difficult to categorize.

Collaborative filtering - Collaborative filtering is a family of methods that identfiy a subset of
users who have preferences similar to the target user. From these, a ratings matrix is created. The
items preferred by these users are combined and filtered to create a ranked list of recommended
items.

Content-based filtering - Predictions are made based on the properties and characteristics of an
item. User behavior is not considered.

Matrix factorization is a class of collaborative filtering algorithms used in recommender systems.
Matrix factorization algorithms work by decomposing the user-item interaction matrix into the
product of lower-dimension matrices. In general, the user-item interaction matrix will be very, very
large, and very sparse. The lower-dimension matrices will be much smaller and denser and can be used
to reconstruct the user-item interaction matrix, including predictions where values were previously
missing.

Matrix factorization is generally accomplished using Alternating Least Squares (ALS) or Stochastic
Gradient Descent (SGD). Hyperparameters are used to control regularization and the relative
weighting of implicit versus explicit ratings matrices. With recommender systems we are most
concerned with scale at prediction. Because user ratings change slowly, if at all, the algorithm
does not need to be retrained frquently and so this can be done at night. For this reason, Spark is
a common platform for developing recommender systems. The computation is already distributed under
the Spark framework so scaling infrastructure is straightforward.

There are several Python packages available to help create recommenders including surprise. Because
scale with respect to prediction is often a concern for recommender systems, many production
environments use the implementation found in Spark MLlib. The Spark collaborative filtering
implementation uses Alternating least Squares.

----------

[1]: Yancheng Jia, Changhua Zhang, Qinghua Lu, and Peng Wang. Users’ brands preference based on
SVD++ in recommender systems. 2014 IEEE Workshop on Advanced Research and Technology in Industry
Applications (WARTIA), pages 1175–1178, 2014.

One issue that arises with recommender systems in production is known as the cold-start problem.
There are two scenario when it comes to the cold start problem:

What shall we recommend to a new user?

If the recommender is popularity-based then the most popular items are recommended and this is not a
problem. If the recommender is similarity-based, the user could rate five items as part of the
sign-up or you could attempt to infer similarity based on user meta-data such as age, gender,
location, etc. Even if recommendations are based on similarities, you may still use the most popular
items to get the user started, but you would likely want to customize the list possibly based on
meta-data.
How should we treat a new item that hasn't been reviewed?

In order to make good recommendations, you need data about how users review the item. But until the
item has been recommended, it’s unlikey that users will review it. To overcome this dilemma, the
item could be randomly suggested for a trial period to collect data. You could put it in a special
section such as new releases to gauge initial interest. You can also use meta-data associated with
the item to find similar items and infer its recommendations from these similar items.
Concurrency can be a challenge for recommender systems. A recommender might, for example, find the
20 closest users based on latent factor profiles. From those users it would identify a list of
potential recommendations that could be sorted and filtered given what is known about the user. The
distances between users can often be pre-computed to speed up the recommendations because user
characteristics change slowly. Nevertheless this process has a few steps to it that require a burst
of compute time. If five users hit the service at the same time, there is the possibility that the
processors get weighed down with these simultaneous requests and recommendations become unusually
slow.

Spark can help get around the problem of concurrency because it has a cluster manager that handles
the distribution of compute tasks. The package celery, which works well with Flask can also be used
to mitigate this problem. If concurrency could be an issue in a system that you develop, even one
based on Spark, it is worth taking the time to simulate the problem. Send a batch of requests over a
set of pre-defined intervals and then plot times to response.

Additional resources
Recommender systems approaches and algorithms

1.Question 1
Which type of recommender system is readily available in Spark Machine learning?

Incorrect
Spark MLlib or Spark ML implements a collaborative filtering based recommender system. To learn
more: https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html

Our Story
The senior members of the data science team at AAVAIL have mentioned that there are three ways you
should look at every model: prediction, training and versioning. Doing this will help you ask the
right questions when it comes to preparing a model for production. One of the most important
questions is, “What kind of scale do we expect?”

In this case study we will connect several of the technologies discussed in this course including
Docker and Spark to create an environment where you can precisely control model training, model
prediction and model versioning. Along the way, you will learn several best practices when it comes
to putting recommenders in production.


THE DESIGN THINKING PROCESS
This case study will allow you to learn the actual methods you’ll be working with in a fast-moving
Design Thinking project.

Keep in mind that you’re using Apache Spark to make scaling the model much easier, and Docker will
help you to more quickly deploy different versions of your models. These skills are important
because you’re focused on the end user, and building solutions for end users in a Design Thinking
project means you have to move quickly.

## CASE STUDY

Ensure that Docker is working before attempting the following command. This command will download a
Docker image and run a Spark environment locally through Docker.

```bash
~$ $ docker run --name sparkbook -p 8881:8888 -v "$PWD":/home/jovyan/work jupyter/pyspark-notebook
start.sh jupyter lab --LabApp.token=''
```

It will take a few minutes to download the image the first time you run it. Once it is running you
can access Jupyter Lab at http://localhost:8881 in your local browser.

The container was named 𝚜𝚙𝚊𝚛𝚔𝚋𝚘𝚘𝚔, but feel free to name it what you would like
The -𝚙 flag is exposing port 8881, so as not to collide with any other notebooks you have running.
The -𝚟 flag connects the filesystem in the container to your computer’s filesystem.
start.sh is a script in the container that allows for notebook configuration
--𝙻𝚊𝚋𝙰𝚙𝚙.𝚝𝚘𝚔𝚎𝚗=' ' turns off secured access (for dev and debug only)
If you want to run the server in the background, add the docker flag \verb|-d|-d to the above
command.

Note: If you want to make your entire home folder visible to the docker container, navigate to ~
before executing the docker run command.

To stop the notebook use.

```bash
~$ docker stop sparkbook
```

A note on data access
These definitions are from the Docker storage documentation

Volumes - are stored in a part of the host filesystem which is managed by Docker. Non-Docker
processes should not modify this part of the filesystem. Volumes are the best way to persist data in
Docker.

Bind mounts - may be stored anywhere on the host system. They may even be important system files or
directories. Non-Docker processes on the Docker host or a Docker container can modify them at any
time.

You are ready to get started. Download the following notebook into the directory where you executed
the docker run command. You should see it in the work folder.

To begin the case study
1. Download the following notebook and data to a local folder that you will have as the base for
   your case study.

2. Use the docker run command above to initialize a Spark environment. The navigate to
   http://localhost:8881 in your local browser.

3. From the Jupyter Lab menu navigate to the downloaded notebook (most likely in the folder
  \verb|work|work) and get started.

Additional resources

[Docker storage volumes](https://docs.docker.com/storage/volumes/)
[User guide for the Spark Docker image](https://jupyter-docker-stacks.readthedocs.io/en/latest/)

# QUIZ
Spark containers run a private file system that is isolated from the host and other containers. What
is the suggested way to access notebooks and scripts from within the container?

Volumes are the preferred mechanism for persisting data generated by and used by Docker containers.
While bind mounts are dependent on the directory structure of the host machine, volumes are
completely managed by Docker. Volumes have several advantages over bind mounts described in the
volumes docs.

# RECAP
Spark Machine Learning
There are two APIs for Spark MLlib. The RDD-based API and the dataframe based API, which is often
referred to as Spark ML. Each has its own documentation.

Spark MLlib docs
Spark ML docs
Spark MLlib has a suite of available tools for unsupervised learning—namely dimension reduction and
clustering. For clustering K-means and Gaussian Mixture Models (GMMs) are the main tools. Latent
Dirichlet Allocation (LDA) is available as a tool for clustering over documents of natural language.

Spark clustering documentation
Spark MLlib has a number of available supervised learning algorithms that is classification and
regression. Many of the commonly used algorithms have been implemented including: random forests,
gradient boosted trees, linear support vector machines and even basic multilayer perceptrons.

Spark MLlib - supervised learning
Spark Recommenders
The majority of modern recommender systems embrace either a collaborative filtering or a
content-based approach. A number of other approaches and hybrids exists making some implemented
systems difficult to categorize.

Collaborative filtering - Based off of a ratings matrix collaborative filtering is a family of
methods that infers a subset of users that have behavior similar to a particular user. The items
preferred by these users are combined and filtered to create a ranked list of recommended items.

Content-based filtering - Predictions are made based on the properties/characteristics of an item.
User behavior is not considered.

Matrix factorization is a class of collaborative filtering algorithms used in recommender systems.
Matrix factorization algorithms work by decomposing the user-item interaction matrix into the
product of lower dimensionality matrices.

There are several Python packages available to help create recommenders including surprise. Because
scale with respect to prediction is often a concern for recommender systems many production
environments use the implementation found in Spark MLlib. The Spark collaborative filtering
implementation uses Alternating least Squares.

CASE STUDY: model deployment
We used a Docker image to create a local Spark environment. In this environment a recommender
systems was created using Spark MLlib’s collaborative filtering implementation. The model itself was
tuned, by modifying hyperparameters and by toggling between implicit and explicit versions of the
underlying algorithm. 𝚜𝚙𝚊𝚛𝚔-𝚜𝚞𝚋𝚖𝚒𝚝 was used to simulate a deployed model.
