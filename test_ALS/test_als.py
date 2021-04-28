#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import sys
import argparse 

# And pyspark.sql to get the spark session
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

def main(spark, netID, fraction):

    # specify path
    path = f'hdfs:/user/{netID}/final_project/subsample/'

    # read training file
    save = str(round(fraction, 2)).split('.')[1]
    data_name = f'cf_train_{save}_indexed.parquet'
    original = spark.read.parquet(path + data_name)

    #split into train and test
    (training, test) = original.randomSplit([0.8, 0.2])
    
    #build model
    als = ALS(rank=10, regParam=0.01, userCol="user_id_index", itemCol="track_id_index", ratingCol="count",
          coldStartStrategy="drop")
    model = als.fit(training)

# Only enter this block if we're in main
if __name__ == "__main__":
    # change default spark context config
    # conf = [('spark.executor.memory', '10g'), ('spark.driver.memory', '10g')]
    # config = spark.sparkContext._conf.setAll(conf)
    # spark.sparkContext.stop()

    # Get user netID from the command line
    parser = argparse.ArgumentParser(description='Test ALS model.')
    parser.add_argument('--netID',help='Enter your netID for constructing path to your HDFS.')
    parser.add_argument('--fraction',type=float,help='Enter fraction of the dataset.')
    args = parser.parse_args()

    netID = args.netID
    fraction = args.fraction

    # Create the spark session object
    spark = SparkSession.builder.appName('subsample').config('spark.blacklist.enabled',False).getOrCreate()

    # Call our main routine
    main(spark, netID, fraction)
