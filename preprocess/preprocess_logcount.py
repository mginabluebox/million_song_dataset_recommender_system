#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import sys
import argparse

from functools import reduce

# And pyspark.sql to get the spark session
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

import math

def union_all(*dfs):
    return reduce(DataFrame.union, dfs)

def preprocess_data(data_paths, data_out_paths):
    dfs = [spark.read.parquet(data_path) for data_path in data_paths]
    original = union_all(*dfs) if len(dfs) > 1 else dfs[0]

    # index id
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",
                stringOrderType='alphabetAsc')
                for column in ['user_id','track_id'] ]
    pipeline = Pipeline(stages=indexers)
    
    def logplusone(count):
        return math.log10(1 + count/1e-8)

    logplusoneUDF = F.udf(lambda r: logplusone(r), T.FloatType())

    for i in range(len(dfs)):
        indexed_df = pipeline.fit(original).transform(dfs[i])

        #change float index to int
        indexed_df = indexed_df.selectExpr("cast(user_id_index as int) user_id_index",
                                           "cast(count as float) count",
                                           "cast(track_id_index as int) track_id_index")
        indexed_df = indexed_df.select(
            F.col('user_id_index'), F.col('track_id_index'),
            logplusoneUDF('count').alias('count')
        )
        indexed_df.write.parquet(data_out_paths[i])

        #show schema to check
        indexed_df.printSchema()

def main(spark, netID, fraction):
    out_path = f'hdfs:/user/{netID}/final_project/subsample/'
    if fraction:
        # specify path
        path = f'hdfs:/user/{netID}/final_project/subsample/'
        # read training file
        save = str(round(fraction, 2)).split('.')[1]
        data_name = f'cf_train_{save}.parquet'
        data_out_name = f'cf_train_{save}_indexed_logcount.parquet'
        preprocess_data([path + data_name], [out_path + data_out_name])
    else:
        path = f'hdfs:/user/bm106/pub/MSD/'
        data_types = ['train', 'validation', 'test']
        data_names = [f'cf_{data_type}.parquet' for data_type in data_types]
        data_out_names = [f'cf_{data_type}_indexed_logcount.parquet' for data_type in data_types]
        data_paths = [path + data_name for data_name in data_names]
        data_out_paths = [out_path + data_out_name for data_out_name in data_out_names]
        preprocess_data(data_paths, data_out_paths)

# Only enter this block if we're in main
if __name__ == "__main__":
    # change default spark context config
    # conf = [('spark.executor.memory', '10g'), ('spark.driver.memory', '10g')]
    # config = spark.sparkContext._conf.setAll(conf)
    # spark.sparkContext.stop()

    # Get user netID from the command line
    parser = argparse.ArgumentParser(description='Index the training dataset.')
    parser.add_argument('--netID',help='Enter your netID for constructing path to your HDFS.')
    parser.add_argument('--fraction',type=float,default=None,help='Enter fraction of the dataset.')
    args = parser.parse_args()

    netID = args.netID
    fraction = args.fraction

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocess').config('spark.blacklist.enabled',False).getOrCreate()

    # Call our main routine
    main(spark, netID, fraction)
