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

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

def main(spark, netID, fraction):
	# try:
	# 	print(spark.conf.get('spark.executor.memory'))
	# 	print(spark.conf.get('spark.driver.memory'))
	# except:
	# 	pass

	# specify path
	path = f'hdfs:/user/{netID}/final_project/subsample/'

	# read training file
    save = str(round(fraction, 2)).split('.')[1]
    data_name = f'cf_train_{save}.parquet'
	original = spark.read.parquet(path + data_name)

	# index id
	indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",stringOrderType='alphabetAsc').fit(original) for column in ['user_id','track_id'] ]
	pipeline = Pipeline(stages=indexers)
    indexed_df = pipeline.fit(original).transform(original)
    
    #change float index to int
    indexed_df = indexed_df.selectExpr("cast(user_id_index as int) user_id_index", "cast(track_id_index as int) track_id_index").drop('user_id','track_id')
    data_out_name = f'cf_train_{save}_indexed.parquet'
    indexed_df.write.parquet(path + data_out_name)

# Only enter this block if we're in main
if __name__ == "__main__":
	# change default spark context config
	# conf = [('spark.executor.memory', '10g'), ('spark.driver.memory', '10g')]
	# config = spark.sparkContext._conf.setAll(conf)
	# spark.sparkContext.stop()

	# Get user netID from the command line
	parser = argparse.ArgumentParser(description='Index the training dataset.')
	parser.add_argument('--netID',help='Enter your netID for constructing path to your HDFS.')
	parser.add_argument('--fraction',type=float,help='Enter fraction of the dataset.')
	args = parser.parse_args()

	netID = args.netID
	fraction = args.fraction

	# Create the spark session object
	spark = SparkSession.builder.appName('subsample').config('spark.blacklist.enabled',False).getOrCreate()

	# Call our main routine
	main(spark, netID, fraction)
