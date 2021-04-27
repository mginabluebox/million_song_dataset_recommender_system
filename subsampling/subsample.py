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


def main(spark, netID, fraction):
	# try:
	# 	print(spark.conf.get('spark.executor.memory'))
	# 	print(spark.conf.get('spark.driver.memory'))
	# except:
	# 	pass

	# specify in and out paths
	inpath = 'hdfs:/user/bm106/pub/MSD/'
	outpath = f'hdfs:/user/{netID}/final_project/subsample/'

	# read training file
	original = spark.read.parquet(inpath + 'cf_train_new.parquet')

	# subsample data
	sample = original.sample(fraction)
	sample.collect()
	save = str(round(fraction, 2)).split('.')[1]
	sample_outname = f'cf_train_{save}.parquet'
	sample.write.parquet(outpath+sample_outname)

	# TODO: sample training, validation and test files by user_id

	# TODO: repartition training file
	# original.repartition('user_id').write.parquet(outpath+'cf_train_repartitioned.parquet')

# Only enter this block if we're in main
if __name__ == "__main__":
	# change default spark context config
	# conf = [('spark.executor.memory', '10g'), ('spark.driver.memory', '10g')]
	# config = spark.sparkContext._conf.setAll(conf)
	# spark.sparkContext.stop()

	# Get user netID from the command line
	parser = argparse.ArgumentParser(description='Subsample the training dataset.')
	parser.add_argument('--netID',help='Enter your netID for constructing path to your HDFS.')
	parser.add_argument('--fraction',type=float,help='Enter fraction of the dataset to subsample.')
	args = parser.parse_args()

	netID = args.netID
	fraction = args.fraction

	# Create the spark session object
	spark = SparkSession.builder.appName('subsample').config('spark.blacklist.enabled',False).getOrCreate()

	# Call our main routine
	main(spark, netID, fraction)