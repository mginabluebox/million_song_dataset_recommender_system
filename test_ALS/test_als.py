#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import os
import sys
import argparse 

# And pyspark.sql to get the spark session
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row

from pyspark.mllib.evaluation import RankingMetrics


def train(training, rank, regParam):
    als = ALS(rank=rank, regParam=regParam,
              implicitPrefs=True,
              userCol="user_id_index",
              itemCol="track_id_index",
              ratingCol="count")
              #coldStartStrategy="drop")
    model = als.fit(training)
    return model

def predict(test_users, model, k):
    user_predictions = model.recommendForUserSubset(test_users, k)
    # Output format:  number of users x k (100000 x 500)
    # user_id  | recommendations
    # ----------------------------------------------------------------------------
    # user_id1 | [(pre_item_id1, pred_rating1), (pred_item_id2, pred_rating2), ...]

    def extractRecTrack(rec):
        # udf => convert recommenations to [pred_item_id1, pred_item_id2, .....]
        return [row.track_id_index for row in rec]

    extractRecTrackUDF = F.udf(lambda r: extractRecTrack(r),
        T.ArrayType(T.IntegerType())
    )
    user_predictions = user_predictions.select(
        F.col('user_id_index').alias('p_user_id_index'),
        extractRecTrackUDF('recommendations').alias('rec_track_id_indices')
    )#.repartition(10000, 'p_user_id_index')
    return user_predictions

def evaluate(targets, predictions, metrics):
    # Rankingmetrics
    # RDD[([pred_item_id1, pred_item_id2, ....], [target_item_id1, target_item_id2, ....])]
    pred_and_targets = predictions.join(targets,
        F.col('p_user_id_index') == F.col('user_id_index')
    ).dropna().rdd\
    .map(lambda r: (r.rec_track_id_indices, r.tgt_track_id_indices))

    rk_metrics = RankingMetrics(pred_and_targets)
    scores = []
    for metric in metrics:
        if metric == 'MAP':
            score = rk_metrics.meanAveragePrecision
        elif 'AP@' in metric:
            k = int(metric.split('@')[-1])
            score = rk_metrics.precisionAt(k)
        elif 'NDCG@' in metric:
            k = int(metric.split('@')[-1])
            score = rk_metrics.ndcgAt(k)
        scores.append(score)
    return scores

def main(spark, netID, fraction):
    sc = spark.sparkContext

    # specify path
    path = f'hdfs:/user/{netID}/final_project/subsample/'
    model_path = f'hdfs:/user/{netID}/final_project/model/'
    pred_path = f'hdfs:/user/{netID}/final_project/prediction/'
    output_path = 'output/'

    # read training file
    if fraction:
        save = str(round(fraction, 2)).split('.')[1]
        data_name = f'cf_train_{save}_indexed.parquet'
        original = spark.read.parquet(path + data_name).dropna()

        # split into train and test
        (training, valid) = original.randomSplit([0.8, 0.2])
        test = None
    else:
        save = None
        train_data_name = f'cf_train_indexed.parquet'
        valid_data_name = f'cf_validation_indexed.parquet'
        test_data_name = f'cf_test_indexed.parquet'
        training = spark.read.parquet(path + train_data_name).dropna()
        valid = spark.read.parquet(path + valid_data_name).dropna()
        test = spark.read.parquet(path + test_data_name).dropna()

    k = 500
    ranks = [10, 20, 40]
    regParams = [0.001, 0.01, 0.1, 1]
    for rank in ranks:
        for regParam in regParams:
            # train MFImplicit
            if save:
                model_name = f'MFImp_frac{save}_r{rank}_reg{regParam}'
                pred_name = f'Pred_frac{save}_r{rank}_reg{regParam}.parquet'
                output_name = f'Output_frac{save}_r{rank}_reg{regParam}.txt'
            else:
                model_name = f'MFImp_r{rank}_reg{regParam}'
                pred_name = f'Pred_r{rank}_reg{regParam}.parquet'
                output_name = f'Output_r{rank}_reg{regParam}'

            model = train(training, rank=rank, regParam=regParam)
            #model.write().overwrite().save(model_path + model_name)
            
            # predict on validation users
            valid_targets = valid.groupBy('user_id_index')\
                .agg(F.collect_set('track_id_index')
                      .alias('tgt_track_id_indices'))#\
                #.limit(100000)#.repartition(10000, 'user_id_index')
            valid_users = valid_targets.select('user_id_index')
 
            valid_predictions = predict(valid_users, model, k)
            #valid_predictions.write.parquet(pred_path + pred_name)
            
            # evluate by ranking metrics
            metrics = ['MAP',
                       'AP@10', 'AP@20', 'AP@50', 'AP@100',
                       'NDCG@10', 'NDCG@20', 'NDCG@50', 'NDCG@100']
            scores = evaluate(valid_targets, valid_predictions, metrics=metrics)
            score_str = ' '.join([metric+f'={score}'
                                  for metric, score in zip(metrics, scores)])
            
            score_rdd = sc.parallelize([f'Validation: {score_str}\n'])
            score_rdd.coalesce(1).saveAsTextFile(output_path + output_name)


# Only enter this block if we're in main
if __name__ == "__main__":
    # change default spark context config
    # conf = [('spark.executor.memory', '10g'), ('spark.driver.memory', '10g')]
    # config = spark.sparkContext._conf.setAll(conf)
    # spark.sparkContext.stop()

    # Get user netID from the command line
    parser = argparse.ArgumentParser(description='Test ALS model.')
    parser.add_argument('--netID',help='Enter your netID for constructing path to your HDFS.')
    parser.add_argument('--fraction',type=float,default=None, help='Enter fraction of the dataset.')
    args = parser.parse_args()

    netID = args.netID
    fraction = args.fraction

    # Create the spark session object
    spark = SparkSession.builder.appName('als').config('spark.blacklist.enabled',False).getOrCreate()

    # Call our main routine
    main(spark, netID, fraction)
