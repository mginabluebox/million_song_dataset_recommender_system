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

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics

config = {
    '': {
        'k': 500,
        'ranks': [10, 40, 100, 200],
        'regParams': [0.01, 0.1, 1],
        'alphas': [1, 40],
        'maxIters': [10],
    },
    'logcount': {
        'k': 500,
        'ranks': [10, 40, 100, 200],
        'regParams': [0.01, 0.1, 1],
        'alphas': [1, 40],
        'maxIters': [10]
    }
}

def train(training, rank, regParam, maxIter, alpha):
    """
    Train ML-Implicit model
    """
    als = ALS(rank=rank, regParam=regParam,
              implicitPrefs=True,
              maxIter=maxIter,
              alpha=alpha,
              userCol="user_id_index",
              itemCol="track_id_index",
              ratingCol="count")
    model = als.fit(training)
    return model 

def predict(test_users, model, k):
    """
    Recommend top k items for test users
    """
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
    )
    return user_predictions

def evaluate(targets, predictions, metrics):
    """
    Evaluate by ranking metrics
    """
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

def compute_scores(targets, model, k, metrics):
    """
    Evaluate by ranking metrics given the model and k
    """
    users = targets.select('user_id_index')
    predictions = predict(users, model, k)
    scores = evaluate(targets, predictions, metrics=metrics)
    return scores

def main(spark, netID, fraction):
    sc = spark.sparkContext

    home_dir = f'hdfs:/user/{netID}/final_project'
    data_dir = f'{home_dir}/subsample'
    model_dir = f'{home_dir}/model'
    output_dir = f'{home_dir}/output'

    for data_name, cfg in config.items():
        data_sfx = '' if data_name == '' else '_' + data_name
        frac_sfx = '' if not fraction else '_' + str(round(fraction, 2)).split('.')[1]

        # Use subsampled data
        if fraction: 
            data_name = f'cf_train{frac_sfx}_indexed.parquet'
            original = spark.read.parquet(data_dir + data_name).dropna()
            (training, valid), test = original.randomSplit([0.8, 0.2]), None
        # Use full data
        else:
            train_data_name = f'cf_train_indexed{data_sfx}.parquet'
            valid_data_name = f'cf_validation_indexed{data_sfx}.parquet'
            test_data_name = f'cf_test_indexed{data_sfx}.parquet'
            training = spark.read.parquet(f'{data_dir}/{train_data_name}').dropna()
            valid = spark.read.parquet(f'{data_dir}/{valid_data_name}').dropna()
            test = spark.read.parquet(f'{data_dir}/{test_data_name}').dropna()
        
        valid_targets = valid.groupBy('user_id_index')\
            .agg(F.collect_set('track_id_index')
                  .alias('tgt_track_id_indices')).cache()
        #test_targets = test.groupBy('user_id_index')\
        #    .agg(F.collect_set('track_id_index')
        #          .alias('tgt_track_id_indices'))
        for alpha in cfg['alphas']: 
            for rank in cfg['ranks']:
                for regParam in cfg['regParams']:
                    for maxIter in cfg['maxIters']:
                        param_name = f'a{alpha}r{rank}_reg{regParam}_it{maxIter}'
                        cfg_name = f'{data_sfx}{frac_sfx}_{param_name}'
                        model_name = f'MFImp{cfg_name}'

                        # Train model
                        model = train(training, rank=rank, regParam=regParam, alpha=alpha, maxIter=maxIter)

                        # Evaluate model
                        metrics = ['MAP', 'NDCG@500']
                        valid_scores = compute_scores(valid_targets, model, cfg['k'], metrics)
                        #test_scores = compute_scores(test_targets, model, cfg['k'], metrics)

                        # Output & Save result
                        valid_score_str = 'Validation: ' + \
                            ' '.join([metric+f'={score}' for metric, score in zip(metrics, valid_scores)])
                        print(valid_score_str)
                        #test_score_str = 'Test: ' + \
                        #    ' '.join([metric+f'={score}' for metric, score in zip(metrics, test_scores)])
                        #print(test_score_str)
                        
                        score_rdd = sc.parallelize([valid_score_str + '\n'])
                        score_rdd.coalesce(1).saveAsTextFile(f'{output_dir}/{model_name}')

                        # Save model
                        model.write().overwrite().save(f'{model_dir}/{model_name}')

# Only enter this block if we're in main
if __name__ == "__main__":

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
