# DSGA1004 - BIG DATA
## Final project
- The following document is adapted from project description by Prof Brian McFee

## Folder Structure
    .
    ├── subsampling
        └── subsample.py                             # python code for subsampling the training dataset
    ├── preprocess
        ├── preprocess.py                            # python code for converting the ids from String to Integer
        └── preprocess_logcount.py                   # python code for converting the ids from String to Integer with logarithmic smoothed count
    ├── run_ALS
        └── run.py                                   # python code for running MF-Implicit ALS model hyper-parameter tuning
    ├── test_ALS 
        └── test_als.py                              # python code for testing ALS model
    ├── output                                       # validation MAP/NDCG outputs
    ├── exploration
        └── exploration.ipynb			# python code for exploration 
    ├── fast_search
        ├── fast_search.ipynb                        # jupyter notebook for testing annoy and scann packages & implmenting brute-force method
        ├── fast_search.py                           # python version for submitting jobs on Greene
        ├── plot.ipynb                               # plot annoy, scann and bruteforce results 
        ├── run_fast_search_*.sbatch                 # sbatch files for submitting jobs on Greene
        ├── slurm_output                             # slurm stdout
        └── time_outputs_scann                       # process time in json format (time_outputs_old & time_outputs_new contain older results)
    └── cold_start 
        ├── make_feature_dataframe.ipynb             # feature extraction and aggregation
        ├── modeling.ipynb                           # feature engineering and multioutput regression modeling
        └── best_model_weights                       # weights for all 200 ridge regressors used in multioutput regression
    └── project_report
        ├── 1004_final_report.pdf             # final project report

# Recommender system
## The dataset

We used the [Million Song Dataset](http://millionsongdataset.com/) (MSD) collected by 
> Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
> The Million Song Dataset. In Proceedings of the 12th International Society
> for Music Information Retrieval Conference (ISMIR 2011), 2011.

The MSD consists of (you guessed it) one million songs, with metadata (artist, album, year, etc), tags, partial lyrics content, and derived acoustic features. 

The user interaction data comes from the [Million Song Dataset Challenge](https://www.kaggle.com/c/msdchallenge)
> McFee, B., Bertin-Mahieux, T., Ellis, D. P., & Lanckriet, G. R. (2012, April).
> The million song dataset challenge. In Proceedings of the 21st International Conference on World Wide Web (pp. 909-916).

The interaction data consists of *implicit feedback*: play count data for approximately one million users.
The interactions are partitioned into training, validation, and test sets, as described below.

  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`

Each of these files contains tuples of `(user_id, count, track_id)`, indicating how many times (if any) a user listened to a specific track.
For example, the first few rows of `cf_train.parquet` look as follows:

|    | user_id                                  |   count | track_id           |
|---:|:-----------------------------------------|--------:|:-------------------|
|  0 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRIQAUQ128F42435AD |
|  1 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRIRLYL128F42539D1 |
|  2 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       2 | TRMHBXZ128F4238406 |
|  3 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRYQMNI128F147C1C7 |
|  4 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRAHZNE128F9341B86 |

## Basic recommender system

Our recommendation model used Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.

We tuned the following hyper-parameters to optimize performance on the validation set: 

  - the *rank* (dimension) of the latent factors
  - the regularization parameter.

### Evaluation

We used ranking metrics MAP/NDCG to evaluate our model's accuracy on the validation and test data.

## Extensions

We implemented the following extensions on top of the baseline collaborative filter model. 

  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  We used [annoy](https://github.com/spotify/annoy) or [scann](https://github.com/google-research/google-research/tree/master/scann). Our report provides an evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.
  - *Exploration*: use the learned representation to develop a visualization of the items and users, e.g., using T-SNE or UMAP.  The visualization should somehow integrate additional information (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.
  - (Future Work) *Cold-start*: using the MSD's supplementary features (tags, acoustic features, etc), build a model that can map observable data to the learned latent factor representation for items. To evaluate its accuracy, we simulated a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.
