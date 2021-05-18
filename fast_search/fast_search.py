import numpy as np
import pandas as pd
from numpy.linalg import norm
from annoy import AnnoyIndex
import scann
import sys
import json
import time

k = int(sys.argv[1])
ann = sys.argv[2]
num_users = 5
n_loops = 10

# specify directories
home_dir = '../model_outputs'
alpha = 40
rank = 200
regParam = 1
maxIter = 20
model_dir = f'{home_dir}/model_csv/MFImp_a{alpha}r{rank}_reg{regParam}_it{maxIter}'
target_dir = f'{home_dir}/target_csv'

user_factor_path = f'{model_dir}/userFactors.csv' 
item_factor_path = f'{model_dir}/itemFactors.csv'
target_path = f'{target_dir}/valid_targets.csv' # validation targets 
output_dir = f'time_outputs_scann'

# get user and item features
user_df = pd.read_csv(user_factor_path, converters={'features': eval}) # [id: Int, features: Array(Float)]
item_df = pd.read_csv(item_factor_path, converters={'features': eval}) # [id: Int, features: Array(Float)]
target_df = pd.read_csv(target_path, converters={'tgt_track_id_indices': eval}) # [user_id_index: Int, tgt_id_indices: Array(Int)]
target_users = target_df['user_id_index'].tolist()
target_user_set = set(target_users)

user_ids, item_ids = [], []
user_features, item_features = [], []
for index, row in user_df.iterrows():
    if row['id'] in target_user_set:
        user_ids.append(row['id'])
        user_features.append(row['features'])

for index, row in item_df.iterrows():
    item_ids.append(row['id'])
    item_features.append(row['features'])
    
user_features = np.vstack(user_features)
item_features = np.vstack(item_features)

# define classes
class LinearScanner:
    def __init__(self, item_features):
        self.item_features = item_features

    def get_nns_by_vector(self, user_feature, k):
        id_scores = [(i, np.dot(user_feature, item_feature))
                     for i, item_feature in enumerate(item_features)]
        id_scores.sort(key=lambda x: -x[1])
        return id_scores[:k]

class Retriever:
    def __init__(self, retriever_type, item_features, k):
        self.k = k
        self.retriever_type = retriever_type
        if retriever_type == 'bruteforce':
            self.rt = LinearScanner(item_features)
        elif retriever_type == 'annoy':
            feat_dim = item_features.shape[1]
            self.rt = AnnoyIndex(feat_dim, 'angular')
            for i, item_feature in enumerate(item_features):
                self.rt.add_item(i, item_feature)
            self.rt.build(feat_dim)
        elif retriever_type == 'scann':
            normalized_item_features = item_features / norm(item_features, axis=1)[:, np.newaxis]
            self.rt = scann.scann_ops_pybind.builder(normalized_item_features, self.k, "dot_product")\
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)\
            .score_ah(2, anisotropic_quantization_threshold=0.2)\
            .reorder(self.k*2).build()

    def query(self, user_feature):
        if self.retriever_type == 'bruteforce' or self.retriever_type == 'annoy':
            top_item_ids = self.rt.get_nns_by_vector(user_feature, self.k)
        elif self.retriever_type == 'scann':
            top_item_ids, distances = self.rt.search_batched([user_feature])
        return top_item_ids

    def query_all(self, user_features):
        all_top_item_ids = []
        for user_feature in user_features:
            top_item_ids = self.query(user_feature)
            all_top_item_ids.append(top_item_ids)
        return all_top_item_ids

if ann == 'annoy':
    annoy_rt = Retriever('annoy', item_features, k)    
    t = []
    start = time.process_time() 
    for i in range(n_loops): 
        annoy_rt.query_all(user_features[:num_users])
        t.append(time.process_time() - start)
        start = time.process_time()
elif ann == 'bruteforce':
    bf_rt = Retriever('bruteforce', item_features, k)
    t = []
    start = time.process_time() 
    for i in range(n_loops): 
        bf_rt.query_all(user_features[:num_users])
        t.append(time.process_time() - start)
        start = time.process_time()
elif ann == 'scann': 
    scann_rt = Retriever('scann', item_features, k)
    t = []
    start = time.process_time() 
    for i in range(n_loops): 
        scann_rt.query_all(user_features[:num_users])
        t.append(time.process_time() - start)
        start = time.process_time()

out_d = {'average':sum(t)/n_loops,\
    'timings':t,\
    'best':min(t),\
    'loops':n_loops,\
    'worst':max(t)}

with open(f'{output_dir}/{ann}_MFImp_a{alpha}r{rank}_reg{regParam}_it{maxIter}_k{k}.json', 'w') as fp:
    json.dump(out_d, fp)
    fp.close()
