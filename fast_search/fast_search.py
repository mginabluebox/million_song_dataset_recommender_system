import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import sys
from IPython import get_ipython
import json
import time
from numpy.linalg import norm

k = int(sys.argv[1])
ann = bool(sys.argv[2])
num_users = 5

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
output_dir = f'{home_dir}/time_outputs'

user_df = pd.read_csv(user_factor_path, converters={'features': eval}) # [id: Int, features: Array(Float)]
item_df = pd.read_csv(item_factor_path, converters={'features': eval}) # [id: Int, features: Array(Float)]
target_df = pd.read_csv(target_path, converters={'tgt_track_id_indices': eval}) # [user_id_index: Int, tgt_id_indices: Array(Int)]
target_users = target_df['user_id_index'].tolist()
target_user_set = set(target_users)
user_features = {row['id']: np.array(row['features']) for index, row in user_df.iterrows() if row['id'] in target_user_set}

def cosine_similarity(a, b):
    return np.dot(a,b)

class LinearScanner:
    def __init__(self, item_df):
        self.item_df = item_df
        self.item_df['normalized_features'] = item_df['features'].apply(lambda x: x/norm(x))

    def get_nns_by_vector(self, user_feature, k):
        user_feature = user_feature/norm(user_feature)
        id_scores = [(row['id'], np.dot(user_feature, row['normalized_features']))
                     for index, row in self.item_df.iterrows()]
        id_scores.sort(key=lambda x: -x[1])
        return id_scores[:k]

class Retriever:
    def __init__(self, retriever_type, feat_dim, item_df):
        if retriever_type == 'bruteforce':
            self.rt = LinearScanner(item_df)
        elif retriever_type == 'annoy':
            self.rt = AnnoyIndex(feat_dim, 'angular') 
            for index, row in item_df.iterrows():
                item_id = row['id']
                item_feature = np.array(row['features'])
                self.rt.add_item(item_id, item_feature)
            self.rt.build(feat_dim)

    def query(self, user_feature, k):
        top_item_ids = self.rt.get_nns_by_vector(user_feature, k)
        return top_item_ids

    def query_all(self, user_features, k):
        all_top_item_ids = []
        for user_id, user_feature in user_features.items():
            top_item_ids = self.query(user_feature, k)
            all_top_item_ids.append(top_item_ids)
        return all_top_item_ids

selected_users = {k:user_features[k] for k in list(user_features.keys())[:num_users]}

n_loops = 7
if ann:
    annoy_rt = Retriever('annoy', rank, item_df)
    t = []
    start = time.process_time() 
    for i in range(n_loops): 
        annoy_rt.query_all(selected_users, k=k)
        t.append(time.process_time() - start)
        start = time.process_time()
    ann_str = 'annoy'
else:
    bf_rt = Retriever('bruteforce', rank, item_df)
    t = []
    start = time.process_time() 
    for i in range(n_loops): 
        bf_rt.query_all(selected_users, k=k)
        t.append(time.process_time() - start)
        start = time.process_time()
    ann_str = 'bruteforce'


out_d = {'average':sum(t)/n_loops,\
    'timings':t,\
    'best':min(t),\
    'loops':n_loops,\
    'worst':max(t)}

with open(f'{output_dir}/{ann_str}_MFImp_a{alpha}r{rank}_reg{regParam}_it{maxIter}_k{k}.json', 'w') as fp:
    json.dump(out_d, fp)
    fp.close()
