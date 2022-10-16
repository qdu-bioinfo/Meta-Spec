import os
import sys
sys.path.append('../')
import logging
from config import parse_args

import pickle
import pandas as pd
import numpy as np
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features, combined_dnn_input

from model.MetaSpec import MetaSpec

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(args):    
    data = pd.read_csv(args.microbe)
    dense_features = [x for x in data.columns if x!="SampleID"]
    if len(args.host) > 0:
        df_hosts = pd.read_csv(args.host)
        data = pd.merge(data, df_hosts, how='left')
        
    with open(args.o+"/train/disease_name.pickle", "rb") as f:
        target = pickle.load(f)
    sparse_features = []
    if len(args.host) > 0:
        sparse_features = [x for x in df_hosts.columns if x!="SampleID"]
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size = data[feat].max()+1, embedding_dim=args.embedding_size)
                                      for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    feature_names = get_feature_names(dnn_feature_columns)
    test_model_input = {name: data[name] for name in feature_names}  
    train_model = torch.load(args.o+'/train/meta_spec.model')

    test_pred_ans = train_model.predict(test_model_input, batch_size=512) 
    test_pred_ans = test_pred_ans.transpose()    

    for i, disease in enumerate(target):
        data[disease] = test_pred_ans[i]
    data[["SampleID"]+target].to_csv(args.o+"/test/prediction.csv", index=False)
        
    x_input = torch.from_numpy(data[sparse_features+dense_features].values).to(device).to(torch.float32)
    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns, embedding_dict)

    sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)
    test_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    with open(args.o+'/test/dnn_input_test.pickle', 'wb') as f:
        pickle.dump(test_dnn_input, f)
        

def main():
    args = parse_args()
    if not os.path.exists(args.o+"/test/"):
        os.makedirs(args.o+"/test/")
    test(args)

if __name__=='__main__':
    main()

