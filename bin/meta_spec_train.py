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

def train(args):    
    df_microbe = pd.read_csv(args.microbe)
    df_labels = pd.read_csv(args.label)
    data = pd.merge(df_microbe, df_labels, how='left')
    if len(args.host) > 0:
        df_hosts = pd.read_csv(args.host)
        data = pd.merge(data, df_hosts, how='left')
    
    target = [x for x in df_labels.columns if x!="SampleID"]  
    
    dense_features = [x for x in df_microbe.columns if x!="SampleID"]
    sparse_features = []
    if len(args.host) > 0:
        sparse_features = [x for x in df_hosts.columns if x!="SampleID"]
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size = data[feat].max()+1, embedding_dim=args.embedding_size)
                                      for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
    else:
        dnn_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
        
    feature_names = get_feature_names(dnn_feature_columns)

    train_model_input = {name: data[name] for name in feature_names}
    train_labels = data[target].values
       
    train_model = MetaSpec(dnn_feature_columns, dense_num=len(dense_features), 
                           num_tasks=args.task_num, num_experts=args.n_expert,
                           dnn_hidden_units=args.hidden_units,
                       tasks=['binary']*args.task_num, device=device)
    train_model.compile("adagrad", loss='binary_crossentropy')    

    for epoch in range(2):
        history = train_model.fit(train_model_input, train_labels, batch_size=args.batch_size, epochs=args.n_epoch//2, verbose=1)

    train_model.eval()
    gate_ = train_model.gate_list   
    torch.save(train_model, args.o+'/train/meta_spec.model')
    

    ## prepare for msi
    x_input = torch.from_numpy(data[sparse_features+dense_features].values).to(device).to(torch.float32)
    embedding_dict = train_model.embedding_dict
    sparse_embedding_list, dense_value_list = train_model.input_from_feature_columns(x_input, dnn_feature_columns, embedding_dict)

    sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    feature_index = build_input_features(dnn_feature_columns)

    train_dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    with open(args.o+'/train/dnn_input_train.pickle', 'wb') as f:
        pickle.dump(train_dnn_input, f)
    
    ## save diseases correlation
    gate_dict = {}
    for i in range(len(target)):
        gate_dict[target[i]] = np.array(gate_[i].detach().cpu())
    df_corr = pd.DataFrame(gate_dict).corr()
    df_corr.to_csv(args.o+"/train/disease_corr.csv")
    
    with open(args.o+"/train/disease_name.pickle", "wb") as f:
        pickle.dump(target, f)
        

def main():
    args = parse_args()
    if not os.path.exists(args.o+'/train/'):
        os.makedirs(args.o+'/train/')
    train(args)
        

if __name__=='__main__':
    main()


    
