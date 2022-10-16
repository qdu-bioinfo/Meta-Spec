import os
import sys
sys.path.append('../')
import logging
from config import parse_args

import pickle
import pandas as pd
import numpy as np

import torch
from model.MetaSpec_plot import MetaSpec
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features, combined_dnn_input

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_msi(args):    
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
    train_model = MetaSpec(dnn_feature_columns, num_tasks = args.task_num, 
                           dense_num = len(dense_features), sparse_num = len(sparse_features),
                           num_experts=args.n_expert, dnn_hidden_units=args.hidden_units,
                           emb_size = args.embedding_size,
                           tasks=['binary']*args.task_num, device=device)
    train_model.compile("adagrad", loss='binary_crossentropy')    

    for epoch in range(2):
        history = train_model.fit(None, train_labels, args.o, batch_size=args.batch_size, epochs=args.n_epoch//2, verbose=1)
        
    torch.save(train_model, args.msi_model)
        

def main():
    args = parse_args()
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    train_msi(args)
    
if __name__=='__main__':
    main()