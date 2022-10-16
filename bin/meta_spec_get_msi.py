import os
import sys
sys.path.append('../')
import pickle
import logging
from config import parse_args

import pandas as pd
import numpy as np
import shap
import torch
import matplotlib.pyplot as pl
from model.MetaSpec_plot import MetaSpec
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features, combined_dnn_input

def get_msi(args):
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

    train_model = torch.load(args.o+'/msi/msi.model')
    
    pkl_file = open(args.o+'/train/dnn_input_train.pickle', 'rb')
    dnn_input_train = pickle.load(pkl_file)
    pkl_file = open(args.o+'/test/dnn_input_test.pickle', 'rb')
    dnn_input_test = pickle.load(pkl_file)

    print("generate msi values ...")
    e = shap.GradientExplainer(train_model, dnn_input_train)
    shap_values = e.shap_values(dnn_input_test)    

    shap_list = []
    for i in range(len(target)):
        sh = shap_values[i]
        shap_list.append(sh)

    shape_ls = []
    for i in range(len(target)):
        shape_ls.append([])
        l = []
        for k in range(len(sparse_features)):
            l.append(sum(shap_list[i][0][k*args.embedding_size:k*args.embedding_size+args.embedding_size]))
        l = l + list(shap_list[i][0][len(sparse_features)*args.embedding_size:])
        shape_ls[i].append(l)

    shape_ls = np.array(shape_ls)
    shap_ls_new = []
    for i in range(len(target)):
        shap_ls_new.append(shape_ls[i])
                       
    df_dict = {}
    for i in range(len(target)):
        df = pd.DataFrame(shap_ls_new[i])
        df.columns = sparse_features+dense_features
        d = {'features':[],'msi':[]}
        for j in dense_features+sparse_features:
            imp = abs(np.median(df[j]))
            d['features'].append(j)
            d['msi'].append(imp)
        sta = pd.DataFrame(d)
        sta['msi'] = sta['msi'].apply(lambda x: np.log(x+1/100000))
        sta['msi'] = sta['msi']-min(sta['msi'])
        sta['msi'] = sta['msi']/sum(sta['msi'])
        sta = sta.sort_values('msi', ascending=False).reset_index(drop=True)
        sta.to_csv(args.o + '/msi/' + target[i] + '_msi.csv',index=False)               
                 

    if args.is_plot:
        for i in range(len(target)):
            pl.clf()
            shap.summary_plot([shap_ls_new[i]], features = sparse_features+dense_features, 
                              class_names=[target[i]], max_display=args.max_plot, show=False)
            pl.savefig(args.o+'/msi/'+target[i]+'_top'+str(args.max_plot)+'.jpg', dpi=300, bbox_inches = 'tight')
        
def main():
    args = parse_args()
    if not os.path.exists(args.o+"/msi/"):
        os.makedirs(args.o+"/msi/")
    get_msi(args)

if __name__=='__main__':
    main()