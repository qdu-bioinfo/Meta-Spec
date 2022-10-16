
import torch
import torch.nn as nn
import os


from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer, CrossNet
from deepctr_torch.models.deepfm import *
from model.basemodel_plot import *
from layer.MMOELayer import MMOELayer


class MetaSpec(MyBaseModel):

    def __init__(self, dnn_feature_columns, num_tasks, tasks, dense_num, sparse_num, emb_size=128,
                 num_experts=7, cross_num=1, dnn_hidden_units=(128, 128),l2_reg_embedding=0.01, l2_reg=0.01, l2_cross=0.01, dr=0, seed=1024, device='cpu'):
        super(MetaSpec, self).__init__(linear_feature_columns=dnn_feature_columns, dnn_feature_columns=dnn_feature_columns,
                                       num_tasks=num_tasks, task = tasks,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.tasks = tasks
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.sparse_num = sparse_num
        self.cross_num = cross_num
        self.dense_num = dense_num
        self.emb_size = emb_size
        
        self.mmoe_layer = MMOELayer(self.compute_input_dim(dnn_feature_columns), num_tasks, num_experts, dnn_hidden_units, dr, device)
        self.cross_asv = CrossNet(dense_num, layer_num = cross_num,
                                   parameterization='matrix',device=device)
        self.add_regularization_weight(self.cross_asv.kernels, l2 = l2_cross)
        
        if self.compute_input_dim(dnn_feature_columns)-dense_num>0:
            self.cross_host = CrossNet(self.compute_input_dim(dnn_feature_columns)-dense_num, layer_num = cross_num,
                                       parameterization='matrix',device=device)
            self.add_regularization_weight(self.cross_host.kernels, l2 = l2_cross)

        
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mmoe_layer.expert_network.named_parameters()), l2=l2_reg)   
        tower_dim = dnn_hidden_units[-1]+self.compute_input_dim(dnn_feature_columns)
        self.tower_network = nn.ModuleList([nn.Linear(tower_dim, 1, bias=False).to(device) for _ in range(num_tasks)])
        
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        
        self.to(device)

    def forward(self, X):
        
        sparse_embedding_list = [X[:,i*self.embedding_size:i*self.emb_size+self.emb_size].reshape(-1,1,self.emb_size) for i in range(self.sparse_num)]
        
        dense_value_list = []
        dense_input = X[:,self.sparse_num*self.emb_size:].transpose(0,1).reshape(self.dense_num,-1,1)

        dense_value_list = [x for x in dense_input]
        
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        cross_out1 = self.cross_asv(dnn_input[:,-1*self.dense_num:])        
        mmoe_out, self.gate_list = self.mmoe_layer(dnn_input)
        if len(sparse_embedding_list)!=0:
            cross_out2 = self.cross_host(dnn_input[:,:-1*self.dense_num])
        
        task_outputs = []
        for i in range(self.num_tasks):  
            if len(sparse_embedding_list)!=0:
                tower_input = torch.cat((mmoe_out[i], cross_out1, cross_out2),1)     
            else:
                tower_input = torch.cat((mmoe_out[i], cross_out1),1)
            
            logit = self.tower_network[i](tower_input)
            y_pred = self.out[i](logit)
            task_outputs.append(y_pred)

        task_outputs = torch.cat(task_outputs, -1)
        
        return task_outputs
    
    