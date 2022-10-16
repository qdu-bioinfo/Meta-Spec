import torch
import torch.nn as nn
from deepctr_torch.layers import DNN


class MMOELayer(nn.Module):
    def __init__(self, input_dim, num_tasks, num_experts, dnn_hidden_units, dr, device):
        super(MMOELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.dnn_hidden_units = dnn_hidden_units
        self.expert_network = nn.ModuleList([DNN(input_dim, dnn_hidden_units,
               activation='relu', l2_reg=0.1, dropout_rate=dr, use_bn=True,
               init_std=0.0001, device=device) for _ in range(self.num_experts)])
        
        self.gate_list = {}
        for g in range(self.num_tasks):
            self.gate_list[g] = []
        
        self.d = {}
        for i in range(self.num_tasks):
            self.d['gate_'+str(i)] = nn.Parameter(torch.rand(self.num_experts),requires_grad=True)
        self.gate = nn.ParameterDict(self.d)


    def forward(self, inputs):
        expert_list = []
        for i in range(self.num_experts):
            expert_out = self.expert_network[i](inputs)
            expert_list.append(expert_out)
            
        final_expert = torch.stack(expert_list,2)

        outputs = []
        for i in range(self.num_tasks):
            self.gate_list[i] = self.gate['gate_'+str(i)].softmax(0)
            out_ = final_expert*self.gate_list[i]       
            out = torch.sum(out_,2)
            outputs.append(out)
                    
        return outputs, self.gate_list