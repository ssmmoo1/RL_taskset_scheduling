import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.function as fn
from torch.distributions import Categorical
from collections import deque

#Edge prediction model that uses RGCN and DP-Predictor
class TaskGCN_Dot(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

        #Save actions and rewards for REINFORCE
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, g, x, etype):
        h = self.sage(g,x) #pass through RGCN to get node embeddings

        predictions = self.pred(g, h, etype) #Predict scores for edges between processors and ready tasks

        predictions = torch.reshape(predictions, ((g.num_nodes("processor"), g.num_nodes("ready_task"))))
        predictions = F.softmax(predictions, dim=1)

        return predictions


class TaskGCN_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features, 1)

        #Save actions and rewards for REINFORCE
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, g, x, etype):
        h = self.sage(g,x) #pass through RGCN to get node embeddings

        predictions = self.pred(g, h, etype) #Predict scores for edges between processors and ready tasks

        predictions = torch.reshape(predictions, ((g.num_nodes("processor"), g.num_nodes("ready_task"))))
        predictions = F.softmax(predictions, dim=1)

        return predictions

#Create the gcn model
class RGCN(nn.Module):

    #Two layer Graph Neural Network
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, "mean") for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(hid_feats, out_feats, "mean") for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        return h #Returns node embeddings

#Calculates a score for a given edge type based on incident node embeddings
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            output = graph.edges[etype].data['score']
            return output


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']