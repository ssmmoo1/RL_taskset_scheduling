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
class TaskGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

        #Save actions and rewards for REINFORCE
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, g, x, etype):
        h = self.sage(g,x) #pass through RGCN to get node embeddings

        #TODO determine shape and make softwax work
        #TODO Need to do this per processor
        predictions = self.pred(g, h, etype) #Predict scores for edges between processors and ready tasks
        predictions = F.softmax(predictions, dim=1)

        return predictions

#Create the gcn model
class RGCN(nn.Module):

    #Two layer Graph Neural Network
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

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
            return graph.edges[etype].data['score']