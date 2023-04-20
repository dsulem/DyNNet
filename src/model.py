import sys

#sys.path.append('~/PycharmProjects/GraphSiamese/')
#sys.path.append('~/data/localhost/sulem/GraphSiamese/')
from src.embedding import GCN, MLP
import torch.nn as nn
import torch
import dgl


class GraphSiamese(nn.Module):
    """Architecture to detect if two graphs are different. """

    def __init__(self, embedding: nn.Module, similarity: str, pooling: str, loss: str, top_k=15, nlinear=2, nhidden=16,
                 dropout=0.1,
                 features=None):
        """

        :param embedding: A module that takes a DGL graph and outputs a n x d embedding.
        :param similarity: str 'euclidean' or 'cosine' for either nn.PairwiseDistance or nn.CosineSimilarity
        :param pooling: str 'average', 'topk', 'max' or 'avgraph' (Pooling layer).
        :param top_k: How many nodes the top_k layer will select
        :param loss: str 'bce' (sigmoid activation) or 'hinge' (tanh activation with topk pooling)
        """
        super(GraphSiamese, self).__init__()
        self.embedding = embedding
        self.input_dim = embedding.input_dim
        self.features = features
        if similarity.lower() == 'cosine':
            self.similarity = nn.CosineSimilarity()
            if pooling == 'topk':
                self.descending = False
        else:
            self.similarity = nn.PairwiseDistance()
            if pooling == 'topk':
                self.descending = True
        self.pooling = pooling
        if pooling == 'average' or pooling == 'avgraph':
            self.pooling_layer = dgl.nn.pytorch.glob.AvgPooling()
        elif pooling == 'max':
            self.pooling_layer = dgl.nn.pytorch.glob.MaxPooling()
        else:
            self.pooling_layer = dgl.topk_nodes
            self.top_k = top_k
            self.nlinear = nlinear
            self.mlp = MLP(nlinear, top_k, nhidden, 1, dropout=dropout)
            # self.linear = nn.Linear(top_k, 1)
        self.loss = loss

    def forward(self, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph):

        # Adds 1 self-loop per node if not there
        # graph1 = dgl.remove_self_loop(graph1)
        # graph1 = dgl.add_self_loop(graph1)
        #
        # graph2 = dgl.remove_self_loop(graph2)
        # graph2 = dgl.add_self_loop(graph2)

        graph1_encoding = self.embedding(graph1)
        graph2_encoding = self.embedding(graph2)

        with graph1.local_scope():

            if self.pooling == 'avgraph':  # compute a graph embedding before computing distance/similarity between embeddings
                graph1_encoding = self.pooling_layer(graph1, graph1_encoding)
                graph2_encoding = self.pooling_layer(graph2, graph2_encoding)

            # otherwise compute similarity/distance between the node-level embeddings
            similarity = self.similarity(graph1_encoding.squeeze(), graph2_encoding.squeeze()).unsqueeze(1)

            # then use a pooling layer to output the final similarity score/distance
            if self.pooling == 'average' or self.pooling == 'max':  # global average or max pooling
                x = self.pooling_layer(graph1, similarity).squeeze()

            elif self.pooling == 'topk':  # top k pooling
                graph1.ndata['similarity'] = similarity
                x, _ = self.pooling_layer(graph1, 'similarity', k=self.top_k, descending=self.descending)
                x = x.squeeze()
                if self.nlinear == 0:  # apply average pooling
                    x = torch.nn.AvgPool1d(x)
                else:  # apply MLP
                    if x.dim() < 2:
                        x = torch.unsqueeze(x, 0)
                    x = self.mlp(x)
            else:
                x = similarity.squeeze()

        return x

