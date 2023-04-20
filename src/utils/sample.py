import argparse
import numpy as np
import networkx as nx
import torch
import pickle
import dgl
import os
from tqdm import trange
from os.path import join
import random
from typing import Union


def which_community(node, sizes):
    """
    Returns the community of a node i in a networkx SBM graph generated using
    `nx.generators.community.stochastic_block_model(sizes, p, nodelist=None)`.
    """
    return np.where(node < np.cumsum(sizes))[0][0]



def nxSBM(sizes, p, q, features=None, signal_means=None, signal_std=1.0):
    """Generates attributed SBM graphs with Networkx library.

    Creates a graph with sum(sizes) nodes. The nodes are ordered 0,1,...,n-1 and are also ordered by their community
    membership. I.e. the memberships will look like 0,0,...,0,0,1,1,....1,1,2,2,... etc.

    sizes: list of community sizes
    features: None or mixture of gaussian
    signal_means: list of vectors for each community. A vector represents the means of a multidimensional signal.
    p,q: intra/inter-cluster link probability
    signal_std: standard deviation of the signal
    """

    n = len(sizes)
    C = q * np.ones((n, n))
    np.fill_diagonal(C, p)
    graph = nx.generators.community.stochastic_block_model(sizes, C)

    if features == 'gaussian' and signal_means is not None:
        attributes = []

        for node in graph.nodes():

            community = which_community(node, sizes)
            loc = signal_means[community]
            if isinstance(loc, np.ndarray) or isinstance(loc, list):
                scale = signal_std * np.ones(len(loc))
            else:
                scale = signal_std * np.ones(1)
            signal = np.random.normal(loc=loc, scale=scale)
            graph.nodes[node]['node_attr'] = signal
            attributes.append(np.array([[graph.degree[node]]]))

        attributes = np.concatenate(attributes, axis=0)

        return graph, attributes

    return graph, None



def sample_dgl_graph(sbm_args):

    if sbm_args['features'] is not None:
        nx_graph, features = nxSBM(**sbm_args)
        adjacency = nx.to_scipy_sparse_matrix(nx_graph, format='csr')
        dgl_graph = dgl.from_scipy(adjacency)
        dgl_graph.ndata['node_attr'] = torch.FloatTensor(features)

    else:
        nx_graph, _ = nxSBM(**sbm_args) # if generated as arrays
        adjacency = nx.to_scipy_sparse_matrix(nx_graph, format='csr')
        dgl_graph = dgl.from_scipy(adjacency)

    return dgl_graph


def sample_pairs(seq, labels, nsamples=np.Inf, filename=None):
    """
    Samples pairs of graphs with labels (0=dissimilar, 1=similar) from a sequence of graphs
    :param seq: list of DGL graphs
    :param labels: array of labels of each graph indicating their generative distribution
    :param nsamples: number of pairs to sample
    :param filename: if None do not save the pairs
    :return: list of triplets (G_1, G_2, label)
    """

    pairs = np.triu_indices(len(seq), k=1)

    if nsamples == np.Inf:
        nsamples = len(pairs[0])

    idx_pairs = np.random.choice(range(pairs[0].shape[0]), min(10*nsamples, len(pairs[0])), replace=False) # random sample of indices in pairs
    data = []
    # Control class balance
    npos = 0
    nneg = 0
    i = -1
    while npos + nneg < nsamples and i < idx_pairs.shape[0]-1:
            i += 1
            id1 = pairs[0][idx_pairs[i]]
            id2 = pairs[1][idx_pairs[i]]

            lab = torch.Tensor([(labels[id1] == labels[id2])])
            if lab < 0.05 and nneg < nsamples//2:
                nneg+=1
            elif lab > 0.05 and npos < nsamples//2:
                npos+=1
            else:
                pass
            rdm = random.random()
            if rdm < 0.5:
                data.append([seq[id1], seq[id2], lab])
            else:
                data.append([seq[id2], seq[id1], lab])
            #print(id1, id2, lab)

    print("{} positive and {} negative examples".format(npos, nneg))
    random.shuffle(data)

    # save data
    if filename is not None:
        os.makedirs('../data', exist_ok=True)
        with open(join('../data', f'{filename}.p'), 'wb') as f:
            pickle.dump(data, f)

    return data



def sample_pairs_in_window(sequence, labels, window_length=10, n_pairs=None, path=None):
    """
    Sample all or a subsample of pairs of graphs in a sequence using a sliding window

    """

    n_data = len(sequence)
    npos = 0
    nneg = 0

    pairs = []

    for i in range(window_length, n_data):
        for j in range(1, window_length + 1):

            lab = torch.Tensor([(labels[i] == labels[i-j]).astype(int)])

            if lab < 0.05:
                nneg += 1
                pairs.append((sequence[i], sequence[i-j], lab))
            elif lab > 0.05:
                npos += 1
                pairs.append((sequence[i], sequence[i-j], lab))
            else:
                pass

    print("{} positive and {} negative examples".format(npos, nneg))
    if n_pairs is not None:
        pairs = random.sample(pairs, n_pairs)

    random.shuffle(pairs)

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(pairs, f)

    return pairs


