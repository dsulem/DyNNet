import numpy as np
import networkx as nx
import torch
from src.utils.graphs import laplacian_embeddings, random_walk_embeddings, degree_matrix
from typing import Union
import json
from torch.utils.data import DataLoader
import pickle
import os
from src.model import GraphSiamese
from src.embedding import GCN
from src.utils.misc import collate, get_device



def dist_labels_to_changepoint_labels(labels: Union[np.ndarray, list]):
    """
    Convert graph distribution labels (phase) to change-point labels (0 or 1)

    :param labels (list or np.ndarray):
    :return: np.ndarray
    """

    if isinstance(labels, list):
        labels = np.array(labels)

    cps = np.concatenate([np.zeros(1).astype(int), (abs(labels[1:] - labels[:-1]) > 0).astype(int)],axis=0)

    return cps


def dist_labels_to_changepoint_labels_adjusted(labels: Union[np.ndarray, list], tolerance=2):
    """
    Convert graph distribution labels (phase) to change-point labels (0 or 1) using adjustment mechanism with level of tolerance

    :param labels:
    :param tolerance (int): flag as change points the timestamps at +/- tolerance around a change-point
    :return:
    """

    if isinstance(labels, list):
        labels = np.array(labels)

    cps = np.concatenate([np.zeros(1).astype(int), (abs(labels[1:] - labels[:-1]) > 0).astype(int)],axis=0)

    for i in range(1,tolerance+1):
        cps = (cps + np.concatenate([np.zeros(i), cps[:-i]], axis=0) + np.concatenate([cps[i:], np.zeros(i)], axis=0) > 0)

    return cps


def normalise_statistics(statistics):
    """
    Transform a statistic in [a,b] into [0,1] by substracting min and dividing by max

    :param statistics (list or nd.array):
    :return:
    """

    norm_stat = np.array(statistics)
    norm_stat = norm_stat - np.min(norm_stat)
    norm_stat = norm_stat / np.max(norm_stat)

    return norm_stat





def prepare_batches(data, window_length):

    tuples = []

    for i in range(window_length, len(data)):
        for j in range(1, window_length+1):
            tuples.append((data[i], data[i-j], i))
    batched_data = DataLoader(tuples, batch_size=window_length, shuffle=False, collate_fn=collate,
                               drop_last=False)

    return  batched_data

def load_sequence(datapath):
    if os.path.isfile(datapath):
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        time = None
        labels = None
    else:
        with open(datapath + '/data.p', 'rb') as f:
            data = pickle.load(f)

        with open(datapath + '/labels.p', 'rb') as f:
            labels = pickle.load(f)

        with open(datapath + '/time.json') as f:
            time = json.load(f)

    print(f"Data loaded: sequence of {len(data)} graphs with a change point at time {time}")

    return data, labels, time



def load_model(model_path: str, device=None):

    #model_path = PosixPath(model_path).expanduser()

    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            model_path = pickle.load(f)
        model_path = str(model_path)
        print("Last model trained loading...")

    with open(model_path + '/args.json', 'r') as f:
        args = json.load(f)

    embedding = GCN(input_dim=args['input_dim'], type=args['embedding_module'],
                    hidden_dim=args['hidden'], layers=args['nlayers'],
                    dropout=args['dropout'])
    model = GraphSiamese(embedding=embedding,
                         dropout=args['dropout'],
                         similarity=args['distance'],
                         loss=args['loss'],
                         pooling=args['pooling'],
                         top_k=args['top_k'],
                         nlinear=args['nlayers_mlp'],
                         nhidden=args['hidden'],
                         features=args['features'])
    model.load_state_dict(torch.load(model_path + '/model.pt', map_location='cpu'))
    if device is not None:
        model = model.to(device)
    model.eval()
    print("Model loaded")

    return model, args


def add_features(G, feat: str = 'degree', dim: int = 2):
    """
    Add features to a single graph

    """


    adjacency = G.adj(scipy_fmt='csr')

    if feat == 'laplacian':
        attributes = laplacian_embeddings(adjacency, k=dim)
    elif feat == 'random_walk':
        attributes = random_walk_embeddings(adjacency, k=dim)
    elif feat == 'degree':
        attributes = np.diag(degree_matrix(adjacency).todense(), k=0).reshape(-1,1)
    elif feat == 'identity':
        attributes = np.eye(adjacency.shape[0])
    else:
        raise ValueError('Type of features not recognised')

    G.ndata['node_attr'] = torch.FloatTensor(attributes)

    return G


def add_features_dataset(G, feat: str = 'degree', dim: int = 2):
    """
    Add features to a list of graphs

    """

    for i in range(len(G)):

        if isinstance(G[i], tuple) or isinstance(G[i], list):
            graph1, graph2, label = G[i]
            graph1, graph2 = add_features(graph1, feat, dim), add_features(graph2, feat, dim)
            G[i] = (graph1, graph2, label)
        else:
            graph = add_features(G[i], feat, dim)
            G[i] = graph


    return G