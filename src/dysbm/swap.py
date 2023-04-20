import argparse
import numpy as np
import networkx as nx
import torch
import pickle
import dgl
import os
from tqdm import trange
from os.path import join
from src.utils.sample import which_community, nxSBM
import random
from pathlib import Path, PosixPath
import math
import json
from pytorch_lightning.utilities.parsing import str_to_bool
from src.utils.graphs import laplacian_embeddings, random_walk_embeddings
from copy import deepcopy
import scipy.sparse as ss
import matplotlib.pyplot as plt
from datetime import datetime



def permute_swaps(A, F=None, swaps=[]):
    """ Permute nodes in a graph given by adjacency and features

    swaps: A list of pairs. Each pair of nodes is sequentially swapped.
    """
    A = A.toarray()
    A2 = deepcopy(A)

    if F is not None:
        F = F.toarray()
        F2 = deepcopy(F)
    else:
        F2 = None

    for i in range(len(swaps)):
        u, v = swaps[i]
        A2[u, :], A2[v, :] = A[v, :], A[u, :] # permute rows and columns
        A2[:, u], A2[:, v] = A[:, v], A[:, u] # permute columns
        if F is not None:
            F2[u, :], F2[v, :] = F[v, :], F[u, :]

    return ss.csr_array(A2), F2


def sample_swap(sbm_args, swaps=[]):

    if sbm_args['features'] is not None:
        graph, attributes = nxSBM(**sbm_args)

    else:
        graph = nxSBM(**sbm_args)
        attributes = None

    adjacency = nx.to_scipy_sparse_array(graph, format='csr')
    A2, F2 = permute_swaps(adjacency, attributes, swaps)

    A2 = (A2 > 1e-5).astype(int)

    graph = dgl.from_scipy(A2.tocoo())
    if sbm_args['features'] is not None:
        graph.ndata['node_attr'] = torch.FloatTensor(F2)

    return graph


def generate_pairs_memberships(args=None):

    sbm = {'sizes': args.sizes, 'signal_means': args.signal_means, 'p': args.p, 'q': args.q, "signal_std": args.signal_std,
           "features":args.features}

    data = []

    # negative label data points
    for _ in trange(args.n_samples, desc='Generating negative examples'):

        # choose random swaps
        number_swaps = int(args.permute_rate * sum(args.sizes)) // 2

        c1 = np.arange(args.sizes[0] + args.sizes[1])
        c2 = np.arange(args.sizes[0] + args.sizes[1], sum(args.sizes))

        np.random.shuffle(c1)
        np.random.shuffle(c2)

        #nodes = np.random.choice(np.arange(sum(args.sizes)), number_swaps, replace=False)
        swaps = []
        for i in range(number_swaps):
            swaps.append([c1[i], c2[i]])

        pair = [sample_swap(sbm), sample_swap(sbm, swaps)]
        random.shuffle(pair)
        graph1, graph2 = pair


        label = torch.tensor(0)
        data.append((graph1, graph2, label))

    # positive label data points
    for _ in trange(args.n_samples, desc='Generating positive examples'):

        # choose random swaps
        number_swaps = int(args.permute_rate * sum(args.sizes)) // 2

        c1 = np.arange(args.sizes[0] + args.sizes[1])
        c2 = np.arange(args.sizes[0] + args.sizes[1], sum(args.sizes))

        np.random.shuffle(c1)
        np.random.shuffle(c2)

        # nodes = np.random.choice(np.arange(sum(args.sizes)), number_swaps, replace=False)
        swaps = []
        for i in range(number_swaps):
            swaps.append([c1[i], c2[i]])

        pair = [sample_swap(sbm, swaps), sample_swap(sbm, swaps)]
        graph1, graph2 = pair

        label = torch.tensor(1)
        data.append((graph1, graph2, label))

    random.shuffle(data)

    # save data
    n = sum(args.sizes)
    k  = len(args.sizes)
    save_dir = args.save_dir + f"{2*args.n_samples}_pairs_membership_sbm_{n}_{k}_{args.p}_{args.q}_{args.permute_rate}_{args.features}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(data, open(save_dir / 'data.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    return data


def generate_sequence_memberships(args=None):

    """

    size: list of nb of nodes in each community
    length: number of graphs in each block
    """

    sbm = {'sizes': args.sizes, 'p': args.p, 'q': args.q, "features":args.features}

    # choose random swaps
    number_swaps = int(args.permute_rate * sum(args.sizes)) // 2

    clusters = [list(range(args.sizes[0]))] + [list(range(args.sizes[i-1], args.sizes[i])) for i in range(1, len(args.sizes))]
    for i in range(len(clusters)):
        random.shuffle(clusters[i])
    # Sample pairs of nodes to swap
    swaps = []
    idx = 0
    while len(swaps) < number_swaps and idx < 2*sum(args.sizes):
        c1, c2 = np.random.choice(np.arange(len(args.sizes)), size=2, replace=False)
        if (len(clusters[c1]) > 0) and (len(clusters[c2]) > 0):
            n1 , n2 = clusters[c1][-1], clusters[c2][-1]
            clusters[c1].pop()
            clusters[c2].pop()
            swaps.append([n1, n2])
        idx+=1

    cp_time = np.random.randint(args.n_samples // 4, 3 * args.n_samples // 4)
    print(f" Change point at t = {cp_time}")

    g1, g2 = [], []
    for _ in range(cp_time):
        g1.append(sample_swap(sbm))
    for _ in range(args.n_samples - cp_time):
        g2.append(sample_swap(sbm, swaps))
    seq = g1 + g2

    # save data
    n = sum(args.sizes)
    k = len(args.sizes)
    time_exp = str(datetime.utcnow().strftime("%m_%d_%H:%M:%S"))
    save_dir = args.save_dir + f"{time_exp}_swap_T_{args.n_samples}_n_{n}_k_{k}_p_{args.p}_q_{args.q}_{args.permute_rate}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(seq, open(save_dir / 'data.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    # save change point time
    with open(save_dir / 'time.json', 'w') as fp:
        json.dump(cp_time, fp)

    return seq, cp_time

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--sizes', type=int, nargs="+", default=[100, 100, 100, 100])
    parser.add_argument('--p', type=float, default=0.25)
    parser.add_argument('--q', type=float, default=0.2)
    parser.add_argument('--permute_rate', type=float, default=0.2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--features', type=str, choices=['gaussian', 'degree'], default=None)
    parser.add_argument('--sequence', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str,  default='/data/kenlay/DyNNet/data/synthetic/')
    parser.add_argument('--rep', type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if not args.sequence:
        generate_pairs_memberships(args=args)
    else:
        generate_sequence_memberships(args=args)

if __name__ == '__main__':
    main()