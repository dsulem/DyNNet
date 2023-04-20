import sys
import sklearn.metrics
import argparse
import pickle
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path, PosixPath
from pytorch_lightning.utilities.parsing import str_to_bool
import scipy.sparse as ss
from src.utils.graphs import laplacian_embeddings, laplacian, norm_laplacian
from src.utils.distances import SubspaceDistance, DeltaConDistance, WL_distance, distance_frobenius, distance_procrustes_LE
from src.utils.metrics import find_best_threshold, adjust_predicts_donut, binary_metrics_adj, compute_ari
from src.utils.functions import dist_labels_to_changepoint_labels_adjusted, normalise_statistics
from tqdm import tqdm
import scipy
from sklearn.cluster import KMeans
import networkx as nx
import dgl
import pandas
import matplotlib.pyplot as plt



def laplacian_spectrum_similarity(data, window_length, normalize=True, n_eigen=6):
    """
    Compute Laplacian Anomaly Detection statistic [Huang et al. 2020]

    :param data (list of DGL graphs): dynamic network sequence
    :param window_length (int): length of backward window
    :param normalize (bool): use symmetric Laplacian matrix
    :param n_eigen (int): nb of eigenvalues used to compute the 'signature'
    :return:
    """

    Lap_spec = []
    for i in range(len(data)):
        A = data[i]
        if not isinstance(A, ss.csr_matrix):
            A = ss.csr_matrix(A.adj().to_dense().numpy())
        if normalize:
            L = norm_laplacian(A)
        else:
            L = laplacian(A)

        L = L.asfptype()
        try:
            w = ss.linalg.eigsh(L, return_eigenvectors=False, k=n_eigen, which='SA')
            Lap_spec.append(w / np.linalg.norm(w))
        except:
            return None

    Zsc = []
    for i in range(window_length, len(data)):
        window = np.stack(Lap_spec[i - window_length:i], axis=1)
        try:
            u, s, v = scipy.linalg.svd(window, compute_uv=True)
        except:
            return None
        Zsc.append(1 - abs(np.dot(u[:, 0], Lap_spec[i])))

    #dzsc = np.clip(np.array(Zsc[1:]) - np.array(Zsc[:-1]), a_min=0, a_max=np.Inf)

    return Zsc, np.arange(window_length, len(data))



def avg_deltacon_similarity(data, window_length, diff=False):
    """
    Computes average similarity statistics with DeltaCon similarity [Koutra et al. 2016]
    :param data:
    :param window_length:
    :return:
    """

    if not isinstance(data[0], np.ndarray):
        data = [data[i].adj().to_dense().numpy() for i in range(len(data))]

    avg_sim = []
    for i in range(window_length, len(data)):
        sim_t = []
        for j in range(1, window_length + 1):
            sim_t.append(DeltaConDistance(data[i], data[i - j]))
        avg_sim.append( np.mean(sim_t) )

    if diff:
        d_avg_sim = np.abs(np.array(avg_sim[1:]) - np.array(avg_sim[:-1]))
        return d_avg_sim, np.arange(window_length+1, len(data))

    return np.array(avg_sim), np.arange(window_length, len(data))


def avg_frobenius_distance(data, window_length, diff=False):
    """
    Computes averaged Frobenius distance statistics over a backward window

    :param data:
    :param window_length:
    :return:
    """

    if not isinstance(data[0], np.ndarray):
        data = [data[i].adj().to_dense().numpy() for i in range(len(data))]

    avg_dist = []
    for i in range(window_length, len(data)):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append(distance_frobenius(data[i], data[i - j]))
        avg_dist.append( np.mean(dist_t) )

    if diff:
        d_avg_dist = np.abs(np.array(avg_dist[1:]) - np.array(avg_dist[:-1]))
        return d_avg_dist, np.arange(window_length+1, len(data))

    return np.array(avg_dist), np.arange(window_length, len(data))


def avg_procrustes_distance(data, window_length, n_eigen=2, diff=False, normalize=True):
    """
    Computes averaged Procrustes distance statistics over a backward window

    :param data:
    :param window_length:
    :param n_eigen:
    :return:
    """

    if not isinstance(data[0], np.ndarray):
        data = [data[i].adj().to_dense().numpy() for i in range(len(data))]

    avg_dist = []
    for i in range(window_length, len(data)):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append( distance_procrustes_LE(data[i], data[i - j], k=n_eigen, normalize=normalize))
        avg_dist.append(np.mean(dist_t))

    if diff:
        d_avg_dist = np.abs(np.array(avg_dist[1:]) - np.array(avg_dist[:-1]))
        return d_avg_dist, np.arange(window_length + 1, len(data))

    return np.array(avg_dist), np.arange(window_length, len(data))


def avg_wl_distance(data, window_length, n_iter=2, diff=False):
    """
    Computes averaged WL distance (inverse of WL kernel) statistics over a backward window

    :param data:
    :param window_length:
    :param n_eigen:
    :return:
    """

    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].adj().to_dense().numpy()) for i in range(len(data))]

    avg_dist = []
    for i in range(window_length, len(data)):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append(WL_distance(data[i], data[i - j], n_iter=n_iter))
        avg_dist.append(np.mean(dist_t))

    if diff:
        d_avg_dist = np.abs(np.array(avg_dist[1:]) - np.array(avg_dist[:-1]))
        return d_avg_dist, np.arange(window_length+1, len(data))

    return np.array(avg_dist), np.arange(window_length, len(data))




def NCPD(data, window_length, n_eigen, normalize=False):
    """
    Compute NCPD statistics of the Spectral Clustering method [Cribben et al. 2017]
    :param data (list of DGL graphs): dynamic network sequence:
    :param window_length (int): length of backward window
    :param n_eigen (int): nb of eigenvalues used to compute the node embeddings
    :return:
    """


    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].adj().to_dense().numpy()) for i in range(len(data))]

    gamma = []
    for i in range(window_length, len(data) - window_length):
        avl = sum(data[i - window_length:i]) / window_length
        avr = sum(data[i:i + window_length]) / window_length
        if normalize:
            lapl = norm_laplacian(avl)
            lapr = norm_laplacian(avr)
        else:
            lapl = laplacian( avl )
            lapr = laplacian( avr )

        vl, wl = ss.linalg.eigsh(lapl, n_eigen, which="SA")
        vr, wr = ss.linalg.eigsh(lapr, n_eigen, which="SA")
        xl = KMeans(n_clusters=n_eigen, n_init='auto').fit(wl[:, :n_eigen])
        xr = KMeans(n_clusters=n_eigen, n_init='auto').fit(wr[:, :n_eigen])

        UL = np.stack([xl.cluster_centers_[xl.labels_[j]] for j in range(data[i].shape[0])], axis=0)
        UR = np.stack([xr.cluster_centers_[xr.labels_[j]] for j in range(data[i].shape[0])], axis=0)

        s = scipy.linalg.svd(UL.transpose().dot(UR), compute_uv=False)
        gamma.append(np.sum(s))

    d_gamma = np.maximum(np.abs(np.array(gamma)[1:-1] - np.array(gamma)[:-2]),
                         np.abs(np.array(gamma)[1:-1] - np.array(gamma)[2:]))

    return d_gamma, 1 + np.arange(window_length, len(data) - window_length -2)







def CUMSUM(data, window_length):
    """
    CUMSUM statistics from Optimal network online change point localisation [Yu, Padilla, Wang & Rinaldo 2021] (without USVT)

    :param data:
    :param window_length:
    :return:
    """


    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].adj().to_dense().numpy()) for i in range(len(data))]

    A = [data[2*i].toarray() for i in range(len(data) //2 )]
    B = [data[2*i + 1].toarray() for i in range(len(data) //2 )]

    Y = []
    for i in range(window_length, len(A) - window_length):

        C_A = 1.0 / (np.sqrt(2 * window_length)) * (np.sum(np.stack(A[i-window_length: i], axis=2), axis=2) - np.sum(np.stack(A[i: i+window_length], axis=2), axis=2))
        C_B = 1.0 / (np.sqrt(2 * window_length)) * (np.sum(np.stack(B[i-window_length: i], axis = 2), axis = 2) - np.sum(np.stack(B[i: i+window_length], axis = 2), axis = 2))

        C_B = C_B / np.linalg.norm(C_B)

        Y.append(np.sum(C_A * C_B))

    times = np.arange(2*window_length, len(data) - 2*window_length, 2)


    return Y, times


def CUMSUM_2(data, window_length):
    """
    CUMSUM statistics from Change-Point Detection in Dynamic Networks with Missing Links [Enikeeva & Klopp 2021]

    :param data:
    :param window_length:
    :return:
    """

    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].adj().to_dense().numpy()) for i in range(len(data))]

    data = [data[i].toarray() for i in range(len(data))]
    stat = []

    for i in range(window_length, len(data) - window_length):

        csum = 1.0 / (np.sqrt(2 * window_length)) * (np.sum(np.stack(data[i - window_length: i ], axis=2), axis=2) - np.sum(np.stack(data[i: i+ window_length], axis=2), axis=2))

        stat.append(np.linalg.norm(csum, ord=2))

    return stat, np.arange( window_length, len(data) - window_length)


def evaluate_baseline(model, training_data, training_labels, test_data, test_labels, window_length, metric='adjusted_f1', tolerance=3, normalize=True, n_eigen=4, n_iter=3, diff=False):
    """
    Evaluate SC-NCPD method on a dynamic network sequence, for a given metric and using a detection threshold selected on training sequence

    :param training_data: dynamic network sequence for selecting threshold
    :param training_labels: distribution labels of the training sequence
    :param test_data: dynamic network sequence for evaluation
    :param test_labels:
    :param window_length:
    :param metric:
    :param tolerance:
    :param normalize:
    :param n_eigen:
    :return:
    """

    T_train, T_test = len(training_data), len(test_data)

    # Computes CP statistic on train and test sequence
    if model == 'ncpd':
        stat_train, stat_train_times = NCPD(training_data, window_length=window_length, n_eigen=n_eigen, normalize=normalize)
        stat_test, stat_test_times = NCPD(test_data, window_length=window_length, n_eigen=n_eigen, normalize=normalize)
    elif model == 'cusum':
        stat_train, stat_train_times = CUMSUM(training_data, window_length=window_length)
        stat_test, stat_test_times = CUMSUM(test_data, window_length=window_length)
    elif model == 'cusum_2':
        stat_train, stat_train_times = CUMSUM_2(training_data, window_length=window_length)
        stat_test, stat_test_times = CUMSUM_2(test_data, window_length=window_length)
    elif model == 'deltacon':
        stat_train, stat_train_times = avg_deltacon_similarity(training_data, window_length=window_length, diff=diff)
        stat_test, stat_test_times = avg_deltacon_similarity(test_data, window_length=window_length, diff=diff)
    elif model == 'frobenius':
        stat_train, stat_train_times = avg_frobenius_distance(training_data, window_length=window_length, diff=diff)
        stat_test, stat_test_times = avg_frobenius_distance(test_data, window_length=window_length, diff=diff)
    elif model == 'wl':
        stat_train, stat_train_times = avg_wl_distance(training_data, window_length=window_length, n_iter=n_iter, diff=diff)
        stat_test, stat_test_times = avg_wl_distance(test_data, window_length=window_length, n_iter=n_iter, diff=diff)
    elif model == 'procrustes':
        stat_train, stat_train_times = avg_procrustes_distance(training_data, window_length=window_length, n_eigen=n_eigen, diff=diff, normalize=normalize)
        stat_test, stat_test_times = avg_procrustes_distance(test_data, window_length=window_length, n_eigen=n_eigen, diff=diff, normalize=normalize)
    elif model == 'lad':
        stat_train, stat_train_times = laplacian_spectrum_similarity(training_data, window_length=window_length, normalize=normalize, n_eigen=n_eigen)
        stat_test, stat_test_times = laplacian_spectrum_similarity(test_data, window_length=window_length,
                                                                     normalize=normalize, n_eigen=n_eigen)
    else:
        raise ValueError('Method not yet implemented')


    # Normalise the statistics in [0,1]
    stat_train_norm = normalise_statistics(stat_train)
    stat_test_norm = normalise_statistics(stat_test)

    if model == 'deltacon': # compute 1 - stat for DeltaCon similarity
        stat_test_norm, stat_train_norm = 1. - stat_test_norm, 1. - stat_train_norm

    # Convert and adjust the distribution labels of the snaphots with the given tolerance level
    cp_lab_train = dist_labels_to_changepoint_labels_adjusted(training_labels, tolerance=tolerance)[stat_train_times]
    cp_lab_test = dist_labels_to_changepoint_labels_adjusted(test_labels, tolerance=tolerance)[stat_test_times]

    # Find best threshold for the given metric on train sequence
    thresh, train_score = find_best_threshold(score=stat_train_norm, target=cp_lab_train, metric=metric)

    # Evaluate on test sequence
    test_score = binary_metrics_adj(score=stat_test_norm, target=cp_lab_test, threshold=thresh,
                                  adjust_predicts_fun=adjust_predicts_donut,
                                  only_f1=True) # adjusted f1 score
    # ARI
    det_cps = stat_test_times[np.where(stat_test_norm > thresh)[0]]
    cp_lab_test = dist_labels_to_changepoint_labels_adjusted(test_labels, tolerance=1)[stat_test_times]
    true_cps = stat_test_times[np.where(cp_lab_test == 1)[0]]
    test_ari = compute_ari(det_cps, true_cps, T_test)


    return test_ari, test_score, train_score, thresh




