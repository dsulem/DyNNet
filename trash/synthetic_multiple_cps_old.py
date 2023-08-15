import random

import dgl
import os
import sys
import pickle
import scipy.sparse as ss
print(os.getcwd())
sys.path.append('/data/kenlay/DyNNet/')
from src.utils.functions import load_model, load_sequence, prepare_batches, add_features_dataset, normalise_statistics
from src.utils.misc import get_device
from src.detect import compute_sgnn_similarity, detect_change_point
from src.utils.baselines import evaluate_baseline, laplacian_spectrum_similarity, NCPD, CUMSUM, CUMSUM_2, avg_deltacon_similarity, avg_wl_distance, avg_frobenius_distance, avg_procrustes_distance
from src.dysbm.clique import generate_sequence_clique_multiple
from src.train import train
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pytorch_lightning.utilities.parsing import str_to_bool
import json
import copy
from time import time
from datetime import datetime
from pathlib import PosixPath
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil


class DataArgs():
    def __init__(self, n_nodes=400, p=0.05, q=0.02, n_changes=2, size_clique=40, n_samples=200, save_dir='/data/kenlay/DyNNet/data/synthetic/', rep=0):
        self.n_nodes = n_nodes
        self.p = p
        self.q = q
        self.n_changes = n_changes
        self.size_clique = size_clique
        self.n_samples = n_samples
        self.features = None
        self.save_dir = save_dir
        self.rep = rep


class ModelArgs():
    def __init__(self, training_data, features='identity', nepochs=100, hidden=64, top_k=100, lr=0.001, n_pairs=5000, batch_size=64, \
                 nlayers=3, validation_proportion=0.5, patience=10, dropout=0.05, nlayers_mlp=2, weight_decay=0.0001, cuda=1, single=False,
                 window_length=6, tolerance=3, threshold=0.5, rep=0):
        self.training_data = training_data
        self.features = features
        self.nepochs = nepochs
        self.hidden = hidden
        self.top_k = top_k
        self.lr = lr
        self.save_dir = '/data/kenlay/DyNNet/results/synthetic/'
        self.validation_proportion = validation_proportion
        self.validation_data, self.test_data = None, None
        self.test_proportion = 0.
        self.n_pairs = n_pairs
        self.cuda = cuda
        self.input_dim = None
        self.pair_sampling = 'random'
        self.batch_size, self.nlayers, self.nlayers_mlp = batch_size, nlayers, nlayers_mlp
        self.embedding_module, self.distance, self.pooling, self.loss = 'gcn', 'euclidean', 'topk', 'bce'
        self.dropout, self.weight_decay = dropout, weight_decay
        self.patience = patience
        self.profiler = False
        self.single = single
        self.task = 'detection'
        self.window_length = window_length
        self.tolerance = tolerance
        self.threshold = threshold
        self.rep = rep


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_workers', type=int, default=16)

    # Synthetic data parameters
    parser.add_argument('--n_nodes', type=int, default=400)
    parser.add_argument('--n_change_points', type=int, default=10, help='number of change-points in test sequence')
    parser.add_argument('--sizes_clique', type=int, nargs="+", default=80)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--q', type=float, default=0.03)
    parser.add_argument('--n_samples_train', type=int, default=200, help='length of each period with one change-point in train and validation data')
    parser.add_argument('--n_samples_test', type=int, default=50, help='length of each period with one change-point in test sequence')
    parser.add_argument('--data_features', type=str, choices=['gaussian'], default=None)
    parser.add_argument('--sequence', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='/data/kenlay/DyNNet/results/synthetic/')
    parser.add_argument('--rep', type=int, default=1, help='Number of repetitions of the experiment')

    # Synthetic data parameters
    parser.add_argument('--validation_proportion', type=float, default=0.5)
    parser.add_argument('--n_pairs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training.')
    parser.add_argument('--embedding_module', type=str, default='gcn', choices=['identity', 'gcn', 'gin', 'gat'],
                        help='Model to use for the node embedding.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers of the graph encoder.')
    parser.add_argument('--nlayers_mlp', type=int, default=2, help='Number of layers of the MLP following topk.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units in each layer of the graph encoder.')
    parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='How to compute distance after the embedding.')
    parser.add_argument('--pooling', type=str, default='topk', choices=['average', 'topk', 'avgraph', 'max'],
                        help='Pooling layer to use after computing similarity or distance.')
    parser.add_argument('--loss', type=str, default='bce', choices=['hinge', 'bce', 'mse', 'contrastive'],
                        help='Loss function to use on predictions.')
    parser.add_argument('--weight_loss', type=float, default=1.0,
                        help='Weight on negative examples in loss function.')
    parser.add_argument('--margin_loss', type=float, default=1.0,
                        help='Margin parameter in loss function.')
    parser.add_argument('--top_k', type=int, default=100, help='Number of nodes in top-k pooling.')
    parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--features', type=str, default='identity', choices=['degree', 'random_walk', 'laplacian', 'identity'], help='Type of added input features')
    parser.add_argument('--input_dim', type=int, default=None, help='Dimension of input features if needed to be added')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=20, help='Patience parameter for early stopping.')
    parser.add_argument('--cuda', type=int, default=3, choices=[0, 1, 2, 3], help='GPU id')
    parser.add_argument('--profiler', type=str_to_bool, default=False, help='Check CPU and GPU consumption')

    parser.add_argument('--window_lengths', type=int, default=5, nargs="+", help='Length of backward window')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold on the similarity statistic to detect change-points')
    parser.add_argument('--tolerance', type=int, default=3, help='Tolerance level in the adjusted F1 metric')

    parser.add_argument('--n_eigen', type=int, default=4, help='Number of eigenvectors for LAD and SC-NCPD')
    parser.add_argument('--normalize', type=str_to_bool, default=False, help='Use symmetric Laplacian for baselines')

    args = parser.parse_args()

    return args


def task(args=None):

    ari_test, f1_test, f1_train, thresholds, compute_times, results_paths = {}, {}, {}, {}, {}, {}
    for method in ['sgnn', 'ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
        ari_test[method], f1_test[method], f1_train[method], thresholds[method], compute_times[
            method] = {}, {}, {}, {}, {}
    compute_times['train_sgnn'] = {}

    # Directory to save results
    save_dir = (f'multiple_cps_{datetime.utcnow().strftime("%m_%d_%H:%M:%S")}'
                f'_cliques_k_{args.n_change_points}'
                f'_S_{args.S}_L_{args.L}'
                f'_features_{args.features}'
                )

    global worker_task

    def worker_task(i):

        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        print("Starting repetition ", i + 1)

        print("Generating train and test data...")

        result = {}

        train_args = DataArgs(n_nodes=args.n_nodes, p=args.p, q=args.q, n_changes=2, size_clique=S,
                              n_samples=args.n_samples_train, rep=i)
        test_args = DataArgs(n_nodes=args.n_nodes, p=args.p, q=args.q, n_changes=args.n_change_points,
                             size_clique=S, n_samples=args.n_samples_test, rep=i)

        # Generate dynamic network sequence with one change point of type clique (for training and validation)
        _, _, _, data_train_path = generate_sequence_clique_multiple(args=train_args)

        # Generate dynamic network sequence with multiple change points of type clique (for testing)
        _, _, _, data_test_path = generate_sequence_clique_multiple(args=test_args)

        margs = ModelArgs(training_data=data_train_path, features=args.features, nepochs=args.nepochs,
                          hidden=args.hidden, \
                          top_k=args.top_k, lr=args.lr, n_pairs=args.n_pairs, batch_size=args.batch_size,
                          nlayers=args.nlayers, \
                          validation_proportion=0.5, patience=args.patience, dropout=args.dropout,
                          nlayers_mlp=args.nlayers_mlp, \
                          weight_decay=args.weight_decay, cuda=(i + 2) % 4, single=(args.n_change_points == 1),
                          window_length=L,
                          tolerance=args.tolerance, threshold=args.threshold, rep=i)

        print("Starting GNN training...")
        t0 = time()
        # Train a GSL model
        model_path = train(args=margs)
        t1 = time()

        margs.test_data = data_test_path
        margs.model_path = model_path

        print("Testing model and baselines...")

        # Detect change-points in test sequence
        path_to_results, sgnn_results, sgnn_ari, sgnn_results_m2, sgnn_ari_m2 = detect_change_point(args=margs)
        result['ari_sgnn'] = sgnn_ari
        result['ari_sgnn_m2'] = sgnn_ari_m2
        result['f1_sgnn'] = sgnn_results['f1']
        result['f1_sgnn_m2'] = sgnn_results_m2['f1']
        result['time_train_sgnn'] = t1 - t0
        result['time_test_sgnn'] = time() - t1
        result['path'] = path_to_results

        print("Computing time: ", time() - t1)

        # Load train and test data
        train_data, train_labels, train_cps = load_sequence(data_train_path)
        test_data, test_labels, test_cps = load_sequence(data_test_path)

        # Compute results of baselines
        for method in ['ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
            t0 = time()
            ari, f1t, f1tr, thresh = evaluate_baseline(method, train_data, train_labels, test_data, test_labels, \
                                                       window_length=L, tolerance=args.tolerance, n_eigen=args.n_eigen,
                                                       normalize=args.normalize, diff=True)
            result[method] = [ari, f1t, f1tr, thresh, time() - t0]
            print("Method : ", method)
            print("Selected threshold : ", thresh)
            print("F1 test and train scores : ", f1t, f1tr)
            print("Computing time : ", time() - t0)

        print(f"Task {i} has terminated")

        print("Deleting data...")
        shutil.rmtree(data_train_path)
        shutil.rmtree(data_test_path)

        # Saving results
        print("Saving results...")
        save_dir = PosixPath(args.save_dir).expanduser() / save_dir
        if not os.path.isdir(save_dir / f'rep_{d}'):
            os.makedirs(save_dir / f'rep_{d}')
        with open(save_dir / f'rep_{d}/results.p', 'wb') as fp:
            pickle.dump(result, fp)

        return i, result

    save_dir = PosixPath(args.save_dir).expanduser() / save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    executor = ProcessPoolExecutor(max_workers=args.n_workers)
    for result in executor.map(worker_task, range(args.rep)):
        print(result)
        d = result[0]
        ari_test['sgnn'][d], f1_test['sgnn'][d] = result[1]['ari_sgnn'], result[1]['f1_sgnn']
        ari_test['sgnn_m2'][d], f1_test['sgnn_m2'][d] = result[1]['ari_sgnn_m2'], result[1]['f1_sgnn_m2']
        results_paths[d] = result[1]['path']
        compute_times['train_sgnn'][d] = result[1]['time_train_sgnn']
        compute_times['sgnn'][d] = result[1]['time_test_sgnn']
        for method in ['ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
            ari_test[method][d], f1_test[method][d], f1_train[method][d] = result[1][method][0], result[1][method][1], \
                                                                           result[1][method][2]
            thresholds[method][d], compute_times[method][d] = result[1][method][3], result[1][method][4]

        args_dict = vars(args)
        config = {}
        for key in args_dict:
            config[key] = args_dict[key]
        config['window_length'] = L
        config['size_clique'] = S

        # save results for this iteration
        if not os.path.isdir(save_dir / f'rep_{d}'):
            os.makedirs(save_dir / f'rep_{d}')
        with open(save_dir / f'rep_{d}/exp_config.json', 'w') as fp:
            json.dump(config, fp, indent=2)
        # with open(save_dir / f'rep_{d}/test_f1scores.p', 'wb') as fp:
        #     pickle.dump(f1_test[d], fp)
        # with open(save_dir / f'rep_{d}/train_f1scores.p', 'wb') as fp:
        #     pickle.dump(f1_train[d], fp)
        # with open(save_dir / f'rep_{d}/thresholds.p', 'wb') as fp:
        #     pickle.dump(thresholds[d], fp)
        # with open(save_dir / f'rep_{d}/compute_times.p', 'wb') as fp:
        #     pickle.dump(compute_times[d], fp)
        # with open(save_dir / f'rep_{d}/test_ari.p', 'wb') as fp:
        #     pickle.dump(ari_test[d], fp)
    executor.shutdown(wait=True)

    for dic in [ari_test, f1_test, f1_train, thresholds, compute_times]:
        for method in ['sgnn', 'ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
            dic[method] = np.array([dic[method][d] for d in dic[method]])
    compute_times['train_sgnn'] = np.array([compute_times['train_sgnn'][d] for d in compute_times['train_sgnn']])

    print(compute_times)

    args_dict = vars(args)
    config = {}
    for key in args_dict:
        config[key] = args_dict[key]
    config['window_length'] = L
    config['size_clique'] = S

    with open(save_dir / 'exp_config.json', 'w') as fp:
        json.dump(config, fp, indent=2)
    with open(save_dir / 'test_f1scores.p', 'wb') as fp:
        pickle.dump(f1_test, fp)
    with open(save_dir / 'train_f1scores.p', 'wb') as fp:
        pickle.dump(f1_train, fp)
    with open(save_dir / 'thresholds.p', 'wb') as fp:
        pickle.dump(thresholds, fp)
    with open(save_dir / 'compute_times.p', 'wb') as fp:
        pickle.dump(compute_times, fp)
    with open(save_dir / 'test_ari.p', 'wb') as fp:
        pickle.dump(ari_test, fp)


def main():

    args = get_args()

    if isinstance(args.window_lengths, list):
        window_lengths = args.window_lengths
    else:
        window_lengths = [args.window_lengths]
    if isinstance(args.sizes_clique, list):
        sizes_clique = args.sizes_clique
    else:
        sizes_clique = [args.sizes_clique]

    for L in window_lengths:

        for S in sizes_clique:

            ari_test, f1_test, f1_train, thresholds, compute_times, results_paths = {}, {}, {}, {}, {}, {}
            for method in ['sgnn', 'ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
                ari_test[method], f1_test[method], f1_train[method], thresholds[method], compute_times[method] = {}, {}, {}, {}, {}
            compute_times['train_sgnn'] = {}

            # Directory to save results
            save_dir = (f'multiple_cps_{datetime.utcnow().strftime("%m_%d_%H:%M:%S")}'
                        f'_cliques_k_{args.n_change_points}'
                        f'_S_{S}_L_{L}'
                        f'_features_{args.features}'
                        )

            global worker_task

            def worker_task(i):

                random.seed(i)
                np.random.seed(i)
                torch.manual_seed(i)

                print("Starting repetition ", i+1)

                print("Generating train and test data...")

                result = {}
                # result['ari_sgnn'], result['f1_sgnn'], result['time_train_sgnn'], \
                # result['time_test_sgnn'], result['path']  = 0., 0., 0., 0., 0.0
                # for method in ['ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'procrustes', 'wl']:
                #     result[method] = [ 0., 0., 0., 0., 0.0]

                train_args = DataArgs(n_nodes=args.n_nodes, p=args.p, q=args.q, n_changes=2, size_clique=S,
                                      n_samples=args.n_samples_train, rep=i)
                test_args = DataArgs(n_nodes=args.n_nodes, p=args.p, q=args.q, n_changes=args.n_change_points,
                                     size_clique=S, n_samples=args.n_samples_test, rep=i)

                # Generate dynamic network sequence with one change point of type clique (for training and validation)
                _,_,_, data_train_path = generate_sequence_clique_multiple(args=train_args)

                # Generate dynamic network sequence with multiple change points of type clique (for testing)
                _, _, _, data_test_path = generate_sequence_clique_multiple(args=test_args)

                margs = ModelArgs(training_data=data_train_path, features = args.features, nepochs = args.nepochs, hidden = args.hidden, \
                                  top_k = args.top_k, lr = args.lr, n_pairs = args.n_pairs, batch_size = args.batch_size, nlayers = args.nlayers,\
                                  validation_proportion = 0.5, patience = args.patience, dropout = args.dropout, nlayers_mlp = args.nlayers_mlp, \
                                  weight_decay = args.weight_decay, cuda = (i+2)%4, single = (args.n_change_points == 1), window_length=L,
                                  tolerance=args.tolerance, threshold=args.threshold, rep=i)

                print("Starting GNN training...")
                t0 = time()
                # Train a GSL model
                model_path = train(args=margs)
                t1 = time()

                margs.test_data = data_test_path
                margs.model_path = model_path

                print("Testing model and baselines...")

                # Detect change-points in test sequence
                path_to_results, sgnn_results, sgnn_ari, sgnn_results_m2, sgnn_ari_m2 = detect_change_point(args=margs)
                result['ari_sgnn'] = sgnn_ari
                result['ari_sgnn_m2'] = sgnn_ari_m2
                result['f1_sgnn'] = sgnn_results['f1']
                result['f1_sgnn_m2'] = sgnn_results_m2['f1']
                result['time_train_sgnn'] = t1- t0
                result['time_test_sgnn'] = time() - t1
                result['path'] = path_to_results

                print("Computing time: ", time()-t1)

                # Load train and test data
                train_data, train_labels, train_cps = load_sequence(data_train_path)
                test_data, test_labels, test_cps = load_sequence(data_test_path)

                # Compute results of baselines
                for method in ['ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
                    t0 = time()
                    ari, f1t, f1tr, thresh = evaluate_baseline(method, train_data, train_labels, test_data, test_labels, \
                                                               window_length=L, tolerance=args.tolerance, n_eigen=args.n_eigen, normalize=args.normalize, diff=True)
                    result[method] = [ari, f1t, f1tr, thresh, time()-t0]
                    print("Method : ", method)
                    print("Selected threshold : ", thresh)
                    print("F1 test and train scores : ", f1t, f1tr)
                    print("Computing time : ", time() - t0)

                print(f"Task {i} has terminated")

                print("Deleting data...")
                shutil.rmtree(data_train_path)
                shutil.rmtree(data_test_path)

                # Saving results
                print("Saving results...")
                save_dir = PosixPath(args.save_dir).expanduser() / save_dir
                if not os.path.isdir(save_dir / f'rep_{d}'):
                    os.makedirs(save_dir / f'rep_{d}')
                with open(save_dir / f'rep_{d}/results.p', 'wb') as fp:
                    pickle.dump(result, fp)

                return i, result

            save_dir = PosixPath(args.save_dir).expanduser() / save_dir
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            executor = ProcessPoolExecutor(max_workers=args.n_workers)
            for result in executor.map(worker_task, range(args.rep)):
                print(result)
                d = result[0]
                ari_test['sgnn'][d], f1_test['sgnn'][d] = result[1]['ari_sgnn'], result[1]['f1_sgnn']
                ari_test['sgnn_m2'][d], f1_test['sgnn_m2'][d] = result[1]['ari_sgnn_m2'], result[1]['f1_sgnn_m2']
                results_paths[d] = result[1]['path']
                compute_times['train_sgnn'][d] = result[1]['time_train_sgnn']
                compute_times['sgnn'][d] = result[1]['time_test_sgnn']
                for method in ['ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
                    ari_test[method][d], f1_test[method][d], f1_train[method][d] = result[1][method][0], result[1][method][1], result[1][method][2]
                    thresholds[method][d], compute_times[method][d] = result[1][method][3], result[1][method][4]

                args_dict = vars(args)
                config = {}
                for key in args_dict:
                    config[key] = args_dict[key]
                config['window_length'] = L
                config['size_clique'] = S

                # save results for this iteration
                if not os.path.isdir(save_dir / f'rep_{d}'):
                    os.makedirs(save_dir / f'rep_{d}')
                with open(save_dir / f'rep_{d}/exp_config.json', 'w') as fp:
                    json.dump(config, fp, indent=2)
                # with open(save_dir / f'rep_{d}/test_f1scores.p', 'wb') as fp:
                #     pickle.dump(f1_test[d], fp)
                # with open(save_dir / f'rep_{d}/train_f1scores.p', 'wb') as fp:
                #     pickle.dump(f1_train[d], fp)
                # with open(save_dir / f'rep_{d}/thresholds.p', 'wb') as fp:
                #     pickle.dump(thresholds[d], fp)
                # with open(save_dir / f'rep_{d}/compute_times.p', 'wb') as fp:
                #     pickle.dump(compute_times[d], fp)
                # with open(save_dir / f'rep_{d}/test_ari.p', 'wb') as fp:
                #     pickle.dump(ari_test[d], fp)
            executor.shutdown(wait=True)

            for dic in [ari_test, f1_test, f1_train, thresholds, compute_times]:
                for method in ['sgnn', 'ncpd', 'lad', 'cusum', 'cusum_2', 'frobenius', 'wl']:
                    dic[method] = np.array([dic[method][d] for d in dic[method]])
            compute_times['train_sgnn'] = np.array([compute_times['train_sgnn'][d] for d in compute_times['train_sgnn']])

            print(compute_times)

            args_dict = vars(args)
            config = {}
            for key in args_dict:
                config[key] = args_dict[key]
            config['window_length'] = L
            config['size_clique'] = S

            with open(save_dir  / 'exp_config.json', 'w') as fp:
                json.dump(config, fp, indent=2)
            with open(save_dir / 'test_f1scores.p', 'wb') as fp:
                pickle.dump(f1_test, fp)
            with open(save_dir / 'train_f1scores.p', 'wb') as fp:
                pickle.dump(f1_train, fp)
            with open(save_dir / 'thresholds.p', 'wb') as fp:
                pickle.dump(thresholds, fp)
            with open(save_dir / 'compute_times.p', 'wb') as fp:
                pickle.dump(compute_times, fp)
            with open(save_dir / 'test_ari.p', 'wb') as fp:
                pickle.dump(ari_test, fp)


if __name__ == '__main__':
    main()
