import sys
#sys.path.append('/Users/sulem/PycharmProjects/GraphSiamese/')
#sys.path.append('/data/localhost/sulem/GraphSiamese/')
from src.model import GraphSiamese
from src.embedding import GCN, GINConv
import argparse
from torch.utils.data import DataLoader
import torch
import pickle
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from src.utils.misc import collate, correct_prediction, get_device, get_batchs, save_config
from src.utils.sample import sample_pairs, sample_pairs_in_window
import pandas as pd
from src.utils.loss import HingeLoss, ContrastiveLoss
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, roc_auc_score, f1_score
import time
import psutil
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path, PosixPath
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.utilities.parsing import str_to_bool
from src.utils.functions import add_features_dataset
from src.utils.dataset import GraphPairsDataset
import gc

#torch.manual_seed(123)


def train(args=None):

    args_dict = vars(args)

    device = torch.device(f"cuda:{args.cuda}" if (args.cuda is not None and torch.cuda.is_available()) else "cpu")
    print("Working on ", device)
    #device = get_device(args.cuda)

    # Memory monitoring
    process = psutil.Process(os.getpid())

    # load data
    # if isinstance(args.training_data, str):
    #     args.training_data = PosixPath(args.training_data).expanduser()
    with open(args.training_data + '/data.p', 'rb') as f:
        data = pickle.load(f)
    with open(args.training_data + '/labels.p', 'rb') as f:
        graph_labels = pickle.load(f)

    # add initial features if not provided in data
    if args.features is not None:
        data = add_features_dataset(data, args.features, args.input_dim)

    # load validation data
    if args.validation_data is not None:
        # if isinstance(args.validation_data, str):
        #     args.validation_data = PosixPath(args.validation_data).expanduser()
        with open(args.validation_data + '/data.p', 'rb') as f:
            validation_data = pickle.load(f)
        with open(args.validation_data + '/labels.p', 'rb') as f:
            validation_labels = pickle.load(f)

        # add initial features if not provided in data
        if args.features is not None:
            validation_data = add_features_dataset(validation_data, args.features, args.input_dim)

        n_train = len(data)
        n_valid = len(validation_data)

    else: # divide data into train, validation and test
        n_train = int(len(data) * (1. - args.validation_proportion - args.test_proportion))
        n_valid = int(len(data) * args.validation_proportion)
        validation_data = data[n_train:n_train+n_valid]
        validation_labels = graph_labels[n_train:n_train + n_valid]

    if args.test_data is not None:
        with open(args.test_dataset + '/data.p', 'rb') as f:
            test_data = pickle.load(f)
        with open(args.test_dataset + '/labels.p', 'rb') as f:
            test_labels = pickle.load(f)

        # add initial features if not provided in data
        if args.features is not None:
            test_data = add_features_dataset(test_data, args.features, args.input_dim)
    elif args.test_proportion > 0.0:
        test_data = data[n_train+n_valid:]
    else:
        test_data = None

    training_data = data[:n_train]
    training_labels = graph_labels[:n_train]
    print(f"Data loaded: training sequence with T={len(training_data)} and validation sequence with T={len(validation_data)} ")

    # Sample graph pairs from training and validation sequences
    if args.pair_sampling == 'random':
        training_data_pairs = sample_pairs(training_data, training_labels, nsamples=args.n_pairs)
        validation_data_pairs = sample_pairs(validation_data, validation_labels, nsamples=args.n_pairs)

    else:
        training_data_pairs = sample_pairs_in_window(training_data, training_labels, window_length=args.window_length)
        validation_data_pairs = sample_pairs_in_window(validation_data, validation_labels, window_length=args.window_length)

    # Transform into DataSet object
    # training_pairs_labels = [training_data_pairs[i][2].item() for i in range(len(training_data_pairs))]
    # validation_pairs_labels = [validation_data_pairs[i][2].item() for i in range(len(validation_data_pairs))]
    # os.makedirs(args.training_data + '/training_pairs')
    # os.makedirs(args.training_data + '/validation_pairs')
    # with open(args.training_data + '/training_pairs/data.p', 'wb') as f:
    #     pickle.dump(training_data_pairs)
    # with open(args.training_data + '/training_pairs/labels.p', 'wb') as f:
    #     pickle.load(training_pairs_labels)
    # with open(args.training_data + '/validation_pairs/data.p', 'wb') as f:
    #     pickle.dump(validation_data_pairs)
    # with open(args.training_data + '/validation_pairs/labels.p', 'wb') as f:
    #     pickle.load(validation_pairs_labels)
    # training_data_pairs = GraphPairsDataset(args.training_data + '/training_pairs/')
    # validation_data_pairs = GraphPairsDataset(args.training_data + '/validation_pairs/')

    training_data_pairs = DataLoader(training_data_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                               drop_last=True)
    validation_data_pairs = DataLoader(validation_data_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                               drop_last=True)
    if test_data is not None:
        print(f"Test sequence with T={len(test_data)}")
        test_data_pairs = sample_pairs_in_window(test_data, training_labels, window_length=args.window_length)
        test_data_pairs = DataLoader(test_data_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                                   drop_last=True)

    gc.collect()

    # build model
    input_dim = training_data_pairs.dataset[0][0].ndata['node_attr'].size(1)
    #print(input_dim)
    args.input_dim= input_dim

    embedding = GCN(input_dim=input_dim, type=args.embedding_module, hidden_dim=args.hidden, layers=args.nlayers, dropout=args.dropout)

    if args.distance.lower() not in ['euclidean', 'cosine']:
        raise ValueError('Distance not recognised')

    if args.pooling.lower() not in ['average','topk', 'avgraph', 'max']:
        raise ValueError('Pooling module not recognised')

    if args.top_k is None:
        args_dict["top_k"] = training_data_pairs.dataset[0][0].ndata['node_attr'].size(0) // 4

    model = GraphSiamese(embedding, args.distance, args.pooling, args.loss, args.top_k, nlinear=args.nlayers_mlp,
                         nhidden=args.hidden, dropout=args.dropout, features=args.features)
    model = model.to(device)

    # optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # loss function
    if args.loss == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif args.loss == 'bce':
        loss_fn = torch.nn.BCELoss(reduction='none')
    else:
        raise ValueError("Loss function not implemented")

    # results container
    if args.validation_proportion > 0.0:
        logging = {'train_loss': [], 'train_acc': [], 'train_recall': [], 'train_precision': [], 'train_auc': [], 'train_f1': [],
                   'valid_loss': [], 'valid_acc': [], 'valid_precision': [], 'valid_recall': [], 'valid_auc': [], 'valid_f1': []}
    else:
        logging = {'train_loss': [], 'train_acc': [], 'train_recall': [], 'train_precision': [], 'train_auc': [], 'train_f1': []}
    best_f1, best_weights, best_loss = 0., None, np.Inf
    final_metrics = {'loss' : [0.0, 0., 0.], 'accuracy': [0., 0., 0.], 'recall': [0., 0., 0.], 'precision': [0., 0., 0.],
                     'auc': [0., 0., 0.], 'f1': [0., 0., 0.]}

    # for early stopping
    patience = args.patience
    patience_counter = 0

    t0 = time.time()

    print("CPU Memory usage :", process.memory_info().rss)

    # training loop
    for epoch in range(args.nepochs):

        # training updates
        train_loss, train_acc, train_precision, train_recall, train_auc, train_f1 = [], [], [], [], [], []

        model.train()

        # minibatch loop
        for (graph1, graph2, labels) in training_data_pairs:

            graph1, graph2, labels = graph1.to(device), graph2.to(device), labels.to(device)

            if args.profiler:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    predictions = model(graph1, graph2)
                print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            else:
                predictions = model(graph1, graph2)


            if args.loss == 'hinge':
                predictions = torch.tanh(predictions) # predictions between -1 and 1
            if args.loss == 'bce':
                predictions = torch.sigmoid(predictions) # predictions between 0 and 1

            loss = loss_fn(predictions.squeeze(), labels.float())


            # balanced accuracy score instead of plain accuracy
            accuracy = torch.tensor(np.array((predictions.squeeze().cpu().detach() > 0.5) == labels.cpu(),
                                             dtype=float).mean().item()).unsqueeze(dim=0)
            recall = torch.tensor(
                [recall_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])
            precision = torch.tensor(
                [precision_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])

            if (labels != 0).all() or (labels != 1).all():
                print("No positive or negative labels in this minibatch of the training set")
                auc = torch.zeros_like(accuracy)
            else:
                auc = torch.tensor([roc_auc_score(labels.cpu(), predictions.squeeze().cpu().detach())])
            f1 = torch.tensor([f1_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float())])

            train_loss.append(loss), train_acc.append(accuracy), train_recall.append(recall), train_precision.append(precision), \
            train_auc.append(auc), train_f1.append(f1)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


        logging['train_loss'].append(torch.cat(train_loss).mean().item())
        logging['train_acc'].append(torch.cat(train_acc).mean().item())
        logging['train_recall'].append(torch.cat(train_recall).mean().item())
        logging['train_precision'].append(torch.cat(train_precision).mean().item())
        logging['train_auc'].append(torch.cat(train_auc).mean().item())
        logging['train_f1'].append(torch.cat(train_f1).mean().item())

        scheduler.step()

        # test step
        if args.validation_proportion > 0.0 or args.validation_data is not None:

            # validation step
            model.eval()

            valid_loss, valid_acc, valid_recall, valid_precision, valid_auc, valid_f1 = [], [], [], [], [], []
            with torch.no_grad():
                for (graph1, graph2, labels) in validation_data_pairs:

                    graph1, graph2 = graph1.to(device), graph2.to(device)
                    predictions = model(graph1, graph2)

                    if args.loss == 'hinge':
                        predictions = torch.tanh(predictions)
                    if args.loss == 'bce':
                        predictions = torch.sigmoid(predictions)

                    loss = loss_fn(predictions.squeeze().cpu(), labels.float())

                    recall = torch.tensor(
                        [recall_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
                    precision = torch.tensor(
                        [precision_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
                    accuracy = torch.tensor(np.array((predictions.squeeze().detach().cpu() > 0.5).float() == labels.float(),
                                                     dtype=float).mean().item()).unsqueeze(dim=0)

                    if (labels != 0).all() or (labels != 1).all():
                        #print(labels)
                        print("No positive or negative labels in this minibatch of the validation set")
                        auc = torch.zeros_like(accuracy)
                    else:
                        auc = torch.tensor([roc_auc_score(labels.float(), predictions.squeeze().detach().cpu())])
                    f1 = torch.tensor([f1_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float())])

                    valid_loss.append(loss), valid_acc.append(accuracy), valid_recall.append(
                        recall), valid_precision.append(
                        precision), valid_auc.append(auc), valid_f1.append(f1)

                logging['valid_loss'].append(torch.cat(valid_loss).mean().item())
                logging['valid_acc'].append(torch.cat(valid_acc).mean().item())
                logging['valid_recall'].append(torch.cat(valid_recall).mean().item())
                logging['valid_precision'].append(torch.cat(valid_precision).mean().item())
                logging['valid_auc'].append(torch.cat(valid_auc).mean().item())
                logging['valid_f1'].append(torch.cat(valid_f1).mean().item())

            # save best weights
            #if logging['valid_f1'][-1] > best_f1 and epoch > 0:
            if logging['valid_loss'][-1] < best_loss and epoch > 0:
                best_loss = logging['valid_loss'][-1]
                #best_f1 = logging['valid_f1'][-1]
                final_metrics['loss'][:2] = [logging['train_loss'][-1], logging['valid_loss'][-1]]
                final_metrics['accuracy'][:2] = [logging['train_acc'][-1], logging['valid_acc'][-1]]
                final_metrics['recall'][:2] = [logging['train_recall'][-1], logging['valid_recall'][-1]]
                final_metrics['precision'][:2] = [logging['train_precision'][-1], logging['valid_precision'][-1]]
                final_metrics['auc'][:2] = [logging['train_auc'][-1], logging['valid_auc'][-1]]
                final_metrics['f1'][:2] = [logging['train_f1'][-1], logging['valid_f1'][-1]]
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1


        if patience == patience_counter:
            break

        if epoch % 10 == 0:

            train_f1, train_loss = logging['train_f1'][-1], logging['train_loss'][-1]

            if args.validation_proportion > 0.0:
                valid_acc, valid_loss, valid_f1 = logging['valid_acc'][-1], logging['valid_loss'][-1], logging['valid_f1'][-1]
                print("Epoch, Training loss, f1, Valid loss, f1 :", epoch, train_loss, train_f1, valid_loss,
                      valid_f1)
                print("Patience counter : ", patience_counter)

            else:
                print("Epoch, Training loss, accuracy :", epoch, train_loss, train_f1)
            #print("CPU Memory usage :", process.memory_info().rss)

    # test step
    if test_data is not None:
        # evaluate model with test set
        model.load_state_dict(best_weights)
        model.eval()
        test_loss, test_acc, test_recall, test_precision, test_auc, test_f1 = [], [], [], [], [], []
        with torch.no_grad():
            for (graph1, graph2, labels) in test_data_pairs:

                graph1, graph2 = graph1.to(device), graph2.to(device)
                predictions = model(graph1, graph2)  # .cpu()

                if args.loss == 'hinge':
                    predictions = torch.tanh(predictions)
                if args.loss == 'bce':
                    predictions = torch.sigmoid(predictions)

                loss = loss_fn(predictions.squeeze().cpu(), labels.float())
                recall = torch.tensor(
                    [recall_score(labels.float(), (predictions.squeeze().cpu() > 0.5).float(),
                                  zero_division=0.)])
                precision = torch.tensor(
                    [precision_score(labels.float(), (predictions.squeeze().cpu() > 0.5).float(),
                                     zero_division=0.)])

                accuracy = torch.tensor(np.array((predictions.squeeze().cpu() > 0.5).float() == labels.float(),
                                                 dtype=float).mean().item()).unsqueeze(dim=0)

                f1 = torch.tensor([f1_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float())])

                if (labels != 0).all() or (labels != 1).all():
                    print(labels)
                    print("No positive labels in this minibatch of the test set")
                    auc = torch.zeros_like(accuracy)
                else:
                    auc = torch.tensor(
                        [roc_auc_score(labels.float(), predictions.squeeze().cpu().detach() + 1.0 / 2.0)])

                test_loss.append(loss), test_acc.append(accuracy), test_recall.append(
                    recall), test_precision.append(
                    precision), test_auc.append(auc), test_f1.append(f1)

            final_metrics['accuracy'][2] = torch.cat(test_acc).mean().item()
            final_metrics['recall'][2] = torch.cat(test_recall).mean().item()
            final_metrics['precision'][2] = torch.cat(test_precision).mean().item()
            final_metrics['auc'][2] = torch.cat(test_auc).mean().item()
            final_metrics['f1'][2] = torch.cat(test_f1).mean().item()



    args.training_time = time.time() - t0

    # Create subdirectory for results
    model_path = (f'{datetime.utcnow().strftime("%m_%d_%H:%M")}_gsl_model'
                f'_epochs_{args.nepochs}'
                f'_lr_{args.lr}'
                f'_feat_{args.features}'
                )
    save_dir = PosixPath(args.save_dir).expanduser() / model_path
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pd.DataFrame(logging).to_csv(save_dir / 'logging.csv')

    torch.save(best_weights, save_dir /'model.pt')

    with open(save_dir / 'results.json', 'w') as fp:
        json.dump(final_metrics, fp, indent=2)

    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    return str(save_dir)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, default=None)
    parser.add_argument('--validation_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--path_to_hps', type=str, default=None, help='Path to hyperparameters.')
    parser.add_argument('--validation_proportion', type=float, default=0.2)
    parser.add_argument('--test_proportion', type=float, default=0.2)
    parser.add_argument('--n_pairs', type=int, default=5000)
    parser.add_argument('--pair_sampling', type=str, default='random', choices=['random', 'window'])
    parser.add_argument('--window_length', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training.')
    parser.add_argument('--embedding_module', type=str, default='gcn', choices=['identity', 'gcn', 'gin', 'gat'],
                        help='Model to use for the node embedding.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers of the graph encoder.')
    parser.add_argument('--nlayers_mlp', type=int, default=2, help='Number of layers of the MLP following topk.')
    parser.add_argument('--hidden', type=int, default=16,
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
    parser.add_argument('--top_k', type=int, default=None, help='Number of nodes in top-k pooling.')
    parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--save_dir', type=str, default='~/PycharmProjects/GraphSiamese/trained_models/')
    parser.add_argument('--features', type=str, default=None, choices=['degree', 'random_walk', 'laplacian', 'identity'], help='Type of added input features')
    parser.add_argument('--input_dim', type=int, default=None, help='Dimension of input features if needed to be added')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Patience parameter for early stopping.')
    parser.add_argument('--cuda', type=int, default=None, choices=[0, 1, 2, 3], help='GPU id')
    parser.add_argument('--profiler', type=str_to_bool, default=False, help='Check CPU and GPU consumption')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if args.path_to_hps is not None: # Load default hyperparameters

        with open(args.path_to_hps, 'r') as f:
            hparams = json.load(f)

    else:
        hparams = {}

    args_dict = vars(args)  # arguments as dictionary

    for key in hparams:
        if (key != 'dataset') and (key != 'save_dir') and (key != 'cuda') and (key != 'validation_dataset') :
            args_dict[key] = hparams[key]


    model_path = train(args=args)
    folder = PosixPath(args.save_dir).expanduser()
    with open(folder / 'last_model_trained', 'wb') as f:
        pickle.dump(model_path, f)

if __name__ == '__main__':
    main()


