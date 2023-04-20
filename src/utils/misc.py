import dgl
import torch
import numpy as np
from configparser import ConfigParser
from pathlib import PosixPath
import json
from typing import Union


def convert_labels_into_changepoints(labels: Union[np.ndarray, list], tolerance=0):

    if isinstance(labels, list):
        labels = np.array(labels)

    cps = np.concatenate([np.zeros(1), (abs(labels[1:] - labels[:-1]) > 0)],axis=0)

    for i in range(1,tolerance+1):
        cps = (cps + np.concatenate([np.zeros(i), cps[:-i]], axis=0) + np.concatenate([cps[i:], np.zeros(i)], axis=0) > 0)
    return cps


def collate(samples, add_selfloops=True):
    """Used to create DGL dataloaders."""
    graphs1, graphs2, labels = map(list, zip(*samples))
    if add_selfloops:
        graphs1 = [dgl.add_self_loop(graph) for graph in graphs1]
        graphs2 = [dgl.add_self_loop(graph) for graph in graphs2]
    graphs1, graphs2 = dgl.batch(graphs1), dgl.batch(graphs2)
    return graphs1, graphs2, torch.tensor(labels)


def correct_prediction(predictions, labels, margin=0.5):
    """Returns a 0/1 torch tensor where 1 indicates a correct prediction.
    Consider predictions > margin as label 1"""
    predictions = np.array(predictions.detach().cpu().numpy() > margin, dtype=float)
    labels = labels.detach().cpu().numpy()
    correct = torch.tensor(np.array(predictions == labels, dtype=float))
    return correct



def get_device(cuda=True):
    """Return device string."""
    if torch.cuda.is_available() and cuda: #gpu is not None:
        #device = torch.cuda.current_device()
        device = torch.device('cuda')
        #device = torch.device(f'cuda:{0}')
        print('Using', device, 'with cuda', torch.cuda.get_device_capability(device)[0], flush=True)
    else:
        device = torch.device('cpu')
        print('Using cpu', flush=True)
    return device

def get_batchs(graphs1, graphs2, labels):
    return dgl.batch(graphs1), dgl.batch(graphs2), labels

def save_config(embedding, epochs, lr, weight_decay, hidden, dropout, pooling, datapath, prefix_dir, time, batchsize,
                topk, loss, input_dim, nlayers, nlayers_mlp, margin_loss, final_metrics):
    config = ConfigParser()

    config['PARAMETERS'] = {'datapath': datapath,
                            'epochs': epochs,
                            'lr': lr,
                            'weight_decay': weight_decay,
                            'hidden': hidden,
                            'dropout': dropout,
                            'pooling': pooling,
                            'embedding': embedding,
                            'training_time': time,
                            'minibatch': batchsize,
                            'topk': topk,
                            'loss': loss,
                            'input_dim': input_dim,
                            'nlayers':nlayers,
                            'nlayers_mlp': nlayers_mlp,
                            'margin_loss': margin_loss}

    config['RESULTS'] = {'accuracy': final_metrics['accuracy'],
                         'precision': final_metrics['precision'],
                         'recall': final_metrics['recall'],
                         'auc': final_metrics['auc']}

    with open(prefix_dir / "results.json", 'w+') as configfile:
        config.write(configfile)

    return config


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



def save_args(args:dict, path:PosixPath):
    args_dict = args.copy()
    args_dict = {key:(s)}
