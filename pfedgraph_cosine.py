import copy
import math
import random
import time
from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor
from model import simplecnn, textcnn, simplenn
from prepare_data import get_dataloader
from attack import *
import json
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    pers_acc = []
    gen_acc = []
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        # Calucating accuracies of existing models and updating best models list
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id]
        
        # Training the clients
        net
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x, target
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            # Primary loss
            loss = criterion(out, target)
        
            # This kind of regularization can be used to promote cooperation or alignment between the models of different clients within the same cluster in a federated learning context
            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                # This regularization term encourages similarity between the model's parameters and a cluster model's parameters.
                loss2.backward()
            
            # Backpropagate primary loss
            loss.backward()

            # Update grads using both losses
            optimizer.step()
        
        # Compute acccuracies of trainined model
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)
            pers_acc.append(personalized_test_acc)
            gen_acc.append(generalized_test_acc)
            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean(), pers_acc, gen_acc


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)


# Number of clients in each round
n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]

# List of indices of active clients in each round
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

print("Active clients in each round:")
print(party_list_rounds)

# Trustworthy clients
benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

# Distribute dataset among clients
train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn
elif args.dataset == 'iris':
    model = simplenn
    
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []

# Sending initial model to each client
for i in range(cfg['client_num']):
    local_models.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

for net in local_models:
    net.load_state_dict(global_parameters)

# COLLABORATION GRAPH initialisation
graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

# Stores the cluster model corresponding to each client 
cluster_model_vectors = {}

# Lists
personalised_test_acc_list = [[] for i in range(args.n_parties)]
generalised_test_acc_list = [[] for i in range(args.n_parties)]
final_accuracy = []

for round in range(cfg["comm_round"]):
    # List of active clients in each round
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    # Initialising models and params of each local client
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

    # Perform local training at the clients with num_itr iterations
    # Calculates the loss in each itr using the corresponding cluster model
    # Returns the mean test accuracy of benign clients.
    mean_personalized_acc, pers_acc, gen_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)
    
    final_accuracy.append(mean_personalized_acc)
    for i,acc in enumerate(pers_acc):
        personalised_test_acc_list[i].append(acc)
    for i,acc in enumerate(gen_acc):
        generalised_test_acc_list[i].append(acc)
   
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

    # Update the collaboration graph
    graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs, args.alpha, args.difference_measure)   # Graph Matrix is not normalized yet

    # Update the clustered model
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)                                                    # Aggregation weight is normalized here

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))

    t_np = graph_matrix.numpy() #convert to Numpy array
    if round>0:
        df_old = pd.read_csv(F"/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/graph.csv")
        df = pd.concat([df_old, pd.DataFrame(t_np)]) #convert to a dataframe
        df.to_csv(F"/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/graph.csv",index=False) #save to file
    else:
        df = pd.DataFrame(t_np) #convert to a dataframe
        df.to_csv(F"/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/graph.csv",index=False) #save to file

    # open a file for writing
    with open(F"/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/clusters.json", 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump({k: cluster_model_vectors[k].numpy().tolist() for k in range(args.n_parties)}, f)

    print('-'*80)

# Plot accuracies
import matplotlib.pyplot as plt
import numpy as np
rounds_arr = np.arange(0, cfg["comm_round"])

x = np.array(rounds_arr)
ypoints = np.array(final_accuracy)
plt.plot(x, ypoints)
plt.xlabel("Rounds")
plt.ylabel("Mean Best Personalised Test Accuracy")
plt.legend()
plt.show()
plt.savefig(F'/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/mean.png')

for i,clients_acc in enumerate(personalised_test_acc_list):
  ypoints = np.array(clients_acc)
  plt.plot(x, ypoints, label = 'client'+str(i))
plt.xlabel("Rounds")
plt.ylabel("Personalised Test Accuracy of each client")
plt.legend()
plt.show()
plt.savefig(F'/Users/etiksha/Documents/btp/pFedGraph/results/pfedg_cos/{args.dataset}/clients.png')
 

# Results
# Namespace(gpu='7', model='simplecnn', dataset='iris', partition='noniid-skew', num_local_iterations=200, batch_size=64, lr=0.01, epochs=20, n_parties=10, comm_round=20, init_seed=0, dropout_p=0.0, datadir='./data/', beta=0.5, skew_class=2, reg=1e-05, log_file_name='logs', optimizer='sgd', sample_fraction=1.0, concen_loss='uniform_norm', weight_norm='relu', difference_measure='all', alpha=0.8, lam=0.01, attack_type='inv_grad', attack_ratio=0.0)


#iris-
# python3 Documents/btp/pFedGraph/pfedgraph_cosine.py --gpu "7" --dataset 'iris' --model 'simplenn' --partition 'iid' --n_parties 5 --num_local_iterations 200 --lr 0.01 --epochs 20 --comm_round 20 --reg 1e-5 --log_file_name 'logs' --optimizer 'sgd'