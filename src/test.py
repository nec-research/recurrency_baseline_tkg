## imports
import json
import time
import argparse
import numpy as np
from joblib import Parallel, delayed
import pathlib
import pickle
import os
from copy import copy
from matplotlib import pyplot as plt
from collections import Counter
import ray

import utils.utils as utils
from data import data_handler
from utils.apply_baselines import apply_baselines, apply_baselines_remote
from utils.baselinepsi import score_psi

start_o = time.time()

## args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="YAGO", type=str) #ICEWS14, ICEWS18, GDELT, YAGO, WIKI
parser.add_argument("--rules", "-r", default="1_r.json", type=str) #name is usually the same
parser.add_argument("--window", "-w", default=0, type=int) # set to e.g. 200 if only the most recent 200 timesteps should be considered. set to -2 if multistep
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--includebaselinexi", "-b",  default='y', type=str) #'y' if yes, 'n' if no
parser.add_argument("--includebaselinepsi", "-psi",  default='y', type=str) #'y' if yes, 'n' if no
parser.add_argument("--lmbda", "-l",  default=-1, type=float) # if learned_per_rel: -1; if learned_per_dataset: -2 # either for a fixed lmbda or learned lmbdas
parser.add_argument("--alpha", "-alpha",  default=-1, type=float) # if learned_per_rel: -1; if learned_per_dataset: -2  # either for a fixed lmbda or learned lmbdas 0.99999
parser.add_argument("--scalingfactorob", "-sob",  default=0.0, type=float) # if learned_per_rel: -1; if learned_per_dataset: -2  # either for a fixed lmbda or learned lmbdas 0.99999

parsed = vars(parser.parse_args())
dataset_name = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
num_processes = parsed["num_processes"]
ray.init(num_cpus=num_processes, num_gpus=0)
lmbda_in = parsed["lmbda"]
alpha_in = parsed["alpha"]

method_name = ''
baselinepsi_flag = False
baselinexi_flag = False
if parsed["includebaselinepsi"] == 'y':
    method_name += 'baselinepsi'
    baselinepsi_flag = True
if parsed["includebaselinexi"] == 'y':
    method_name += 'baselinexi'
    baselinexi_flag = True
print("method: ", method_name)

## parameters that are typically the same

score_func_psi = score_psi

## load dataset 
dataset = (dataset_name, 3) # identifier, timestamp_column_idx
train_data, valid_data, test_data, stat = data_handler.load(dataset[0])
num_nodes, num_rels = int(stat[0]), int(stat[1])
train_data = data_handler.add_inverse_quadruples(train_data, num_rels)
valid_data = data_handler.add_inverse_quadruples(valid_data, num_rels)
test_data = data_handler.add_inverse_quadruples(test_data, num_rels)
train_valid_data = np.concatenate((train_data, valid_data))
all_data = np.concatenate((train_data, valid_data, test_data))
rels = np.arange(0,2*num_rels)
test_data_prel = data_handler.group_by(test_data, 1, rels)
all_data_prel = data_handler.group_by(all_data, 1, rels)

## load rules
dir_path =  os.path.join(pathlib.Path().resolve(), 'rules', dataset_name, rules_file)  
basis_dict = json.load(open(dir_path)) # + rules_file))
basis_dict = {int(k): v for k, v in basis_dict.items()}

## load the learned params
if lmbda_in == float(-2) or alpha_in == float(-2):
    dir_path =  os.path.join(pathlib.Path().resolve(), 'configs', dataset_name+'configs_per_dataset.json')  
    if window < 0: 
        filename = dataset_name+ str(0.0)+'configs_per_dataset_multistep.json'
        
    else:
        filename = dataset_name+ str(0.0)+'configs_per_dataset.json'
    dir_path =  os.path.join(pathlib.Path().resolve(), 'configs', filename)  
    best_config = json.load(open(dir_path)) 
if lmbda_in == float(-1) or alpha_in == float(-1):
    if window < 0: 
        filename = dataset_name+ str(0.0)+'configs_multistep.json'        
    else:
        filename = dataset_name+ str(0.0)+'configs.json'
    dir_path =  os.path.join(pathlib.Path().resolve(), 'configs', filename)   
    best_config = json.load(open(dir_path)) 

## init empty stuff
final_logging_dict = {}
scores_dict_for_test = {}
lmbdas_used = []
alphas_used = []


## loop through relations and apply baselines
for rel in rels:
    start = time.time()
    rel_key = int(copy(rel))
    if rel == 1:
        print(rel)
    if rel in test_data_prel.keys():
        if lmbda_in > float(-1):
            lmbda_psi = lmbda_in
        elif lmbda_in == float(-1): # one lmbda learned per relation
            lmbda_psi = best_config[str(rel)]['lmbda_psi'][0]
        elif lmbda_in == float(-2): # one lmbda learned per dataset
            lmbda_psi = best_config['lmbda_psi'][0]
        lmbdas_used.append(lmbda_psi)        

        if alpha_in > -0.001:
            alpha = alpha_in
        elif alpha_in == float(-1): # one alpha learned per relation
            alpha = best_config[str(rel)]['alpha'][0]
        elif alpha_in == float(-2): # one alph learned per dataset
            alpha = best_config['alpha'][0]
        if alpha == 1: #to make sure to always include the xi-values as fill-up instead of zeros, but with significantly lower weight than psi
            alpha = 0.99999
        alphas_used.append(alpha)

        # test data for this relation
        test_data_c_rel = test_data_prel[rel]
        timesteps_test = list(set(test_data_c_rel[:,3]))
        timesteps_test.sort()
        all_data_c_rel = all_data_prel[rel]
        
        # queries per process if multiple processes
        num_queries = len(test_data_c_rel) // num_processes
        if num_queries < num_processes: # if we do not have enough queries for all the processes
            num_processes_tmp = 1
            num_queries = len(test_data_c_rel)
        else:
            num_processes_tmp = num_processes      
        
        ## apply baselines for this relation
        object_references = [
            apply_baselines_remote.remote(i, num_queries, test_data_c_rel, all_data_c_rel, window, 
                                basis_dict, score_func_psi, 
                                num_nodes, 2*num_rels, 
                                baselinexi_flag, baselinepsi_flag,
                                lmbda_psi, alpha) for i in range(num_processes_tmp)]
        output = ray.get(object_references)

        ## updates the scores and logging dict for each process
        for proc_loop in range(num_processes_tmp):
            scores_dict_for_test.update(output[proc_loop][1])
            final_logging_dict.update(output[proc_loop][0])
    else:
        lmbdas_used.append(-1)
        alphas_used.append(-1)

    end = time.time()
    total_time = round(end - start, 6)  
    print("Relation {} finished in {} seconds.".format(rel, total_time))

# print some infos:
value_counts = Counter(lmbdas_used)
for value, count in value_counts.items():
    print(f"{value}: {count} times")
end_o = time.time()
total_time_o = round(end_o- start_o, 6)  
print("Running testing with best configs finished in {} seconds.".format(total_time_o))

# saving scores to pkl
print("Now saving the candidates and scores to files. ")
print('save pkl')
logname = method_name + '_' + dataset[0] + '_' + str(0.0) + '_' +'singlestep' + '_' + str(window) + '_' + str(lmbda_in) + '_' +str(alpha_in)
dirname = os.path.join(pathlib.Path().resolve(), 'results', method_name )
eval_paper_authorsfilename = os.path.join(dirname, logname + ".pkl")
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(eval_paper_authorsfilename,'wb') as file:
    pickle.dump(final_logging_dict, file, protocol=4) 
file.close()
print("Done")

# compute mrr
print("Now computing the test MRR")
timesteps_test = list(set(test_data[:,3]))
timesteps_test.sort()
mrr_and_friends = utils.compute_mrr(scores_dict_for_test, test_data, timesteps_test)
mrr = mrr_and_friends[1]
with open('mrr_check.txt', 'a') as f:
    f.write(logname+':\t')
    f.write(str(mrr))
    f.write('\n')

# plot figures on the alpha/lmbda occurence
oc_dict = {}
all_rels = test_data[:,1]
for rel in rels:
    count_occurrences = np.sum(all_rels[:] == rel)
    oc_dict[rel] = count_occurrences

if not os.path.exists('./results/figs/'):
    os.makedirs('./results/figs/')
co = list(oc_dict.values())
if lmbda_in == float(-1):
    plt.figure()
    plt.grid()
    sca = plt.scatter(rels, lmbdas_used, c=co, s=4)
    plt.title('lambda used per relation id')
    plt.colorbar(sca)
    
    
    plt.savefig('./results/figs/lmbdas_used'+dataset_name+'.pdf')
if alpha_in == float(-1):
    plt.figure()
    plt.grid()
    sca =  plt.scatter(rels, alphas_used, c=co, s=4)
    plt.title('alpha used per relation id')
    plt.colorbar(sca)
    
    plt.savefig('./results/figs/alphas_used'+dataset_name+'.pdf')
print("logname", logname)

end_o = time.time()
total_time_o = round(end_o- start_o, 6)  
print("testing finished in {} seconds.".format(total_time_o))
with open('testing_time.txt', 'a') as f:
    f.write(logname+':\t')
    f.write(str(total_time_o))
    f.write('\n')


ray.shutdown()