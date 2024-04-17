"""/*
 *     Testing for TKG Forecasting
 *
    Subset of test function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/src/main.py  test()
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueq
 *
 *     
"""
import test_utils as test_utils
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import torch

def compute_timefilter_hits(scores_dict, timesteps_test, test_data, num_nodes, num_relations, dataset_name):
    all_ans_list_test, test_data_snaps = test_utils.load_all_answers_for_time_filter(test_data, num_relations, False)
    all_ans_static = test_utils.load_all_answers_for_filter(test_data, num_relations, False)
    scores_t_filter, scores_raw, _ = test(timesteps_test, test_data_snaps, scores_dict, all_ans_list_test, all_ans_static, dataset_name)

    return scores_t_filter, scores_raw

def test(timesteps: list, test_triples: list, final_scores: list, all_ans_list_test: list, all_ans_static, dataset_name, pickle_filename ):
    """
    Subset of test function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/src/main.py  test()
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. 
    Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning. SIGIR 2021.
    Note: there were many changes in the utils mainly related to data structures
    :param timesteps: list of timestep values in the test set
    :param test_triples: list of tensors (no.of triples, 3)
    :param final_scores: list of tensors (no. of triples, num_nodes)
    :param all_ans_list_test: some dictionary structure as per logic in source code
    :param all_ans_static: some dictionary structure as per logic in source code TODO describe
    :return:
    """
    ranks_raw, ranks_t_filter, ranks_s_filter, mrr_raw_list, mrr_t_filter_list, mrr_s_filter_list = [], [], [], [], [], []
    assert len(timesteps) == len(test_triples) == len(final_scores) == len(all_ans_list_test)
    timesteps = list(range(len(timesteps)))  # rename to match the standard of all_and_list_test
    ranks_per_rel = {}
    mrr_per_rel = {}
    combined_tensor = torch.cat(test_triples, dim=0)
    relations = combined_tensor[:,1]
    ind_relations = list(set(relations.numpy()))
    for r in ind_relations:
        ranks_per_rel[r] = []

    
    for time_idx, test_triple, final_score in zip(timesteps, test_triples, final_scores):
        mrr_s_filter_snap, mrr_t_filter_snap, mrr_snap, rank_raw, rank_t_filter, rank_s_filter = test_utils.get_total_rank(
            test_triple, final_score,
            all_ans_list_test[time_idx],
            all_ans_static,
            eval_bz=1000,
            rel_predict=0)
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_t_filter.append(rank_t_filter)
        ranks_s_filter.append(rank_s_filter)
        
        
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_t_filter_list.append(mrr_t_filter_snap)
        mrr_s_filter_list.append(mrr_s_filter_snap)
        for rel, mrr in zip(test_triple[:,1], rank_t_filter):
            ranks_per_rel[int(rel)].append(int(mrr))

    mode = 'test'
    scores_raw = test_utils.stat_ranks(ranks_raw, "Entity Prediction Raw", mode, mrr_raw_list)
    scores_t_filter = test_utils.stat_ranks(ranks_t_filter, "Entity TimeAware Prediction Filter", mode, mrr_t_filter_list)
    scores_s_filter = test_utils.stat_ranks(ranks_s_filter, "Entity Static Prediction Filter", mode, mrr_s_filter_list)

    for rel in ind_relations:
        mrr_per_rel[int(rel)] = torch.mean(1.0 / torch.tensor(ranks_per_rel[rel]).float()).item()
    # plt.figure()
    # rf = torch.cat(ranks_t_filter, dim=0)
    # plt.scatter(range(len(rf)), rf, s=0.1)
    # plt.yscale('log')
    # plt.title('author ranks per test tr.' dataset_name 'frequency baseline:'+str(False)+' MR: '+str(np.mean(rf.numpy())), size=8)
    # plt.savefig(dataset_name+'ranks_'+'author')


    # plt.figure()

    # plt.scatter(list(mrr_per_rel.keys()), list(mrr_per_rel.values()), s=0.3)
    # # plt.yscale('log')
    # plt.title('mrr per relation ' + dataset_name + pickle_filename, size=8)
    # plt.ylabel('MRR')
    # plt.xlabel('Relation ID')
    # plt.savefig(dataset_name+'mrr_per_rel'+pickle_filename[0:-4]+'.png')

    return scores_raw, scores_t_filter, scores_s_filter, mrr_per_rel


