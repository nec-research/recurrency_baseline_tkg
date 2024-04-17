import numpy as np
from utils.utils import score_delta
import pandas as pd

def get_candidates_psi(rule, rule_walks, test_query_ts, cands_dict, score_func, lmbda, sum_delta_t):
    """
    Get answer candidates from the walks that follow the rule.
    Add the confidence of the rule that leads to these candidates.
    originally from TLogic https://github.com/liu-yushan/TLogic/blob/main/mycode/apply.py but heavily modified

    Parameters:
        rule (dict): rule from rules_dict (not used right now)
        rule_walks (np.array): rule walks np array with [[sub, obj]]
        test_query_ts (int): test query timestamp
        cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
        score_func (function): function for calculating the candidate score
        lmbda (float): parameter to describe decay of the scoring function
        sum_delta_t: to be used in denominator of scoring fct
    Returns:
        cands_dict (dict): keys: candidates, values: score for the candidates  """

    cands = set(rule_walks[:,0]) 

    for cand in cands:
        cands_walks = rule_walks[rule_walks[:,0] == cand] 
        score = score_func(cands_walks, test_query_ts, lmbda, sum_delta_t).astype(np.float64)
        cands_dict[cand] = score

    return cands_dict

def update_delta_t(min_ts, max_ts, cur_ts, lmbda):
    """ compute denominator for scoring function psi_delta
    Patameters:
        min_ts (int): minimum available timestep
        max_ts (int): maximum available timestep
        cur_ts (int): current timestep
        lmbda (float): time decay parameter
    Returns:
        delta_all (float): sum(delta_t for all available timesteps between min_ts and max_ts)
    """
    timesteps = np.arange(min_ts, max_ts)
    now = np.ones(len(timesteps))*cur_ts
    delta_all = score_delta(timesteps, now, lmbda)
    delta_all = np.sum(delta_all)
    return delta_all


def score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t):
    """
    Calculate candidate score depending on the time difference.

    Parameters:
        cands_walks (np.array): rule walks np array with [[sub, obj]]
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution

    Returns:
        score (float): candidate score
    """

    all_cands_ts = cands_walks[:,1] #cands_walks["timestamp_0"].reset_index()["timestamp_0"]
    ts_series = np.ones(len(all_cands_ts))*test_query_ts 
    scores =  score_delta(all_cands_ts, ts_series, lmbda) # Score depending on time difference
    score = np.sum(scores)/sum_delta_t

    return score