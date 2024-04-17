from collections import Counter
from utils.utils import score_delta
import numpy as np

def update_distributions(learn_data_ts, ts_edges, obj_dist, 
                         rel_obj_dist, num_rels, cur_ts):
    """ update the distributions with more recent infos, if there is a more recent timestep available, depending on window parameter
    take into account scaling factor
    """
    obj_dist_cur_ts, rel_obj_dist_cur_ts= calculate_obj_distribution(learn_data_ts, ts_edges, num_rels) #, lmbda, cur_ts)
    return  rel_obj_dist_cur_ts, obj_dist_cur_ts

def calculate_obj_distribution(learn_data, edges, num_rels):
    """
    Calculate the overall object distribution and the object distribution for each relation in the data.

    Parameters:
        learn_data (np.ndarray): data on which the rules should be learned
        edges (dict): edges from the data on which the rules should be learned

    Returns:
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation
    """
    obj_dist_scaled = {}

    rel_obj_dist = dict()
    rel_obj_dist_scaled = dict()
    for rel in range(num_rels):
        rel_obj_dist[rel] = {}
        rel_obj_dist_scaled[rel] = {}
    
    for rel in edges:
        objects = edges[rel][:, 2]
        dist = Counter(objects)
        for obj in dist:
            dist[obj] /= len(objects)
        rel_obj_dist_scaled[rel] = {k: v for k, v in dist.items()}

    return obj_dist_scaled, rel_obj_dist_scaled



def calculate_obj_distribution_timeaware(learn_data, edges, num_rels, lmbda, cur_ts):
    """
    Calculate the overall object distribution and the object distribution for each relation in the data.
    take into account the delta function. 
    BUT: this is currently to slow without performance benefits

    Parameters:
        learn_data (np.ndarray): data on which the rules should be learned
        edges (dict): edges from the data on which the rules should be learned

    Returns:
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation
    """


    objects = learn_data[:, 2]
    timesteps_o = learn_data[:,3]
    ts_series = np.ones(len(timesteps_o ))*cur_ts
    denom =  np.sum(score_delta(timesteps_o, ts_series, lmbda))
    obj_dist_scaled = {}

    rel_obj_dist = dict()
    rel_obj_dist_scaled = dict()
    for rel in range(num_rels):
        rel_obj_dist[rel] = {}
        rel_obj_dist_scaled[rel] = {}
    
    for rel in edges:
        objects = edges[rel][:, 2]
        rel_edges = edges[rel]
        rel_edges_ts = rel_edges[:,3]
        ts_series = np.ones(len(rel_edges_ts ))*cur_ts 
        denom =  np.sum(score_delta(rel_edges_ts, ts_series, lmbda))
        dist = Counter(objects)
        for obj in objects: #dist:
            ts_rel_ob = rel_edges[rel_edges[:,2]==obj][:,3]
            ts_series_rel_ob = np.ones(len(ts_rel_ob ))*cur_ts
            nom = np.sum(score_delta(ts_rel_ob, ts_series_rel_ob, lmbda))
            rel_obj_dist_scaled[rel][obj] = nom/denom  
    return obj_dist_scaled, rel_obj_dist_scaled
