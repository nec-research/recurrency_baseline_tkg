
import json
import numpy as np
import os
from data import data_handler


names = ['ICEWS14', 'ICEWS18', 'YAGO', 'WIKI', 'GDELT'] #add new datasets here.
for dataset_name in names:
    dataset = (dataset_name, 3) # identifier, timestamp_column_idx

    train_data, valid_data, _, stat = data_handler.load(dataset[0]) #only needed to extract relation ids
    num_rels = int(stat[1])
    train_data = data_handler.add_inverse_quadruples(train_data, num_rels)
    valid_data = data_handler.add_inverse_quadruples(valid_data, num_rels)

    train_valid_data = np.concatenate((train_data, valid_data))
    rels = list(set(train_valid_data[:,1]))
    new_rules = {}

    for rel in rels:
        rules_id_new = []
        rule_dict = {}
        rule_dict["head_rel"] = int(rel)
        rule_dict["body_rels"] = [int(rel)] #same body and head relation -> what happened before happens again
        rule_dict["conf"] = 1 #same confidence for every rule
        rule_new = rule_dict
        rules_id_new.append(rule_new)
        new_rules[str(rel)] = rules_id_new

    if not os.path.exists('./rules/'+dataset_name):
        os.makedirs('./rules/'+dataset_name)
    with open('./rules/'+dataset_name+'/1_r.json', 'w') as fp: #store rules in directory
        json.dump(new_rules, fp)

