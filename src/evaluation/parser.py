"""
 write results from json file to excel table for better analysis.
 peviously you have to run run_evaluation.py to create output.json with results per method per dataset per config
"""
import os
import json
import pandas as pd
import numpy as np


# noinspection PyShadowingNames
def helper_function(filename: str) -> tuple:
    filename = filename.replace('.pkl', '')
    if "feedvalid" in filename:
        filename = filename.replace('feedvalid', '')
    
    filename = filename.split('_')  # Idea is to use these keywords to decide the location of values in dataframe
    _setting, _filtering, _dataset, _method, _lmbda = 'NA', 'NA', 'NA', 'NA', 'NA'
    for item in filename:
        window = int(filename[4])
        scaling_factor = float(filename[2])
        if window < 0:
            _setting = 'multistep'
        else:
            _setting = 'singlestep'
        # _setting = item if item in setting else _setting
        _filtering = item if item in filtering else _filtering
        _dataset = item if item in datasets else _dataset
        # Invert the method name from small letters to a consistent format
        _method = inv_dict[item] if item in method_names.values() else _method
        if item == 'True':
            return None, None, None, None
    _lmbda = filename[-2]
    _alpha = filename[-1]
    _rulefile = filename[2]
    return _setting, _filtering, _dataset, _method, _alpha, _lmbda, _rulefile, scaling_factor


if __name__ == '__main__':

    ROOT = os.path.join(os.getcwd())
    with open(os.path.join(ROOT, 'output_final.json'), 'r') as stream:
        jsonfile = json.load(stream)
    # normalize_sub_dicts(jsonfile)

    setting = ['singlestep' ] #, 'singlesteponline']
    filtering = ['time', 'raw', 'static']
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    datasets = ['GDELT', 'YAGO', 'WIKI', 'ICEWS14', 'ICEWS18']
    method_names = {        
        'baselinexi': 'baselinexi',
        'baselinepsibaselinexi': 'baselinepsibaselinexi',
        'baselinepsi': 'baselinepsi',
    }
    inv_dict = {value: key for key, value in method_names.items()}  # used in helper function
    # assert len(method_names.keys()) == len(jsonfile.keys()), 'Reports for all methods not present in jsonfile!'

    # Initialise variables relating to the dataframe
    column_names = [f'{dataset}_{metric}' for dataset in datasets for metric in metrics]
    raw_df = pd.DataFrame(columns=column_names)
    static_df = pd.DataFrame(columns=column_names)
    time_df = pd.DataFrame(columns=column_names)

    for method_name in method_names.keys():
        if not 'results/'+ method_name in list(jsonfile.keys()):
            continue
        sub_dict = jsonfile['results/'+ method_name]

        # Iterate on each sub-dict (.pkl report values)
        for pkl_name, report in sub_dict.items():
            print(pkl_name, '\n', '=' * 100)
            if 'baselinepsibaselinexi_WIKI_1_singlestep_0_-1_' in pkl_name:
                print(pkl_name)
            _setting, _, _dataset, _method, alpha, lmbda, rulefile, scaling_factor = helper_function(pkl_name)
            # if int(rulefile) == 2: #with the old tlogic rules
            #     continue
            if 'NA' in _method:
                print(pkl_name)
            if _setting is None:  # special constraint check that avoids pkl files with `True` in their names
                continue

            if _setting == 'multistep':
                continue

            for filter, values in report.items():
                if 'mrr_per_rel' not in filter:
                    index = f'{_method}_{_setting}_{lmbda}_{alpha}_{scaling_factor}'
                    mrr = np.round(values[1] * 100, 2)
                    hits = [np.round(value * 100, 2) for value in values[2]]

                    if filter == 'raw':
                        raw_df.loc[index, f'{_dataset}_mrr'] = mrr
                        raw_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        raw_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        raw_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    elif filter == 'static':
                        static_df.loc[index, f'{_dataset}_mrr'] = mrr
                        static_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        static_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        static_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    elif filter == 'time':
                        time_df.loc[index, f'{_dataset}_mrr'] = mrr
                        time_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                        time_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                        time_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                    else:
                        raise Exception

    # Save the output as a .xlsx document
    writer = pd.ExcelWriter(os.path.join(ROOT, 'output_final_singlestep_2024.xlsx'), engine='xlsxwriter')
    raw_df.to_excel(writer, sheet_name='raw')
    static_df.to_excel(writer, sheet_name='static')
    time_df.to_excel(writer, sheet_name='time')
    writer.close() #save()
