# TKG Forecasting: Recurrency_Baselines

"History repeats itself: A Baseline for Temporal Knowledge Graph Forecasting" 
Julia Gastinger, Christian Meilicke, Federico Errica, Timo Sztyler, Anett Schuelke, Heiner Stuckenschmidt

## Getting started

Install all packages from requirements.txt

## What to run

* src/write_baseline_rules.py # to write for each dataset the baseline rules to rules/dataset_name/1_r.json. this is only needed for new datasets.
* optional: parameter_learning.py # to select the best values for alpha and lmbda_psi for each dataset for each relation; is stores in ./configs
* optional: parameter_learning_per_ds.py # to select the best values for alpha and lmbda_psi for each dataset and all relations ./configs
* test.py: to apply the baselines on the test set and compute final mrr and results file; results file is stored in ./results/dataset_name
* src/evaluation/run_evaluation.py

## How to run

* see run.sh for examples how to run and reproduce our experiments.
* comment or uncomment the desired lines

## How to evaluate

* see /src/evaluation for instructions

## How to cite

```
@inproceedings{gastinger2024baselines,
  title={History repeats itself: A Baseline for Temporal Knowledge Graph Forecasting},
  author={Gastinger, Julia and Meilicke, Christian and Errica, Federico and Sztyler, Timo and Schuelke, Anett and Stuckenschmidt, Heiner},
  booktitle={33nd International Joint Conference on Artificial Intelligence (IJCAI 2024)},
  year={2024},
  organization={International Joint Conferences on Artificial Intelligence Organization}
}
```

## 


## Update June 2025: 

## ⚠️ Important Notes on Evaluation Updates

### 🔴 Update 1 (2025-09-01)

**Table `results` has been updated on 09/01/2025** to address an issue with the recurrency baseline evaluation when handling very small prediction scores.

#### Issue
The problem occurred because we added a very small random value to the scores during evaluation to avoid ties. However, this addition caused unintended effects. Specifically, zero scores could become larger than very small true scores due to the non-negligible random addition.

#### Previous Implementation
```python
score = score * np.max([10000, 10000 * score.max()])
random_values = torch.rand_like(score)
score = score + random_values  
```

📁 *Files affected:*  
- `src/evaluation/test_utils.py`  
- `src/utils/utils.py`  

#### ✅ Fix
We **removed the addition of random values** entirely, as it was unnecessary and led to evaluation distortions.

---

### Precision Improvement

To improve precision, we changed the **data type for loading scores** when computing the **MRR** and **Hits@K** metrics from `float` to `float64`.

#### Before
```python
predictions = torch.zeros(num_nodes, device=device)
predictions.scatter_(0, torch.tensor(list(predictions_dict.keys())).long(), 
    torch.tensor(list(predictions_dict.values())).float())
```

#### After
```python
predictions = torch.zeros(num_nodes, device=device, dtype=torch.float64)
predictions.scatter_(0, torch.tensor(list(predictions_dict.keys())).long(), 
    torch.tensor(list(predictions_dict.values()), dtype=torch.float64))
```

📁 *File affected:*  
- `src/utils/logging_utils.py`

---

### 🔵 Update 2 (2025-06-24)

**Table `results` has been updated again on 24/06/2025** to fix a bug in the **recurrency baseline** implementation.

#### Description
When the `baseline_psi` had no predictions for a query, it incorrectly reused the scores from the previous query instead of predicting all zeros.

#### ✅ Fix
We initialized predictions for both `baseline_psi` and `baseline_xi` to zeros for each query.

#### Updated Code
```python
for j in test_queries_idx:       
    predictions_xi = torch.zeros(num_nodes) 
    predictions_psi = torch.zeros(num_nodes)
```

📁 *File affected:*  
- `src/utils/apply_baselines.py`


## New results with Updates 1 and 2: Updated Evaluation Results

>  Note 2 (24/06/2025):  This table has been updated to address both updates mentioned in Notes 1 and 2 above. Entries marked with 🟦 have (slightly) changed since the original paper.  
> 🟦 Blue entries reflect scores updated in both Note 1 and Note 2.

| Dataset                              | GDELT | GDELT   | YAGO   | YAGO   | WIKI   | WIKI   | ICEWS14   | ICEWS14   | ICEWS18   | ICEWS18  |
|-------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Score                                  | MRR   | Hits10   | MRR   | Hits10      | MRR   | Hits10      | MRR   | Hits10      | MRR   | Hits10    |
| *OLD Strict (as in paper)* ($\psi_\Delta$)       | 23.7 | 38.3 | 90.7 | 92.8   | 81.6   | 87.0   | 36.3   | 48.4   | 27.8   | 41.4   |
| *🟦 new06 Strict* ($\psi_\Delta$)               | 🟦 23.5 | 🟦 38.1 | 90.7 | 🟦 92.7 | 🟦 81.5 | 🟦 86.9 | 🟦 36.0 | 🟦 47.9 | 🟦 27.3 | 🟦 40.7 |

---
| Dataset                              | GDELT | GDELT   | YAGO   | YAGO   | WIKI   | WIKI   | ICEWS14   | ICEWS14   | ICEWS18   | ICEWS18  |
|-------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Score                                  | MRR   | Hits10   | MRR   | Hits10      | MRR   | Hits10      | MRR   | Hits10      | MRR   | Hits10    |
| *OLD Combined (as in paper)* ($\psi_\Delta\xi$)              | 24.5 | 39.8 | 90.9 | 93.0 | 81.5   | 87.1 | 37.2   | 51.8   | 28.7   | 43.7   |
| *🟦 new06 Combined* ($\psi_\Delta\xi$)         | 24.5 | 39.8 | 90.9 | 93.0 | 🟦 81.4 | 87.1 | 🟦 37.4 | 🟦 51.5 | 28.7   | 43.6 |


