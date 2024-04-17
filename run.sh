python3 ./src/write_baseline_rules.py # to write the direct recurrency for every triple for all datasets

# 1).1 LEARNING learned per baseline, all baselines
python3 ./src/parameter_selection.py -d ICEWS14 -r 1_r.json  -p 15
# python3 ./src/parameter_selection.py -d YAGO -r 1_r.json -p 15
# python3 ./src/parameter_selection.py -d ICEWS18 -r 1_r.json  -p 15
# python3 ./src/parameter_selection.py -d WIKI -r 1_r.json  -p 15
# python3 ./src/parameter_selection.py -d GDELT -r 1_r.json  -p 15

# A).1 Apply: learned per relation, all baselines
python3 ./src/test.py -d ICEWS14 -psi y -b y  -p 15
# python3 ./src/test.py -d YAGO -psi y -b y  -p 15
# python3 ./src/test.py -d WIKI -psi y -b y   -p 15
# python3 ./src/test.py -d ICEWS18 -psi y -b y  -p 15
# python3 ./src/test.py -d GDELT -psi y -b y  -p 15

# A).2 learned per relation, baseline psi only
# python3 ./src/test.py -d ICEWS14 -psi y -b n  -p 15
# python3 ./src/test.py -d YAGO -psi y -b n  -p 15
# python3 ./src/test.py -d WIKI -psi y -b n   -p 15
# python3 ./src/test.py -d ICEWS18 -psi y -b n  -p 15
# python3 ./src/test.py -d GDELT -psi y -b n  -p 15

# A).3 learned per relation, all baselines, fixed alpha
# python3 ./src/test.py -d ICEWS14 -alpha 0.99999 -psi y -b y  -p 15
# python3 ./src/test.py -d YAGO -alpha 0.99999 -psi y -b y  -p 15
# python3 ./src/test.py -d WIKI -alpha 0.99999 -psi y -b y   -p 15
# python3 ./src/test.py -d ICEWS18 -alpha 0.99999 -psi y -b y  -p 15
# python3 ./src/test.py -d GDELT -alpha 0.99999 -psi y -b y  -p 15

# evaluation
echo starting evaluation
python3 ./src/evaluation/run_evaluation.py
python3 ./src/evaluation/parser.py
echo finish


# # B).1 LEARNING learned per dataset, all baselines
# python3 ./src/parameter_selection_dataset.py -d ICEWS14 -psi y -b y  -p 15
# python3 ./src/parameter_selection_dataset.py -d YAGO -psi y -b y  -p 15
# python3 ./src/parameter_selection_dataset.py -d ICEWS18 -psi y -b y  -p 15
# python3 ./src/parameter_selection_dataset.py -d WIKI -psi y -b y  -p 15
# python3 ./src/parameter_selection_dataset.py -d GDELT -psi y -b y  -p 15

# # # # B).1 TESTING learned per dataset, all baselines
# python3 ./src/test.py -d ICEWS14 -psi y -b y -l -2 -alpha -2     -p 15
# python3 ./src/test.py -d YAGO -psi y -b y -l -2 -alpha -2     -p 15
# python3 ./src/test.py -d WIKI -psi y -b y  -l -2 -alpha -2    -p 15
# python3 ./src/test.py -d ICEWS18 -psi y -b y  -l -2 -alpha -2    -p 15
# # python3 ./src/test.py -d GDELT -psi y -b y  -l -2 -alpha -2     -p 15

## multistep
# # C).1 LEARNING learned per relation, all baselines, multistep
# python3 ./src/parameter_selection.py -d ICEWS14 -r 1_r.json -w -2 -p 15
# python3 ./src/parameter_selection.py -d YAGO -r 1_r.json -w -2 -p 15
# python3 ./src/parameter_selection.py -d ICEWS18 -r 1_r.json -w -2  -p 15
# python3 ./src/parameter_selection.py -d WIKI -r 1_r.json -w -2 -p 15
# python3 ./src/parameter_selection.py -d GDELT -r 1_r.json -w -2 -p 15

# # C).1 TESTING learned per relation, all baselines, multistep
# python3 ./src/test.py -d ICEWS14 -psi y -b y -w -2 -p 15
# python3 ./src/test.py -d YAGO -psi y -b y -w -2 -p 15
# python3 ./src/test.py -d WIKI -psi y -b y  -w -2 -p 15
# python3 ./src/test.py -d ICEWS18 -psi y -b y -w -2 -p 15
# python3 ./src/test.py -d GDELT -psi y -b y -w -2 -p 15
