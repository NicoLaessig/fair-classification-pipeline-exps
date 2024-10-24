"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import copy
import pandas as pd


#For the experiment with two protected attributes, the dataset dictionary has to add the second attribute
#to the sens_attrs list and the corresponding value of the privileged group to the favored tuple.
data_dict = {
    "german": {"sens_attrs": ["sex"], "label": "job", "favored": (0)},
    "compas": {"sens_attrs": ["Race"], "label": "Two_yr_Recidivism", "favored": (1)},
    "communities": {"sens_attrs": ["race"], "label": "crime", "favored": (1)},
    "credit_card_clients": {"sens_attrs": ["sex"], "label": "payment", "favored": (0)},
    "adult_data_set_race": {"sens_attrs": ["race"], "label": "salary", "favored": (0)},
    "adult_data_set_sex": {"sens_attrs": ["sex"], "label": "salary", "favored": (0)},
    "acs2017_census": {"sens_attrs": ["Race"], "label": "Income", "favored": (0)},
    "implicit30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social30": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "adult_data_set": {"sens_attrs": ["sex", "race"], "label": "salary", "favored": (0,0)},
    "compas_mult": {"sens_attrs": ["Race", "Female"], "label": "Two_yr_Recidivism", "favored": (1,0)},
    "ACS_OR_norm": {"sens_attrs": ["SEX"], "label": "PINCP", "favored": (1)},
    "ACS_ID_norm": {"sens_attrs": ["SEX"], "label": "PINCP", "favored": (1)},
    "implicit10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social10": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit20": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social20": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit40": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social40": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "implicit50": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)},
    "social50": {"sens_attrs": ["sensitive"], "label": "label", "favored": (1)}
    }

"""
PARAMETER SETTINGS START
"""
#Choose the algorithms to test from the following list:
pre_model_list = [
    "iFlipper",
    "LFR",
    "FairSSL-xT",
    "FairSSL-Lx",
    "Fair-SMOTE",
    "FairRR",
    "LTDD",
    "FESF"
    ]

in_model_list = [
    "MultiAdversarialDebiasing",
    "AdversarialDebiasing",
    "GerryFairClassifier",
    "MetaFairClassifier",
    "PrejudiceRemover",
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "AdaFair",
    "FairGeneralizedLinearModel",
    "SquaredDifferenceFairLogistic",
    "FairnessConstraintModel",
    "DisparateMistreatmentModel",
    "ConvexFrameworkModel",
    "HSICLinearRegression",
    "GeneralFairERM",
    "FAGTB",
    "FairDummies",
    "HGR",
    "GradualCompatibility",
    "fairret"
    ]

post_model_list = [
    "EqOddsPostprocessing",
    "CalibratedEqOddsPostprocessing",
    "RejectOptionClassification",
    "JiangNachum",
    "FaX",
    "DPAbstention",
    "GetFair",
    "FairBayesDPP",
    "LevEqOpp",
    "GroupConsensus"
    ]

#Choose the randomstate. If set to -1, then a random one is taken. Make sure that all algorithms
#use the same randomstate. Also choose the size of the testdata.
randomstate = -1
testsize = 0.3

#Choose the considered fairness metric: demographic_parity, equalized_odds, treatment_equality, consistency
metric = "demographic_parity"

#Choose the dataset to run the experiments on (see the dictionary above)
ds = "communities"
sensitive = data_dict[ds]["sens_attrs"]
label = data_dict[ds]["label"]
favored = data_dict[ds]["favored"]

#Choose where to store the files
link = "Results/cross_stage/" + str(metric) + "/" + str(ds) + "/"

#Determine if tuning should be turned on or off. It is suggested to first optimize for the pre-processing
#algorithms and then use the saved parameters as input, otherwise it will take the default values for the
#pre-processing algorithm.
tuning = True

try:
    os.makedirs(link)
except FileExistsError:
    pass

"""
PARAMETER SETTINGS END
"""
                        
subprocess.check_call(['python', '-Wignore', 'main_cross.py', '--output', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--testsize', str(testsize), '--randomstate', str(randomstate),
    '--pre_models', str(pre_model_list), '--tuning', str(tuning), '--metric', str(metric),
    '--in_models', str(in_model_list), '--post_models', str(post_model_list)])

model_list = []
for pre in pre_model_list:
    for in_ in in_model_list:
        model_list.append(pre + "_" + in_)
    for post in post_model_list:
        model_list.append(pre + "_" + post)
for post in post_model_list:
    for in_ in in_model_list:
        model_list.append(post + "_" + in_)
    for pre in pre_model_list:
        model_list.append(post + "_" + pre)
model_list_eval = copy.deepcopy(model_list)

if tuning:
    for m in model_list:
        model_list_eval.append(m + "_tuned")


subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--models', str(model_list_eval), '--metric', str(metric)])
