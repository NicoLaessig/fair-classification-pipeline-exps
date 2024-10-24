"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import copy
import pandas as pd

#For the experiment with two protected attributes, the dataset dictionary has to add the second attribute
#to the sens_attrs list and the corresponding value of the privileged group to the favored tuple.

#For the adult data set, race can be used, as well as a combination of ["race", "sex"].
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
model_list = [
    "Reweighing",\
    "LFR",\
    "DisparateImpactRemover",\
    "Fair-SMOTE",\
    "FairSSL-xT",\
    "FairSSL-Lx",\
    "LTDD",\
    "PrejudiceRemover",\
    "SquaredDifferenceFairLogistic",\
    "FairnessConstraintModel",\
    "DisparateMistreatmentModel",\
    "ConvexFrameworkModel",\
    "HSICLinearRegression",\
    "GerryFairClassifier",\
    "AdversarialDebiasing",\
    "ExponentiatedGradientReduction",\
    "GridSearchReduction",\
    "MetaFairClassifier",\
    "AdaFair",\
    "FAGTB",\
    "HGR",\
    "FairDummies",\
    "GeneralFairERM",\
    "MultiAdversarialDebiasing",\
    "FairGeneralizedLinearModel",\
    "GradualCompatibility",\
    "RejectOptionClassification",\
    "EqOddsPostprocessing",\
    "CalibratedEqOddsPostprocessing",\
    "JiangNachum",\
    "DPAbstention",\
    "FaX",\
    "FairBayesDPP",\
    "GetFair",\
    "FS",\
    "FairRR",\
    "LevEqOpp",\
    "GroupDebias",\
    "PFR",\
    "iFlipper",\
    "fairret"]

#Choose the randomstate. If set to -1, then a random one is taken. Make sure that all algorithms
#use the same randomstate. Also choose the size of the testdata.
randomstate = -1
testsize = 0.3

#Choose if the parameter optimization component should be activated.
tuning = True

#Choose the considered fairness metric: demographic_parity, equalized_odds, treatment_equality, consistency
metric = "demographic_parity"

##DATA PREPARATION SETTINGS START
#Choose if the removal component should be activated: (False, True)
removal = False

#Choose if the binarization component should be activated: (False, True)
binarize = False

#Choose if the balancing component should be activated and which one: (False, "classic", "protected")
balance = False

#Choose if the feature selection component should be activated and which one: (False, "VT", "RFECV")
fselection = False

#Choose if the dimensionality reduction component should be activated: (False, True)
dimred = False
##DATA PREPARATION SETTINGS END

#Choose the dataset to run the experiments on (see the dictionary above)
ds = "communities"
sensitive = data_dict[ds]["sens_attrs"]
label = data_dict[ds]["label"]
favored = data_dict[ds]["favored"]


#Choose where to store the files
link = "Results/" + str(metric) + "/" + str(ds) + "/"

try:
    os.makedirs(link)
except FileExistsError:
    pass

"""
PARAMETER SETTINGS END
"""

subprocess.check_call(['python', '-Wignore', 'main.py', '--output', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--testsize', str(testsize), '--randomstate', str(randomstate),
    '--models', str(model_list), '--metric', str(metric), '--tuning', str(tuning),
    '--binarize', str(binarize), '--removal', str(removal),
    '--balance', str(balance), '--fselection', str(fselection), '--dimred', str(dimred)])


model_list_eval = copy.deepcopy(model_list)
if tuning:
    for model in model_list:
        model_list_eval.append(model + "_tuned")
    model_list_eval.sort()

subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
    '--ds', str(ds), '--sensitive', str(sensitive), '--favored', str(favored),
    '--label', str(label), '--models', str(model_list_eval), '--metric', str(metric)])
