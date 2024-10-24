# Fair Generalized Linear Models with a Convex Penalty

Implementation of the code for reproducing the results of the paper **Fair Generalized Linear Models with a Convex Penalty** published on ICML2022. [[paper](https://proceedings.mlr.press/v162/do22a.html)/[arxiv](https://arxiv.org/abs/2206.09076)]

## Requirements

You have to use ```pip``` to install the requirements ([```requirements.txt```](requirements.txt)) not ```conda```.
```
git clone https://github.com/hyungrok-do/fair-glm-cvx
cd fair-glm-cvx
pip install -r requirements.txt
```
The codes were developed with the following specific versions of packages:
- ```Python==3.x```
- [```scipy==1.7.3```](https://scipy.org/)
- [```numpy==1.21.5```](https://numpy.org/)
- [```pandas==1.3.5```](https://pandas.pydata.org/)
- [```fairlearn==0.7.0```](https://fairlearn.org/)
- [```scikit-learn==1.0.2```](https://scikit-learn.org/)
- [```matplotlib==3.5.1```](https://matplotlib.org/)
- [```pyyaml==6.0```](https://pyyaml.org/)
- [```wget==3.2```](https://pypi.org/project/wget/)
- [```cvxpy==1.2.1```](https://www.cvxpy.org/)
- [```dccp==1.0.3```](https://github.com/cvxgrp/dccp)

Note that ```cvxpy``` is required for
- [```FairConstraintModel```](/models/zafar.py) 
- [```DisparateMistreatmentModel```](/models/zafar.py)
- [```GeneralFairERM```](/models/oneto.py)

and

```dccp``` is required for
- [```FairConstraintModel```](/models/zafar.py) 
- [```DisparateMistreatmentModel```](/models/zafar.py) 

## Reproducing the Paper Results

``` bash reproduce.sh``` will help you to reproduce the paper results (except for the discretization plots. To reproduce Figure 2 in Appendix B, run ```python discretization.py```).

If you are working on a HPC with [```slurm```](https://slurm.schedmd.com/documentation.html), you may use ```sbatch reproduce.s``` (before execute it, you may want to make a dir for the log files: ```mkdir -p ./logs```)

## Run for a Single Dataset
You may substitute a dataset's name (see the **Argument** column of the table below) for ```DATASET```.
```
python experiment.py --dataset DATASET
```

## Configuration
Configuration for each experiment can be found in ```yaml``` files in [```configs```](configs) folder.

## Datasets
We use 11 datasets for our experiments (8 from UCI Machine Learning Repository, except for the COMPAS, LSAC, and HRS).  

|  Outcome   |                                                                             Dataset                                                                              |                       Dataset Class Name                        |           Argument           | Sensitive Attribute | #instances | #features |
|:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------:|:--------------------:|-----------:|----------:|
|   Binary   |                                                    [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)                                                     |           [```AdultDataset```](dataloaders/adult.py)            |         ```adult```          |          Gender (2) |     45,222 |        34 |
|   Binary   |                                               [Arrhythmia](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)                                                |      [```ArrhythmiaDataset```](dataloaders/arrhythmia.py)       |       ```arrhythmia```       |          Gender (2) |        418 |        80 |
|   Binary   |                                                   [COMPAS](https://github.com/propublica/compas-analysis/)                                                    |          [```COMPASDataset```](dataloaders/compas.py)           |         ```compas```         |            Race (4) |      6,172 |        11 |
|   Binary   |                                 [Drug Consumption](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)                                 |    [```DrugConsumptionBinaryDataset```](dataloaders/drug.py)    |    ```drug_consumption```    |            Race (2) |      1,885 |        25 |
|   Binary   |                                   [German Credit](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)                                   |    [```GermanCreditDataset```](dataloaders/german_credit.py)    |     ```german_credit```      |          Gender (2) |      1,000 |        46 |
| Continuous |                                    [Communities and Crime](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)                                     |           [```CrimeDataset```](dataloaders/crime.py)            |         ```crime```          |            Race (3) |      1,993 |        97 |
| Continuous | [Law School](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Pandas_Case_Study.ipynb) |            [```LSACDataset```](dataloaders/lsac.py)             |          ```lsac```          |            Race (5) |     20,715 |         7 |
| Continuous |                                [Parkinsons Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)                                 | [```ParkinsonsUPDRSDataset```](dataloaders/parkinsons_updrs.py) |    ```parkinsons_updrs```    |          Gender (2) |      5,875 |        25 |
| Continuous |                                      [Student Performance](https://archive.ics.uci.edu/ml/datasets/student+performance)                                       |    [```StudentPerformanceDataset```](dataloaders/student.py)    |  ```student_performance```   |          Gender (2) |        649 |        39 |
| Multiclass |                                 [Drug Consumption](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)                                 | [```DrugConsumptionMultiDataset```](dataloaders/drug_multi.py)  | ```drug_consumption_multi``` |            Race (2) |      1,885 |        25 |
| Multiclass |                [Obesity](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+)                 |         [```ObesityDataset```](dataloaders/obesity.py)          |        ```obesity```         |          Gender (2) |      2,111 |        23 | 
|   Count    |                                                                   Health & Retirement Study                                                                   |             [```HRSDataset```](dataloaders/hrs.py)              |         ```hrs```            |            Race (4) |     12,774 |        23 |


## Fairness-aware Methods
We provide implementations of several linear model-based fair approaches (or their linear versions). 

|                  Method                   |                     Model Class Name                      |                                                                                        Reference                                                                                        |                                                                                                                                                                          
|:-----------------------------------------:|:---------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|              Fair Constraint              |     [```FairnessConstraintModel```](models/zafar.py)      |                      [Zafar et al., 2017 (AISTATS)](https://proceedings.mlr.press/v54/zafar17a.html) [[code]](https://github.com/mbilalzafar/fair-classification)                      |
|          Disparate Mistreatment           |    [```DisparateMistreatmentModel```](models/zafar.py)    |                      [Zafar et al., 2017 (WWW)](https://dl.acm.org/doi/abs/10.1145/3038912.3052660) [[code]](https://github.com/mbilalzafar/fair-classification)                       |
|       Squared Difference Penalizer        | [```SquaredDifferenceFairLogistic```](models/bechavod.py) |                                                                [Bechavod et al., 2017](https://arxiv.org/abs/1707.00044)                                                                |
|   Group Fairness / Individual Fairness    |       [```ConvexFrameworkModel```](models/berk.py)        |                                                                  [Berk et al., 2017](https://arxiv.org/abs/1706.02409)                                                                  |
|       Independence Measured by HSIC       |       [```HSICLinearRegression```](models/perez.py)       |                                                         [Perez-Suay et al., 2017](https://doi.org/10.1007/978-3-319-71249-9_21)                                                         |
 |     Fair Empirical Risk Minimization      |           [```LinearFERM```](models/donini.py)            |           [Donini et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/83cdcec08fbf90370fcf53bdd56604ff-Abstract.html) [[code]](https://github.com/jmikko/fair_ERM)            | 
|            Reductions Approach            |       [```ReductionsApproach```](models/agarwal.py)       | [Agarwal et al., 2018](https://proceedings.mlr.press/v80/agarwal18a.html), [2019](https://proceedings.mlr.press/v97/agarwal19d.html) [[code]](https://github.com/fairlearn/fairlearn)  | 
| General Fair Empirical Risk Minimization  |          [```GeneralFairERM```](models/oneto.py)          |                                                          [Oneto et al., 2020](https://doi.org/10.1109/IJCNN48605.2020.9206819)                                                          |                                                                                                                    
|      Fair Generalized Linear Models       |  [```FairGeneralizedLinearModel```](models/fair_glm.py)   |                                                                   [Do et al., 2022](https://proceedings.mlr.press/v162/do22a.html)                                                                   | 


# Citation
Please cite as:

``` bibtex
@InProceedings{pmlr-v162-do22a,
  title     = {Fair Generalized Linear Models with a Convex Penalty},
  author    = {Do, Hyungrok and Putzel, Preston and Martin, Axel S and Smyth, Padhraic and Zhong, Judy},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages     = {5286--5308},
  year      = {2022},
  editor    = {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume    = {162},
  series    = {Proceedings of Machine Learning Research},
  month     = {17--23 Jul},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v162/do22a.html}
}

```
 

