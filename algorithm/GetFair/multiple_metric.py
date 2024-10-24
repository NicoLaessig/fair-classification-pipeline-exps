from utils import DataDispImp, DataDispMis, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from OptimizerModel import MetaOptimizerDirection, MetaOptimizerMLP
from Rewardfunctions import stat_parity, diff_FPR, diff_FNR, diff_FPR_FNR, diff_Eoppr, diff_Eodd
from Optimizer import TrainerEpisodic, TrainerEpisodicMultiple
import math
import numpy as np
import click


def get_data(datatype):

    if datatype == 1: # disparate impact
        name = "statistical parity synthetic dataset"
        means = [[2, 2], [-2, -2]]
        covars = [[[5, 1], [1, 5]], [[10, 1], [1, 3]]]
        n_samples = 5000
        disc_factor = math.pi / 4.0
        data = DataDispImp(means, covars, n_samples, disc_factor)

    elif datatype == 2:  # disparate mistreatment only TPR
        name = "equal opportunity sythetic dataset"
        means = [[-2, -2], [-1, 0], [2, 2], [2, 2]]
        covars = [[[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    elif datatype == 3: # disparate mistreatment both FPR, FNR opposite signs (used)
        name = "equalized odds sythetic dataset"
        means = [[2, 0], [2, 3], [-1, -3], [-1, 0]]
        covars = [[[5, 1], [1, 5]], [[5, 1], [1, 5]], [[5, 1], [1, 5]], [[5, 1], [1, 5]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    elif datatype == 4: # disparate mistreatment both FPR, FNR same sign (not used)
        means = [[1, 2], [2, 3], [0, -1], [-5, 0]]
        covars = [[[5, 2], [2, 5]], [[10, 1], [1, 4]], [[7, 1], [1, 7]], [[5, 1], [1, 5]]]
        n_samples = 2500
        data = DataDispMis(means, covars, n_samples)

    elif datatype == 5:
        name = "adult dataset"
        data = Dataset('Datasets/adult.p')

    elif datatype == 6:
        name = "compas dataset"
        data = Dataset('Datasets/compas_data.p')

    elif datatype == 7:
        name = 'bank dataset'
        data = Dataset('Datasets/bank.p')

    else:
        data = None
        name = None

    return data, name


def select_classifier(name=None):
    clf = None
    clf_name = None
    if name == 'log-reg':
        clf_name = "logistic regression classifier"
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=25000)
    elif name == 'lin-svm':
        clf_name = "linear linear SVM classifier"
        clf = LinearSVC(loss='squared_hinge', max_iter=125000)
    elif name == 'mlp':
        clf_name = "multilayer perceptron"
        clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
    else:
        pass
    return clf, clf_name

@click.command()
@click.option("-d", "--dataset", required=True, type=int,
              help="Dataset to execute", default=6)
@click.option("-c", "--cls", required=True, type=str,
              help="classifier model- log-reg, lin-svm, mlp", default='log-reg')
def main(dataset, cls):
    metric_1 = stat_parity
    metric_2 = diff_Eoppr
    data, d_name = get_data(datatype=dataset)
    name = cls

    print(f"Executing cls: {name}, dataset: {dataset}")
    all_accr_fairness = {}
    for k in range(8):
        all_accr_fairness[k] = []
    accr_ = []
    fair_ = []

    for _ in range(10):
        data.test_train_split_comp()
        classifiertype = 'linear' if name in ('log-reg', 'lin-svm') else 'neuralnet'
        clf, clf_name = select_classifier(name=name)
        clf.fit(data.tr_dt, data.tr_c)
        ts_pred = clf.predict(data.vl_dt)
        accr_.append(accuracy_score(data.vl_c, ts_pred))
        # print(f"accuracy - {accr_[-1]}")
        r_1 = metric_1(data.vl_s, ts_pred, data.vl_c)
        r_2 = metric_2(data.vl_s, ts_pred, data.vl_c)
        fair_.append((r_1, r_2))
        # print(f"fairness - {fair_[-1]}")
        # print('===================================')

        meta = MetaOptimizerDirection(hidden_size=30, layers=2, output_size=2)
        trainer = TrainerEpisodicMultiple(meta, clf, data, (metric_1, metric_2), classifiertype=classifiertype)
        trainer.train(accuracy_threshold=0.55, step_size=0.04, episodes=100)
        res = trainer.get_best_parameters_alt()
        # print(res)

        for key in res:
            all_accr_fairness[key].append(res[key])
        # print(f'Completed iteration - {_}')

    print("########### Final Results ###########\n")
    fair_1, fair_2 = zip(*fair_)
    print(f"Average accuracy - {np.mean(accr_)}")
    print(f"Average fairness score for metric 1- {np.mean(fair_1)}")
    print(f"Average fairness score for metric 2- {np.mean(fair_2)}")
    print("Fairness results after tuning............\n")

    for key in all_accr_fairness:
        if len(all_accr_fairness[key]) > 0:
            scores, rewards = zip(*all_accr_fairness[key])
            fair_1, fair_2 = zip(*rewards)
            print(f"Average accuracy - {np.mean(scores)}")
            print(f"Average fairness score for metric 1- {np.mean(fair_1)}")
            print(f"Average fairness score for metric 2- {np.mean(fair_2)}")
            print("-----------------------------------------------------------------------------")


if __name__ == '__main__':
    main()