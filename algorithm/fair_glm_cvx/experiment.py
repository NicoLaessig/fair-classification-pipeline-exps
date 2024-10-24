
import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import pickle

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from util import Evaluator
from models import define_models
from dataloaders import get_dataset_by_name

save_path = os.path.join(os.getcwd(), 'results')
os.makedirs(save_path, exist_ok=True)

parser = ArgumentParser()
parser.add_argument('--dataset', default='arrhythmia')
args = parser.parse_args()

dataset = args.dataset

with open(f'configs/{dataset}.yaml', 'rb') as f:
    configs = yaml.safe_load(f)

data = get_dataset_by_name(dataset)
models = define_models(configs['models'])
random_seeds = range(configs['seed'], configs['seed'] + configs['num_repeats'])
sensitive_predictor = configs['sensitive_predictor']

if data.outcome == 'binary':
    family = 'bernoulli'
elif data.outcome == 'count':
    family = 'poisson'
elif data.outcome == 'multi':
    family = 'multinomial'
else:
    family = 'normal'

evaluator = Evaluator(family=family)
results = dict()
for model in models:
    results[model] = dict()

for seed in random_seeds:
    data.reset(seed)
    train_y = data.train['target'].values
    train_A = data.train[data.sensitive].values
    train_X = data.train.drop(['target', data.sensitive], axis=1).values

    test_y = data.test['target'].values
    test_A = data.test[data.sensitive].values
    test_X = data.test.drop(['target', data.sensitive], axis=1).values

    train_X = np.column_stack([train_A, train_X])
    if sensitive_predictor:
        test_X = np.column_stack([test_A, test_X])

    for model_name, (model_instance, param_grid) in models.items():
        results[model_name][seed] = dict()
        model = model_instance(standardize=False,
                               sensitive_predictor=sensitive_predictor)

        if 'family' in list(model.get_params().keys()):
            setattr(model, 'family', family)

        if model_name == 'SP':
            setattr(model, 'penalty', 'SP')
        elif model_name == 'BGL':
            setattr(model, 'penalty', 'BGL')
        elif model_name == 'IF':
            setattr(model, 'penalty', 'individual')
        elif model_name == 'GF':
            setattr(model, 'penalty', 'group')

        for param in param_grid:
            model.set_params(**param)
            model.fit(train_X, train_y)

            if model.status:
                if family == 'bernoulli':
                    test_p = model.predict_proba(test_X)[:, 1]
                elif family == 'multinomial':
                    test_p = model.predict_proba(test_X)
                    test_y = model._enc.transform(test_y.reshape(-1, 1)) if test_y.ndim == 1 else test_y
                else:
                    test_p = model.predict(test_X)

                results[model_name][seed][list(param.values())[0]] = evaluator.evaluate(test_y, test_p, test_A)


for model_name in models:
    results[model_name]['avg'] = dict()
    results[model_name]['std'] = dict()
    results[model_name]['Q1'] = dict()
    results[model_name]['Q3'] = dict()
    for param in results[model_name][seed]:
        stack = [pd.DataFrame(results[model_name][seed][param], index=[seed]) for seed in random_seeds if param in list(results[model_name][seed].keys())]
        results[model_name]['avg'][param] = pd.DataFrame(np.mean(stack, axis=0),
                                                         columns=stack[-1].columns,
                                                         index=stack[-1].index)

        results[model_name]['std'][param] = pd.DataFrame(np.std(stack, axis=0),
                                                         columns=stack[-1].columns,
                                                         index=stack[-1].index)

        results[model_name]['Q1'][param] = pd.DataFrame(np.quantile(stack, q=0.25, axis=0),
                                                         columns=stack[-1].columns,
                                                         index=stack[-1].index)

        results[model_name]['Q3'][param] = pd.DataFrame(np.quantile(stack, q=0.75, axis=0),
                                                        columns=stack[-1].columns,
                                                        index=stack[-1].index)


with open(os.path.join(save_path, f'{dataset}-results.json'), 'wb') as f:
    pickle.dump(results, f)

with open('configs/_graphics.yaml', 'rb') as f:
    graphics_param = yaml.safe_load(f)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['legend.handlelength'] = 0
matplotlib.rcParams['legend.numpoints'] = 1

for metric in evaluator.metric_names:
    for disparity in evaluator.disparity_names:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        for model_name in models:
            avg = results[model_name]['avg']
            std = results[model_name]['std']
            Q1 = results[model_name]['Q1']
            Q3 = results[model_name]['Q3']

            if len(random_seeds) == 1:
                ax.plot(
                    [avg[param][disparity] for param in sorted(avg.keys())],
                    [avg[param][metric] for param in sorted(avg.keys())],
                    label=model_name,
                    **graphics_param[model_name])

            else:
                graphics_param[model_name]['ecolor'] = graphics_param[model_name]['color']

                x_val = np.concatenate([avg[param][disparity].values for param in sorted(avg.keys())])
                y_val = np.concatenate([avg[param][metric].values for param in sorted(avg.keys())])

                y_val = 1 - y_val if metric == 'auroc' else y_val

                x_err = np.row_stack([
                    np.concatenate([avg[param][disparity].values - Q1[param][disparity].values for param in sorted(avg.keys())]),
                    np.concatenate([Q3[param][disparity].values - avg[param][disparity].values for param in sorted(avg.keys())])]
                )

                y_err = np.row_stack([
                    np.concatenate([avg[param][metric].values - Q1[param][metric].values for param in sorted(avg.keys())]),
                    np.concatenate([Q3[param][metric].values - avg[param][metric].values for param in sorted(avg.keys())])]
                )

                eb = ax.errorbar(
                    x_val, y_val, y_err, x_err,
                    label=model_name,
                    elinewidth=1, capsize=0, capthick=0.0,
                    **graphics_param[model_name])

                eb[-1][0].set_linestyle('--')
                eb[-1][0].set_alpha(0.3)
                eb[-1][1].set_linestyle('--')
                eb[-1][1].set_alpha(0.3)

        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_locator(MaxNLocator(6))

        metric_ = '1-auroc' if metric == 'auroc' else metric

        disparity_ = 'DELL' if 'negative_log_likelihood' in disparity else disparity
        disparity_ = 'DEO' if 'EO' in disparity_ else disparity_
        print(disparity, disparity_)

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(f'{disparity_}', fontsize=16)
        ax.set_ylabel(f'{metric_}', fontsize=16)

        plt.tight_layout()

        if family == 'bernoulli':
            plt.legend(fontsize=11, borderpad=0.8, loc='best', ncol=2)
        else:
            plt.legend(fontsize=11, borderpad=0.8, loc='best')

        plt.savefig(os.path.join(save_path, f'{dataset}+{metric}+{disparity}.pdf'), dpi=200)
        plt.close(fig)
