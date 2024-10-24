import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

'''

    Drug Consumption Dataset by Evgeny M. Mirkes
    Visit https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29 for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data'

VARIABLES = {'ID': Cat,  # not used for prediction
             'Age': Num,
             'Gender': Cat,
             'Education': Cat,
             'Country': Cat,
             'Race': Cat,
             'Nscore': Num,
             'Escore': Num,
             'Oscore': Num,
             'Ascore': Num,
             'Cscore': Num,
             'Impulsive': Num,
             'SS': Num,
             # Drug usage outcomes
             'Alcohol': Cat,
             'Amphet': Cat,
             'Amyl': Cat,
             'Benzos': Cat,
             'Caff': Cat,
             'Cannabis': Cat,
             'Choc': Cat,
             'Coke': Cat,
             'Crack': Cat,
             'Ecstasy': Cat,
             'Heroin': Cat,
             'Ketamine': Cat,
             'Legalh': Cat,
             'LSD': Cat,
             'Meth': Cat,
             'Mushrooms': Cat,
             'Nicotine': Cat,
             'Semer': Cat,
             'VSA': Cat
             }

gender_tr = {
    0.48246: 'Female',  # 49.97%
    -0.48246: 'Male'}  # 50.03%

# Not sure if I should transform this into categories
education_tr = {
    -2.43591: 'Left school before 16',
    -1.73790: 'Left school at 16',
    -1.43719: 'Left school at 17',
    -1.22751: 'Left school at 18',
    -0.61113: 'Some college or university, no certificate or degree',
    -0.05921: 'Professional certificate / diploma',
    0.45468: 'University degree',
    1.16365: 'Masters degree',
    1.98437: 'Doctorate degree'}

country_tr = {
    -0.09765: 'Australia',
    0.24923: 'Canada',
    -0.46841: 'New Zealand',
    -0.28519: 'Other',
    0.21128: 'Republic of Ireland',
    0.96082: 'UK',
    -0.57009: 'USA'}

race_tr = {
    -0.50212: 'Asian',  # 1.38% (  26)
    -1.10702: 'Black',  # 1.75% (  33)
    1.90725: 'Mixed-Black/Asian',  # 0.16% (   3)
    0.12600: 'Mixed-White/Asian',  # 1.06% (  20)
    -0.22166: 'Mixed-White/Black',  # 1.06% (  20)
    0.11440: 'Other',  # 3.34% (  63)
    -0.31685: 'White'}  # 91.25% (1720)

# because of the scarcity we may want to try just white vs. others
race_bin_tr = {
    -0.50212: 'Non-white',  # 1.38% (  26)
    -1.10702: 'Non-white',  # 1.75% (  33)
    1.90725: 'Non-white',  # 0.16% (   3)
    0.12600: 'Non-white',  # 1.06% (  20)
    -0.22166: 'Non-white',  # 1.06% (  20)
    0.11440: 'Non-white',  # 3.34% (  63)
    -0.31685: 'White'}  # 91.25% (1720)

class_map = {
    'CL0': 0,
    'CL1': 1,
    'CL2': 1,
    'CL3': 1,
    'CL4': 2,
    'CL5': 2,
    'CL6': 2
}


# Define Drug Consumption data class


class DrugConsumptionMultiDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):
        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'multi'
        self._sensitive_attr_name = 'Race'
        self._target = 'Meth'
        self.process()

    def process(self):
        data_path = self.data_dir.joinpath('drug_consumption.data')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, names=VARIABLES.keys(), sep=',', index_col=0)

        data['Gender'] = [gender_tr[raw] for raw in data['Gender']]
        data['Education'] = [education_tr[raw] for raw in data['Education']]
        data['Country'] = [country_tr[raw] for raw in data['Country']]
        data['Race'] = [race_bin_tr[raw] for raw in data['Race']]

        # The dataset if normalized to have (almost) zero-mean unit-variance so I do not do standardize
        # But still needs to be one-hot encoded

        education_enc = OneHotEncoder(sparse=False, drop='first')
        education_cat = pd.DataFrame(
            education_enc.fit_transform(data['Education'].values.reshape(-1, 1)),
            columns=education_enc.categories_[0][1:],
            index=data.index)

        data = pd.concat([data, education_cat], axis=1)

        country_enc = OneHotEncoder(sparse=False, drop='first')
        country_cat = pd.DataFrame(
            country_enc.fit_transform(data['Country'].values.reshape(-1, 1)),
            columns=country_enc.categories_[0][1:],
            index=data.index)

        data = pd.concat([data, country_cat], axis=1)

        data = data.drop(['Education', 'Country'], axis=1)

        data.loc[:, 'Gender'] = (data['Gender'] == 'Female').astype(float)
        data.loc[:, 'Race'] = (data['Race'] == 'White').astype(float)

        self._train, self._test = train_test_split(data, test_size=0.3, random_state=self.random_seed)

        target = self._target
        potential_targets = list(VARIABLES.keys())[13:]
        assert target in potential_targets, f'{target} is not a in the dataset'
        _train_target = self._train[target]
        _test_target = self._test[target]
        #potential_targets.remove(target)
        self._train = self._train.drop(potential_targets, axis=1)
        self._test = self._test.drop(potential_targets, axis=1)

        self._train.loc[:,'target'] = [class_map[item] for item in _train_target]
        self._test.loc[:,'target'] = [class_map[item] for item in _test_target]


if __name__ == '__main__':
    data = DrugConsumptionMultiDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
