
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import StandardScaler, OneHotEncoder

'''
    Adult Dataset by Ronny Kohavi and Barry Becker
    Visit https://archive.ics.uci.edu/ml/datasets/Adult for details.
'''

# Metadata

TRAIN_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

VARIABLES = {
    'age': Num,
    'workclass': Cat,
    'fnlwgt': Num,
    'education': Cat,
    'education-num': Num,
    'marital-status': Cat,
    'occupation': Cat,
    'relationship': Cat,
    'race': Cat,
    'sex': Cat,
    'capital-gain': Num,
    'capital-loss': Num,
    'hours-per-week': Num,
    'native-country': Cat,
    'income': Cat
}

race_map = {'Amer-Indian-Eskimo': 0,
            'Asian-Pac-Islander': 1,
            'Black': 3,
            'White': 4,
            'Other': 5}

# Dataset Class


class AdultDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'binary'
        self._sensitive_attr_name = 'sex'
        self.process()

    def process(self):
        train_data_path = self.data_dir.joinpath('adult.data')
        test_data_path = self.data_dir.joinpath('adult.test')

        self._check_exists_and_download(train_data_path, TRAIN_URL, self.download)
        self._check_exists_and_download(test_data_path, TEST_URL, self.download)

        train = pd.read_csv(train_data_path, names=VARIABLES.keys())
        test = pd.read_csv(test_data_path, names=VARIABLES.keys(), skiprows=1)

        train = train.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        test = test.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        test['income'] = [row.replace('.', '') for row in test['income']]

        train['marital-status'] = ['married' if status.startswith('Married') else status for status in
                                   train['marital-status']]
        test['marital-status'] = ['married' if status.startswith('Married') else status for status in
                                  test['marital-status']]

        train['race'] = [race_map[race] for race in train['race']]
        test['race'] = [race_map[race] for race in test['race']]

        # ? -> np.nan, drop na
        train = train.replace('?', np.nan).dropna(axis=0, how='any')
        test = test.replace('?', np.nan).dropna(axis=0, how='any')

        # drop some variables
        train = train.drop(['education', 'native-country'], axis=1)
        test = test.drop(['education', 'native-country'], axis=1)

        # One-hot encode categorical variables and standardize continuous variables
        catcols = [name for name in VARIABLES if VARIABLES[name] == Cat if
                   name in train.columns and VARIABLES[name] != 'race']
        numcols = [name for name in VARIABLES if VARIABLES[name] == Num if
                   name in train.columns]

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        train_cat = cat_encoder.fit_transform(train[catcols])
        train_num = num_scaler.fit_transform(train[numcols])
        test_cat = cat_encoder.transform(test[catcols])
        test_num = num_scaler.transform(test[numcols])

        catnewcols = np.concatenate([item[1:] for item in cat_encoder.categories_]).tolist()
        catnewcols[catnewcols == 'Male'] = 'sex'
        catnewcols[-1] = 'target'

        self._train = pd.DataFrame(
            np.column_stack([train_num, train_cat]),
            columns=numcols + catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([test_num, test_cat]),
            columns=numcols + catnewcols)


if __name__ == '__main__':
    data = AdultDataset(data_dir='../datafiles')
