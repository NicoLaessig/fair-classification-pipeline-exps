
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

'''

    LSAC Dataset from COLAB tutorial
    https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Pandas_Case_Study.ipynb
'''

# Some metadata

DATA_URL = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'

race_map = {'asian': 0,
            'black': 1,
            'hisp': 2,
            'other': 3,
            'white': 4}

VARIABLES = {
    'race1': Cat,
    'gender': Cat,
    'age': Num,
    'fam_inc': Num,
    'fulltime': Cat,
    'zgpa': Num,
    'ugpa': Num,
    'lsat': Num
}
# Define Drug Consumption data class

class LSACDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'continuous'
        self._sensitive_attr_name = 'race'
        self.process()

    def process(self):

        data_path = self.data_dir.joinpath('bar_pass_prediction.csv')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, sep=',', index_col=0)

        data = data[['race1', 'gender', 'age', 'fam_inc', 'fulltime', 'zgpa', 'ugpa', 'lsat']]

        data = data.dropna(axis=0)

        '''
        unique_counter = data.apply(lambda col: len(set(col)), axis=0)

        VARIABLES = {}
        for key, val in unique_counter.iteritems():
            VARIABLES[key] = ('cat', val) if val < 20 else ('num', None)
        '''

        target = data['zgpa']
        race = [race_map[r] for r in data['race1']]

        dat = data.drop(['zgpa', 'race1'], axis=1)

        train_dat, test_dat, train_race, test_race, train_target, test_target = train_test_split(
            dat, race, target, test_size=0.3, random_state=self.random_seed)

        # One-hot encode categorical variables and standardize continuous variables
        catcols = [var for var in VARIABLES if VARIABLES[var] == Cat if var in train_dat.columns]
        numcols = [var for var in VARIABLES if VARIABLES[var] == Num if var in train_dat.columns]

        #catcols.remove('fam_inc')
        #numcols.append('fam_inc')

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        _train_cat = cat_encoder.fit_transform(train_dat[catcols])
        _train_num = num_scaler.fit_transform(train_dat[numcols])
        _test_cat = cat_encoder.transform(test_dat[catcols])
        _test_num = num_scaler.transform(test_dat[numcols])

        catnewcols = np.concatenate([item[1:] for item in cat_encoder.categories_]).tolist()
        # catnewcols = np.concatenate([item[1:] for item in self._cat_enc.categories_]).tolist()

        self._train = pd.DataFrame(
            np.column_stack([_train_num, _train_cat]),
            index=train_dat.index,
            columns=numcols + catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([_test_num, _test_cat]),
            index=test_dat.index,
            columns=numcols + catnewcols)

        self._train.insert(
            loc=0,
            column='race',
            value=train_race)

        self._train.insert(
            loc=0,
            column='target',
            value=train_target)

        self._test.insert(
            loc=0,
            column='race',
            value=test_race)

        self._test.insert(
            loc=0,
            column='target',
            value=test_target)


if __name__ == '__main__':
    data = LSACDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
    print(data.train.head())
