
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

'''

    Student Performance Dataset by Paulo Cortez
    Visit https://archive.ics.uci.edu/ml/datasets/student+performance for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'

# c stands for categorical, n stands for numeric
VARIABLES = {'school': Cat,
             'sex': Cat,
             'age': Num,
             'address': Cat,
             'famsize': Cat,
             'Pstatus': Cat,
             'Medu': Num,
             'Fedu': Num,
             'Mjob': Cat,
             'Fjob': Cat,
             'reason': Cat,
             'guardian': Cat,
             'traveltime': Num,
             'studytime': Num,
             'failures': Num,
             'schoolsup': Cat,
             'famsup': Cat,
             'paid': Cat,
             'activities': Cat,
             'nursery': Cat,
             'higher': Cat,
             'internet': Cat,
             'romantic': Cat,
             'famrel': Num,
             'freetime': Num,
             'goout': Num,
             'Dalc': Num,
             'Walc': Num,
             'health': Num,
             'absences': Num,
             'G1': Num,
             'G2': Num,
             'G3': Num}

# Define Drug Consumption data class


class StudentPerformanceDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'continuous'
        self._sensitive_attr_name = 'sex'
        self.process()

    def process(self):

        zip_file = self.data_dir.joinpath('student.zip')
        self._check_exists_and_download(zip_file, DATA_URL, self.download)

        data = pd.read_csv(self.data_dir.joinpath('student-por.csv'), sep=';')

        drop_col = ['school', 'G1', 'G2']
        dat = data.drop(drop_col, axis=1)

        self._train, self._test = train_test_split(dat, test_size=0.3, random_state=self.random_seed)

        catcols = [var for var in VARIABLES if VARIABLES[var] == Cat if var not in drop_col]
        numcols = [var for var in VARIABLES if VARIABLES[var] == Num if var not in drop_col]

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        _train_cat = cat_encoder.fit_transform(self._train[catcols])
        _train_num = num_scaler.fit_transform(self._train[numcols])
        _test_cat = cat_encoder.transform(self._test[catcols])
        _test_num = num_scaler.transform(self._test[numcols])

        catnewcols = np.concatenate(
            [[cat] if len(item) == 2 else [cat+'_'+cn for cn in item[1:]] for cat, item in zip(catcols, cat_encoder.categories_)]).tolist()

        self._train = pd.DataFrame(
            np.column_stack([_train_num, _train_cat]),
            columns=numcols+catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([_test_num, _test_cat]),
            columns=numcols+catnewcols)

        self._train.loc[:, 'target'] = self._train.loc[:, 'G3']
        self._train = self._train.drop(['G3'], axis=1)

        self._test.loc[:, 'target'] = self._test.loc[:, 'G3']
        self._test = self._test.drop(['G3'], axis=1)

if __name__ == '__main__':
    data = StudentPerformanceDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
