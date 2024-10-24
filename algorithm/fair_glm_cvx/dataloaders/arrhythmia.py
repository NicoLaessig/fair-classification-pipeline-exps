
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

'''

    Arrhythmia Dataset by H. Altay Guvenir
    Visit https://archive.ics.uci.edu/ml/datasets/Arrhythmia for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data'

COLNAMES = ['sex', 'height', 'age', 'weight',
            'QRS duration', 'P-R interval', 'Q-T interval', 'T interval',
            'P interval', 'QRS', 'T', 'P',
            'QRST', 'J', 'heart rate',
            'Q wave', 'R wave', 'S wave', 'R` wave', 'S` wave',
            'Number of intrinsice deflections',
            'Existence of ragged R wave',
            'Existence of diphasic derivation of R wave',
            'Existence of ragged P wave',
            'Existence of diphasic derivation of P wave',
            'Existence of ragged T wave',
            'Existence of diphasic derivation of T wave']

COLNAMES += [f'V{id:03d}' for id in range(len(COLNAMES) + 1, 279)]
COLNAMES += ['class']

VARTYPES = [Num] * len(COLNAMES)
for i in [0] + list(range(21, 27)) + [278]:
    VARTYPES[i] = Cat


# Note Sex == 1 is female
# Define Arrhythmia data class

class ArrhythmiaDataset(BaseDataset):
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

        data_path = self.data_dir.joinpath('arrhythmia.data')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, names=COLNAMES)

        # drop two unordinary samples with 500+ heights
        data = data[data['height'] < 500]

        # Remove the existence of ragged/disphasic derviation waves variables

        _COLNAMES = np.delete(COLNAMES, [12] + list(range(21, 27)))
        _VARTYPES = np.delete(VARTYPES, [12] + list(range(21, 27)))

        data = data[_COLNAMES]

        # ? -> np.nan, drop na

        data = data.replace('?', np.nan).dropna(axis=0, how='any')
        # Compile binary classification problem: normal vs. others
        target = (data['class'] == 1).astype(float)
        gender = data['sex']

        data.drop(['class', 'sex'], axis=1, inplace=True)
        # remove zero-variance features
        selector = VarianceThreshold(0)
        data = pd.DataFrame(selector.fit_transform(data), index=data.index,
                            columns=data.columns[selector.get_support()])

        data['class'] = target
        data['sex'] = gender

        train, test = train_test_split(data, test_size=0.3, random_state=self.random_seed,
                                       stratify=data['class']*len(np.unique(data['sex'])) + data['sex'])

        # One-hot encode categorical variables and standardize continuous variables
        catcols = [colname for colname, vartype in zip(_COLNAMES, _VARTYPES) if
                   vartype == Cat and colname in data.columns]
        concols = [colname for colname, vartype in zip(_COLNAMES, _VARTYPES) if
                   vartype == Num and colname in data.columns]

        concols = data[concols].columns[np.where(data[concols].median(0) > 1)[0]]
        concols = data[concols].columns[data[concols].std(0).argsort()][:100]

        concols = concols.tolist()

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        train_cat = cat_encoder.fit_transform(train[catcols])
        train_con = num_scaler.fit_transform(train[concols])
        test_cat = cat_encoder.transform(test[catcols])
        test_con = num_scaler.transform(test[concols])

        catnewcols = np.concatenate([item[1:] for item in cat_encoder.categories_]).tolist()
        catnewcols[-2] = 'sex'
        catnewcols[-1] = 'target'

        self._train = pd.DataFrame(
            np.column_stack([train_con, train_cat]),
            columns=concols + catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([test_con, test_cat]),
            columns=concols + catnewcols)




if __name__ == '__main__':
    data = ArrhythmiaDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
