
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

'''

    Obesity Prediction
    Visit https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+ for details.
    
'''

# Some metadata

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic (2).zip"

label_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 5}


class ObesityDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'multi'
        self._sensitive_attr_name = 'gender'
        self.process()

    def process(self):

        zip_file = self.data_dir.joinpath('obesity.zip')
        self._check_exists_and_download(zip_file, DATA_URL, self.download)

        data = pd.read_csv(self.data_dir.joinpath('ObesityDataSet_raw_and_data_sinthetic.csv'), sep=',')

        unique_counter = data.apply(lambda col: len(set(col)), axis=0)
        VARIABLES = {}
        for key, val in unique_counter.iteritems():
            VARIABLES[key] = (Cat, val) if val < 20 else (Num, None)
            
        target = np.array([label_map[t] for t in data['NObeyesdad']])
        gender = (data['Gender'] == 'Female').astype(float)
        
        dat = data.drop(['NObeyesdad', 'Gender'], axis=1)
        dat.loc[dat['CALC'] == 'Always', 'CALC'] = 'Frequently'
        train_dat, test_dat, train_gender, test_gender, train_target, test_target = train_test_split(
            dat, gender, target, test_size=0.3, random_state=self.random_seed)

        # One-hot encode categorical variables and standardize continuous variables
        catcols = [var for var in VARIABLES if VARIABLES[var][0] == Cat if var in train_dat]
        numcols = [var for var in VARIABLES if VARIABLES[var][0] == Num if var in train_dat]

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        _train_cat = cat_encoder.fit_transform(train_dat[catcols])
        _train_num = num_scaler.fit_transform(train_dat[numcols])
        _test_cat = cat_encoder.transform(test_dat[catcols])
        _test_num = num_scaler.transform(test_dat[numcols])

        catnewcols = np.concatenate([item[1:] for item in cat_encoder.categories_]).tolist()

        self._train = pd.DataFrame(
            np.column_stack([_train_num, _train_cat]),
            index=train_dat.index,
            columns=numcols+catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([_test_num, _test_cat]),
            index=test_dat.index,
            columns=numcols+catnewcols)

        self._train.insert(
            loc=0,
            column='gender',
            value=train_gender)

        self._train.insert(
            loc=0,
            column='target',
            value=train_target)

        self._test.insert(
            loc=0,
            column='gender',
            value=test_gender)

        self._test.insert(
            loc=0,
            column='target',
            value=test_target)


if __name__ == '__main__':
    data = ObesityDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
