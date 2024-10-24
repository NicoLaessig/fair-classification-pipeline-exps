
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

'''

    COMPAS Dataset provided by ProPublica
    Visit https://github.com/hyungrok-do/fglm-cvx-dev/blob/main/dataloaders/compas.py for details.

'''

# Some metadata

DATA_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

race_map = {'African-American': 0,
            'Caucasian': 1,
            'Asian': 2,
            'Other': 2,
            'Hispanic': 2,
            'Native American': 2}


# Define Drug Consumption data class

class COMPASDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'binary'
        self._sensitive_attr_name = 'race'
        self.process()

    def process(self):

        data_path = self.data_dir.joinpath('compas-scores-two-years.csv')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, sep=',', index_col=0)

        data = data[data['is_recid'] != -1]
        data = data[data['days_b_screening_arrest'] <= 30]
        data = data[data['days_b_screening_arrest'] >= -30]
        data = data[data['c_charge_degree'] != 'O']

        target = data['two_year_recid'].astype(float)
        race = data['race']
        race = np.array([race_map[r] for r in race]).astype(float)

        dat = data[['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]

        dat.insert(
            column='c_charge_degree',
            value=data['c_charge_degree'] == 'M',
            loc=0)

        sex = data['sex'] == 'Female'
        sex = sex.astype(float)

        dat.insert(
            column='days_in_jail',
            value=(pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in'])).dt.days,
            loc=0)

        dat.insert(
            column='days_in_custody',
            value=(pd.to_datetime(data['out_custody']) - pd.to_datetime(data['in_custody'])).dt.days,
            loc=0)

        dat = dat.astype(float)

        train_X, test_X, train_race, test_race, train_sex, test_sex, train_y, test_y = train_test_split(
            dat, race, sex, target, test_size=0.3, random_state=self.random_seed)

        scaler = StandardScaler()

        self._train = pd.DataFrame(scaler.fit_transform(train_X),
                                   columns=train_X.columns,
                                   index=train_X.index)

        self._test = pd.DataFrame(scaler.transform(test_X),
                                  columns=test_X.columns,
                                  index=test_X.index)

        self._train.insert(
            column='sex',
            value=train_sex,
            loc=0)

        self._train.insert(
            column='race',
            value=train_race,
            loc=0)

        self._train.insert(
            column='target',
            value=train_y,
            loc=0)

        self._test.insert(
            column='sex',
            value=test_sex,
            loc=0)

        self._test.insert(
            column='race',
            value=test_race,
            loc=0)

        self._test.insert(
            column='target',
            value=test_y,
            loc=0)


if __name__ == '__main__':
    data = COMPASDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
