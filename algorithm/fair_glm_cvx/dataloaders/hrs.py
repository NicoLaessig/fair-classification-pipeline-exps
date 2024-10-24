
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataloaders import BaseDataset

# Some metadata

DATA_URL = None

race_map = {'NHW': 0,
            'NHB': 1,
            'Hispanic': 2,
            'Other': 3}

# Define Drug Consumption data class

class HRSDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'count'
        self._sensitive_attr_name = 'race'
        self.process()

    def process(self):

        data_path = self.data_dir.joinpath('HRS_ADL_IADL.csv')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, sep=',', index_col=0)
        data = data.drop(['year', 'BIRTHYR', 'HISPANIC', 'race'], axis=1)
        data = data.dropna(axis=0)
        
        cat = pd.DataFrame(index=data.index)
        cat['marriage'] = (data['marriage'].values == 'Not Married').astype(float)
        cat['gender'] = (data['gender'] == 'Female').astype(float)
        race = np.array([race_map[r] for r in data['race.ethnicity']])
        
        num = data.drop(['marriage', 'gender', 'score', 'race.ethnicity'], axis=1)
        target = data['score']
        
        train_num, test_num, train_cat, test_cat, train_race, test_race, train_y, test_y = train_test_split(
            num, cat, race, target, test_size=0.3, random_state=self.random_seed)
        
        scaler = StandardScaler()
        self._train = pd.DataFrame(scaler.fit_transform(train_num),
                                   index=train_num.index,
                                   columns=train_num.columns)

        self._test = pd.DataFrame(scaler.transform(test_num),
                                   index=test_num.index,
                                   columns=test_num.columns)

        self._train.insert(
            loc=0,
            column='marriage',
            value=train_cat['marriage'])

        self._train.insert(
            loc=0,
            column='gender',
            value=train_cat['gender'])

        self._train.insert(
            loc=0,
            column='race',
            value=train_race)

        self._train.insert(
            loc=0,
            column='target',
            value=train_y)

        self._test.insert(
            loc=0,
            column='marriage',
            value=test_cat['marriage'])

        self._test.insert(
            loc=0,
            column='gender',
            value=test_cat['gender'])

        self._test.insert(
            loc=0,
            column='race',
            value=test_race)

        self._test.insert(
            loc=0,
            column='target',
            value=test_y)


if __name__ == '__main__':
    data = HRSDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
    print(np.unique(data.train.race, return_counts=True))
    
    
    
