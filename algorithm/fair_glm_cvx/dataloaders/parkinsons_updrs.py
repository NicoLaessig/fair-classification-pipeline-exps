
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
'''

    Parkinson's Telemonitoring Dataset by 
    Visit https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data'

# Define Communities and Crime data class

class ParkinsonsUPDRSDataset(BaseDataset):
    def __init__(self, download=True, data_dir=None, random_seed=0):

        super().__init__(
            download=download,
            data_dir=data_dir,
            random_seed=random_seed
        )

        self._outcome_type = 'continuous'
        self._sensitive_attr_name = 'gender'
        self.process()

    def process(self):

        data_path = self.data_dir.joinpath('parkinsons_updrs.data')

        self._check_exists_and_download(data_path, DATA_URL, self.download)
        
        data = pd.read_csv(data_path, sep=',')
        gender = data['sex']
        target = data['total_UPDRS']
        
        dat = data.drop(['subject#', 'sex', 'motor_UPDRS', 'total_UPDRS', 'test_time'], axis=1)
        
        train_X, test_X, train_A, test_A, train_y, test_y = train_test_split(
            dat, gender, target, test_size=0.3, random_state=self.random_seed)
       
        scaler = StandardScaler()
        self._train = pd.DataFrame(scaler.fit_transform(train_X),
                                   index=train_X.index, columns=train_X.columns)
        self._test = pd.DataFrame(scaler.transform(test_X),
                                  index=test_X.index, columns=test_X.columns)

        self._train['gender'] = train_A
        self._train['target'] = train_y
        self._test['gender'] = test_A
        self._test['target'] = test_y
        

if __name__ == '__main__':
    data = ParkinsonsUPDRS(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
    
