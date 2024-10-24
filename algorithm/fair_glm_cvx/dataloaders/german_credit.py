
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

'''

    German Credit Dataset by Dr. Hans Hofmann
    Visit https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

VARIABLES = {
    'Status of Checking Account': Cat,
    'Duration in Month': Num,
    'Credit History': Cat,
    'Purpose': Cat,
    'Credit Amount': Num,
    'Savings Account/Bonds': Cat,
    'Employment': Cat,
    'Disposable Income': Num,
    'Personal Status': Cat,
    'Other Debtors/Guarantors': Cat,
    'Present Residence Since': Num,
    'Property': Cat,
    'Age': Num,
    'Other Installment Plans': Cat,
    'Housing': Cat,
    'Number of Existing Credits at This Bank': Num,
    'Job': Cat,
    'Number of People Being Liable to Provide Maintenance for': Num,
    'Telephone': Cat,
    'Foreign Worker': Cat,
    'credit': Cat
}
    
# Note Sex == 1 is female

# Define German Credit data class

class GermanCreditDataset(BaseDataset):
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

        data_path = self.data_dir.joinpath('german.data')

        self._check_exists_and_download(data_path, DATA_URL, self.download)

        data = pd.read_csv(data_path, names=VARIABLES.keys(), sep=' ')
        
        # Define marital status and sex based on Attr 9
        
        data['single'] = [row in ['A93', 'A95'] for row in data['Personal Status']]
        data['sex'] = [row in ['A92', 'A95'] for row in data['Personal Status']]
        
        data['single'] = data['single'].astype(float)
        data['sex'] = data['sex'].astype(float)
        
        data['class'] = data['credit']-1
        
        data.drop(['Personal Status', 'credit'], axis=1)
        
        # ? -> np.nan, drop na
        data = data.replace('?', np.nan).dropna(axis=0, how='any')
        
        # Compile binary classification problem: normal vs. others
        data['class'] = (data['class'] == 1).astype(float)
        
        # Remove the exsitence of ragged/disphasic derviation waves variables
        _COLNAMES = np.delete(list(VARIABLES.keys()), [8, 20])
        _VARTYPES = np.delete(list(VARIABLES.values()), [8, 20])

        _COLNAMES = np.concatenate([_COLNAMES, ['single', 'sex', 'class']]).tolist()
        _VARTYPES = np.concatenate([_VARTYPES, [Cat, Cat, Cat]]).tolist()
        
        self._train, self._test = train_test_split(data, test_size=0.3, random_state=self.random_seed)

        # One-hot encode categorical variables and standardize continuous variables
        catcols = [colname for colname, vartype in zip(_COLNAMES, _VARTYPES) if vartype == Cat]
        numcols = [colname for colname, vartype in zip(_COLNAMES, _VARTYPES) if vartype == Num]

        cat_encoder = OneHotEncoder(sparse=False, drop='first')
        num_scaler = StandardScaler(with_mean=True, with_std=True)

        _train_cat = cat_encoder.fit_transform(self._train[catcols])
        _train_con = num_scaler.fit_transform(self._train[numcols])
        _test_cat = cat_encoder.transform(self._test[catcols])
        _test_con = num_scaler.transform(self._test[numcols])

        catnewcols = np.concatenate([item[1:] for item in cat_encoder.categories_]).tolist()
        catnewcols[-4] = 'foreign'
        catnewcols[-3] = 'single'
        catnewcols[-2] = 'sex'
        catnewcols[-1] = 'target'

        self._train = pd.DataFrame(
            np.column_stack([_train_con, _train_cat]),
            columns=numcols+catnewcols)

        self._test = pd.DataFrame(
            np.column_stack([_test_con, _test_cat]),
            columns=numcols+catnewcols)

    
if __name__ == '__main__':
    data = GermanCreditDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
