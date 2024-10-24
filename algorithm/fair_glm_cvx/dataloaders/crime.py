
import numpy as np
import pandas as pd
from dataloaders import BaseDataset, Cat, Num
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

'''

    Communities and Crime Dataset by Michael Redmond
    Visit https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime for details.

'''

# Some metadata

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'

VARIABLES = {
    # not used for prediction: will be removed
    'state':         Cat,
    'county':        Cat,
    'community':     Cat,
    'communityname': Cat,
    'fold':          Cat,
    # predictive variables start
    'population':    Num,
    'householdsize': Num,
    'racepctblack':  Num,
    'racePctWhite':  Num,
    'racePctAsian':  Num,
    'racePctHisp':   Num,
    'agePct12t21':   Num,
    'agePct12t29':   Num,
    'agePct16t24':   Num,
    'agePct65up':    Num,
    'numbUrban':     Num,
    'pctUrban':      Num,
    'medIncome':     Num,
    'pctWWage':      Num,
    'pctWFarmSelf':  Num,
    'pctWInvInc':    Num,
    'pctWSocSec':    Num,
    'pctWPubAsst':   Num,
    'pctWRetire':    Num,
    'medFamInc':     Num,
    'perCapInc':     Num,
    'whitePerCap':   Num,
    'blackPerCap':   Num,
    'indianPerCap':  Num,
    'AsianPerCap':   Num,
    'OtherPerCap':   Num,
    'HispPerCap':    Num,
    'NumUnderPov':   Num,
    'PctPopUnderPov': Num,
    'PctLess9thGrade': Num,
    'PctNotHSGrad':  Num,
    'PctBSorMore':   Num,
    'PctUnemployed': Num,
    'PctEmploy':     Num,
    'PctEmplManu':   Num,
    'PctEmplProfServ': Num,
    'PctOccupManu':  Num,
    'PctOccupMgmtProf': Num,
    'MalePctDivorce': Num,
    'MalePctNevMarr': Num,
    'FemalePctDiv': Num,
    'TotalPctDiv':  Num,
    'PersPerFam':   Num,
    'PctFam2Par':   Num,
    'PctKids2Par':  Num,
    'PctYoungKids2Par': Num,
    'PctTeen2Par': Num,
    'PctWorkMomYoungKids': Num,
    'PctWorkMom': Num,
    'NumIlleg': Num,
    'PctIlleg': Num,
    'NumImmig': Num,
    'PctImmigRecent': Num,
    'PctImmigRec5': Num,
    'PctImmigRec8': Num,
    'PctImmigRec10': Num,
    'PctRecentImmig': Num,
    'PctRecImmig5': Num,
    'PctRecImmig8': Num,
    'PctRecImmig10': Num,
    'PctSpeakEnglOnly': Num,
    'PctNotSpeakEnglWell': Num,
    'PctLargHouseFam': Num,
    'PctLargHouseOccup': Num,
    'PersPerOccupHous': Num,
    'PersPerOwnOccHous': Num,
    'PersPerRentOccHous': Num,
    'PctPersOwnOccup': Num,
    'PctPersDenseHous': Num,
    'PctHousLess3BR': Num,
    'MedNumBR': Num,
    'HousVacant': Num,
    'PctHousOccup': Num,
    'PctHousOwnOcc': Num,
    'PctVacantBoarded': Num,
    'PctVacMore6Mos': Num,
    'MedYrHousBuilt': Num,
    'PctHousNoPhone': Num,
    'PctWOFullPlumb': Num,
    'OwnOccLowQuart': Num,
    'OwnOccMedVal': Num,
    'OwnOccHiQuart': Num,
    'RentLowQ': Num,
    'RentMedian': Num,
    'RentHighQ': Num,
    'MedRent': Num,
    'MedRentPctHousInc': Num,
    'MedOwnCostPctInc': Num,
    'MedOwnCostPctIncNoMtg': Num,
    'NumInShelters': Num,
    'NumStreet': Num,
    'PctForeignBorn': Num,
    'PctBornSameState': Num,
    'PctSameHouse85': Num,
    'PctSameCity85': Num,
    'PctSameState85': Num,
    'LemasSwornFT': Num,
    'LemasSwFTPerPop': Num,
    'LemasSwFTFieldOps': Num,
    'LemasSwFTFieldPerPop': Num,
    'LemasTotalReq': Num,
    'LemasTotReqPerPop': Num,
    'PolicReqPerOffic': Num,
    'PolicPerPop': Num,
    'RacialMatchCommPol': Num,
    'PctPolicWhite': Num,
    'PctPolicBlack': Num,
    'PctPolicHisp': Num,
    'PctPolicAsian': Num,
    'PctPolicMinor': Num,
    'OfficAssgnDrugUnits': Num,
    'NumKindsDrugsSeiz': Num,
    'PolicAveOTWorked': Num,
    'LandArea': Num,
    'PopDens': Num,
    'PctUsePubTrans': Num,
    'PolicCars': Num,
    'PolicOperBudg': Num,
    'LemasPctPolicOnPatr': Num,
    'LemasGangUnitDeploy': Num,
    'LemasPctOfficDrugUn': Num,
    'PolicBudgPerPop': Num,
    'ViolentCrimesPerPop': Num}


# Define Communities and Crime data class


class CrimeDataset(BaseDataset):
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
        data_path = self.data_dir.joinpath('communities.data')
        
        self._check_exists_and_download(data_path, DATA_URL, self.download)
                
        data = pd.read_csv(data_path, names=VARIABLES.keys(), sep=',')
        # remove the variables not used for prediction
        data = data.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1)

        # drop observations with missing values
        data = data.replace('?', np.nan)
        #print(data.isna().mean(0).tolist())
        data = data[data.columns[data.isna().mean(0) < 0.1]]
        data = data.replace('?', np.nan).dropna(axis=0, how='any')
        data = data.astype(float)
        
        #target = np.log(data['ViolentCrimesPerPop'].values + 1)
        target = data['ViolentCrimesPerPop'].values
        race = np.argmax(data[['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']].values, 1)
        #race = race == 0
        data = data.drop(['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'ViolentCrimesPerPop'], axis=1)

        sel = VarianceThreshold(0.01)
        _data = sel.fit_transform(data)
        data = pd.DataFrame(_data, columns=data.columns[sel.get_support()], index=data.index)
        self._train, self._test, train_race, test_race, train_y, test_y = train_test_split(
            data, race, target, test_size=0.3, random_state=self.random_seed)

        scaler = StandardScaler(with_mean=True, with_std=True)
        _train = scaler.fit_transform(self._train)
        _test = scaler.transform(self._test)

        self._train = pd.DataFrame(_train, columns=self._train.columns)
        self._test = pd.DataFrame(_test, columns=self._test.columns)

        self._train.loc[:, 'race'] = train_race
        self._train.loc[:, 'target'] = train_y
        self._test.loc[:, 'race'] = test_race
        self._test.loc[:, 'target'] = test_y
        

if __name__ == '__main__':
    data = CrimeDataset(data_dir='../datafiles')
    print(data.train.shape)
    print(data.test.shape)
