
from dataloaders.adult import AdultDataset
from dataloaders.arrhythmia import ArrhythmiaDataset
from dataloaders.german_credit import GermanCreditDataset
from dataloaders.drug import DrugConsumptionBinaryDataset
from dataloaders.drug_multi import DrugConsumptionMultiDataset
from dataloaders.crime import CrimeDataset
from dataloaders.student import StudentPerformanceDataset
from dataloaders.parkinsons_updrs import ParkinsonsUPDRSDataset
from dataloaders.compas import COMPASDataset
from dataloaders.hrs import HRSDataset
from dataloaders.obesity import ObesityDataset
from dataloaders.lsac import LSACDataset

supported_datasets = [
    'adult',
    'arrhythmia',
    'compas',
    'crime',
    'drug_consumption',
    'drug_consumption_multi',
    'german_credit',
    'hrs',
    'lsac',
    'obesity',
    'parkinsons_updrs',
    'student_performance'
]


def get_dataset_by_name(dataset):
    dataset = dataset.lower()
    assert dataset in supported_datasets, f"{dataset} is not supported!"
    if dataset == 'adult':
        return AdultDataset()
    elif dataset == 'arrhythmia':
        return ArrhythmiaDataset()
    elif dataset == 'german_credit':
        return GermanCreditDataset()
    elif dataset == 'drug_consumption':
        return DrugConsumptionBinaryDataset()
    elif dataset == 'drug_consumption_multi':
        return DrugConsumptionMultiDataset()
    elif dataset == 'crime':
        return CrimeDataset()
    elif dataset == 'student_performance':
        return StudentPerformanceDataset()
    elif dataset == 'parkinsons_updrs':
        return ParkinsonsUPDRSDataset()
    elif dataset == 'compas':
        return COMPASDataset()
    elif dataset == 'hrs':
        return HRSDataset()
    elif dataset == 'obesity':
        return ObesityDataset()
    elif dataset == 'lsac':
        return LSACDataset()
    else:
        raise Exception('How did you end up here?')

