
import wget
import zipfile
from pathlib import Path

Cat = 'categorical'
Num = 'numeric'


class BaseDataset:
    def __init__(self, download=True, data_dir=None, random_seed=0):

        if data_dir is None:
            data_dir = './datafiles'

        data_dir = Path(data_dir)

        self.download = download
        self.data_dir = data_dir
        self.random_seed = random_seed
        self._train = None
        self._test = None
        self._outcome_type = None
        self._sensitive_attr_name = None

    def reset(self, random_seed):
        self.__init__(random_seed=random_seed)

    def process(self):
        raise NotImplementedError

    @staticmethod
    def _check_exists_and_download(data, url, download):
        if not data.exists():
            if download:
                print(f'Download dataset from {url} to {data}')
                if not data.parent.exists():
                    data.parent.mkdir(parents=True)
                wget.download(url, out=str(data))

            else:
                raise FileNotFoundError(f'{data} does not exist but the download option is disabled')

        if data.suffix == '.zip':
            print(f'Extract zip file from {data.parent}')
            with zipfile.ZipFile(data, 'r') as f:
                f.extractall(data.parent)


    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def outcome(self):
        return self._outcome_type

    @property
    def sensitive(self):
        return self._sensitive_attr_name

