import os

from data import DATASET_ROOT_PATH, DATASET_MIT_BIH
from data.mit_bih_arrhythmia import MIT_BIH_ARRHYTMIA


class DataParser:
    mit_parser = MIT_BIH_ARRHYTMIA()

    def run_eval(self):
        dataset_list = os.listdir(DATASET_ROOT_PATH)
        mit_bih_dataset_list = [item for item in dataset_list if DATASET_MIT_BIH in item]

        for path in mit_bih_dataset_list:
            if 'arrhythmia' in path:
                if 'p-wave' in path:
                    record_file = 'RECORDS'
                else:
                    record_file = 'RECORDS_NOPACE'
            else:
                record_file = 'RECORDS'
            self.mit_parser.run(path=f'{DATASET_ROOT_PATH}/{path}', record_file=record_file)

DataParser().run_eval()