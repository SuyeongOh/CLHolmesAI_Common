import json
import os

import numpy as np
import wfdb

from test.dataloader import BASE_PARSED_DB_PATH, BASE_RAW_DATA_PATH, LEAD2_TAGS
from test.datamodel.BaseDataModel import BaseDataModel


class MitArrhythmiaLoader:
    dataset_path = ['mit-bih-arrhythmia-database-1.0.0/']

    def __init__(self, type: 'str'):
        if type == 'delineate':
            self.dataset_path = [
                'mit-bih-arrhythmia-database-1.0.0/',
                'mit-bih-noise-stress-test-database-1.0.0',
                'mit-bih-arrhythmia-database-p-wave-annotations-1.0.0'
            ]
        elif type == 'atrial':
            self.dataset_path = [
                'mit-bih-arrhythmia-database-1.0.0/',
                'mit-bih-atrial-fibrillation-database-1.0.0'
            ]
        elif type == 'classify':
            self.dataset_path = [
                'mit-bih-arrhythmia-database-1.0.0/'
            ]

    def load(self):
        loaded_dataset = {}
        for target_data in self.dataset_path:
            parsed_data_path = f'{BASE_PARSED_DB_PATH}{target_data}'
            file_list =  os.listdir(parsed_data_path)

            np_data = {}
            json_data = {}

            for data in file_list:
                data_file_name = f'{parsed_data_path}/{data}'
                if '.npy' in data:
                    file_type = data.split('.')[0].split('_')[-1]
                    np_data[file_type] = np.load(f'{data_file_name}')
                if 'annotation.json' in data:
                    if 'frame' in data:
                        with open(data_file_name) as frame_file_data:
                            json_data['frame'] = json.load(frame_file_data)
                    elif 'afib' in data:
                        with open(data_file_name) as afib_file_data:
                            json_data['afib'] = json.load(afib_file_data)
                    elif 'afl' in data:
                        with open(data_file_name) as afl_file_data:
                            json_data['afl'] = json.load(afl_file_data)

            pid_list = np.unique(np_data['pid'])

            record_raw_data = {}
            for record_id in pid_list:
                record = wfdb.rdrecord(f'{BASE_RAW_DATA_PATH}{target_data}/{record_id}')
                record_tags = [tag for tag in LEAD2_TAGS if tag in record.sig_name]
                if record_tags:
                    record_raw_data[record_id] = {}
                    record_raw_data[record_id]['fs'] = record.fs
                    record_raw_data[record_id]['ecg'] = record.p_signal[:, record.sig_name.index(record_tags[0])].tolist()

            loaded_dataset[target_data] = BaseDataModel(np_data=np_data, json_data=json_data, raw_data=record_raw_data)

        return loaded_dataset