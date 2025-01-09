import json
import os

import numpy as np
import wfdb

from test.dataloader import BASE_PARSED_DB_PATH, BASE_RAW_DATA_PATH, LEAD2_TAGS


class MitArrhythmiaLoader:
    dataset_path = 'mit-bih-arrhythmia-database-1.0.0/'
    parsed_data_path = BASE_PARSED_DB_PATH + dataset_path

    def load(self):
        file_list =  os.listdir(self.parsed_data_path)

        np_data = {}
        json_data = {}

        for data in file_list:
            if '.npy' in data:
                file_type = data.split('.')[0].split('_')[-1]
                np_data[file_type] = np.load(f'{self.parsed_data_path}{data}')
            if 'annotation.json' in data:
                if 'frame' in data:
                    with open(self.parsed_data_path + data) as frame_file_data:
                        json_data['frame'] = json.load(frame_file_data)
                elif 'afib' in data:
                    with open(self.parsed_data_path + data) as afib_file_data:
                        json_data['afib'] = json.load(afib_file_data)
                elif 'afl' in data:
                    with open(self.parsed_data_path + data) as afl_file_data:
                        json_data['afl'] = json.load(afl_file_data)

        pid_list = np.unique(np_data['pid'])

        record_raw_data = {}
        for record_id in pid_list:
            record = wfdb.rdrecord(f'{BASE_RAW_DATA_PATH}{self.dataset_path}/{record_id}')
            record_tags = [tag for tag in LEAD2_TAGS if tag in record.sig_name]
            if record_tags:
                record_raw_data[record_id] = {}
                record_raw_data[record_id]['fs'] = record.fs
                record_raw_data[record_id]['ecg'] = record.p_signal[:, record.sig_name.index(record_tags[0])].tolist()


        return np_data, json_data, record_raw_data