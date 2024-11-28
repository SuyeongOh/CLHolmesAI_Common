import json
import os
import time

import numpy as np
import wfdb

from test import test_delineate, test_beat_analysis, test_atrial

ROOT_DATA_PATH = 'data/data'
BASE_DATA_PATH = 'data/data/parsing'
DATA_LABEL_ARRHYTHMIA = 'arrhythmia'
DATA_LABEL_STRESS = 'stress'

TAG_LEAD2 = 'MLII'

# slice 1s waveform for
if __name__ == "__main__":
    # load dataset
    TARGET_DATA = DATA_LABEL_STRESS
    arrhythmia_path = f'{BASE_DATA_PATH}/{TARGET_DATA}/'
    arrhythmia_data = os.listdir(arrhythmia_path)
    np_data = {}
    json_data = {}

    for data in arrhythmia_data:
        if '.npy' in data:
            file_type = data.split('.')[0].split('_')[-1]
            np_data[file_type] = np.load(arrhythmia_path + data)
        if 'annotation.json' in data:
            with open(arrhythmia_path + data) as file_data:
                json_data = json.load(file_data)
    resample_signal = []
    result_class = []

    raw_ecg_path = f'{BASE_DATA_PATH}/{TARGET_DATA}/{TARGET_DATA}_raw_ecg.json'

    pid_list = np.unique(np_data['pid'])
    if TARGET_DATA == DATA_LABEL_ARRHYTHMIA:
        raw_dataset_path = f'{ROOT_DATA_PATH}/mit-bih-arrhythmia-database-1.0.0'
    if TARGET_DATA == DATA_LABEL_STRESS:
        raw_dataset_path = f'{ROOT_DATA_PATH}/mit-bih-noise-stress-test-database-1.0.0'

    raw_record = {}
    raw_record_fs = {}
    for record_id in pid_list:
        record = wfdb.rdrecord(f'{raw_dataset_path}/{record_id}')
        if TAG_LEAD2 in record.sig_name:
            raw_record_fs[record_id] = record.fs
            raw_record[record_id] = record.p_signal[:, record.sig_name.index(TAG_LEAD2)].tolist()

    start_time = time.time()
    #Test 진행 코드
    test_beat_analysis.run(np_data=np_data)
    test_delineate.run(np_data=np_data, gt_data=json_data, raw_signal=raw_record, raw_fs=raw_record_fs)
    test_atrial.run(np_data=np_data, gt_data=json_data, raw_signal=raw_record, raw_fs=raw_record_fs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"소요 시간: {elapsed_time:.6f}초")
