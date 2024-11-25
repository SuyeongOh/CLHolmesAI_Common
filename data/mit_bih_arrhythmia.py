import json
import os
from collections import Counter

import numpy as np
import wfdb
import pandas as pd

from scipy.interpolate import interp1d

label_group_map = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', '.':'N',
                   'S':'S', 'A':'S', 'J':'S',  'a':'S',
                   'V': 'V', 'E': 'V',
                   'F':'F',
                   '/':'Q', 'f':'Q', 'Q':'Q'}
#label_group_map = {'N':'N', 'V':'V', 'F':'F', 'S':'S',}
#NST와 mit-bih는 같은 형식으로 같은 parser 사용 가능
class MIT_BIH_ARRHYTMIA:
    def resample_unequal(self, ts, fs_in, fs_out):
        """
        interploration
        """
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        else:
            x_old = np.linspace(0, 1, num=fs_in, endpoint=True)
            x_new = np.linspace(0, 1, num=fs_out, endpoint=True)
            y_old = ts
            f = interp1d(x_old, y_old, kind='linear')
            y_new = f(x_new)
            return y_new

#data 폴더에 datset을 넣어주세요
    def run(self, type: str):
        if type == '':
            return
        if type == 'arrhythmia':
            path = 'data/mit-bih-arrhythmia-database-1.0.0'
            record_file = 'RECORDS_NOPACE'
        elif type == 'stress':
            path = 'data/mit-bih-noise-stress-test-database-1.0.0'
            record_file = 'RECORDS'
        elif type == 'afib':
            path = 'data/mit-bih-atrial-fibrillation-database-1.0.0/files/'
            record_file = 'RECORDS'
        save_path = f'data/parsing/{type}/'
        # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5']
        valid_lead = ['MLII']
        fs_out = 250
        test_ratio = 0.2

        train_ind = []
        test_ind = []
        all_pid = []
        all_data = []
        all_label = []
        all_group = []
        all_frame_annotation = {}
        with open(os.path.join(path, record_file), 'r') as fin:
            all_record_name = fin.read().strip().split('\n')
        # test_pid = random.choices(all_record_name, k=int(len(all_record_name) * test_ratio))
        # train_pid = list(set(all_record_name) - set(test_pid))

        for record_name in all_record_name:
            try:
                tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
                tmp_data_res = wfdb.rdsamp(path + '/' + record_name)

            except:
                print('read data failed')
                continue
            fs = tmp_data_res[1]['fs']

            #QRS Complex Parsing
            try:
                annot_series = pd.Series(tmp_ann_res['symbol'], index=tmp_ann_res['sample'], name="annotations")
                qrs_annotations = annot_series.iloc[:].loc[annot_series.isin(label_group_map.keys())]
                frames_annotations_list = qrs_annotations.index.tolist()
                all_frame_annotation[record_name] = frames_annotations_list
                all_frame_annotation[f'{record_name}_FS'] = fs
            except Exception as e:
                print(f'file :: {record_name}, message :: {e}')

            rhythm_label_pid = [(i, s) for i, s in enumerate(tmp_ann_res['aux_note']) if s.strip()]

            ## total 1 second for each
            left_offset = int(1.0 * fs / 2)
            right_offset = int(fs) - int(1.0 * fs / 2)

            lead_in_data = tmp_data_res[1]['sig_name']
            my_lead_all = []
            for tmp_lead in valid_lead:
                if tmp_lead in lead_in_data:
                    my_lead_all.append(tmp_lead)
            if len(my_lead_all) != 0:
                for my_lead in my_lead_all:
                    channel = lead_in_data.index(my_lead)
                    tmp_data = tmp_data_res[0][:, channel]

                    idx_list = list(tmp_ann_res['sample'])
                    label_list = tmp_ann_res['symbol']
                    for i in range(len(label_list)):
                        s = label_list[i]
                        if s in label_group_map.keys():
                            idx_start = idx_list[i] - left_offset
                            idx_end = idx_list[i] + right_offset
                            if idx_start < 0 or idx_end > len(tmp_data):
                                continue
                            else:
                                all_pid.append(record_name)
                                resample_data = self.resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out)
                                all_data.append(resample_data)
                                all_label.append(s)
                                all_group.append(label_group_map[s])
                                # if record_name in train_pid:
                                #     train_ind.append(True)
                                #     test_ind.append(False)
                                # else:
                                #     train_ind.append(False)
                                #     test_ind.append(True)
                        elif s=='~' :
                            idx_start = idx_list[i] - left_offset
                            idx_end = idx_list[i] + right_offset
                            if idx_start < 0 or idx_end > len(tmp_data):
                                continue
                            else:
                                all_pid.append(record_name)
                                resample_data = self.resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out)
                                all_data.append(resample_data)
                                all_label.append('U')
                                all_group.append('U')
                        elif s== '[' or s==']':
                            continue
                        else :
                            continue
                            # idx_start = idx_list[i] - left_offset
                            # idx_end = idx_list[i] + right_offset
                            # if idx_start < 0 or idx_end > len(tmp_data):
                            #     continue
                            # else:
                            #     all_pid.append(record_name)
                            #     resample_data = self.resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out)
                            #     all_data.append(resample_data)
                            #     all_label.append('O')
                            #     all_group.append('O')

                    print('record_name:{}, lead:{}, fs:{}, cumcount: {}'.format(record_name, my_lead, fs, len(all_pid)))
            else:
                print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
                continue

        all_pid = np.array(all_pid)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        all_group = np.array(all_group)
        # train_ind = np.array(train_ind)
        # test_ind = np.array(test_ind)
        print(all_data.shape)
        print(Counter(all_label))
        print(Counter(all_group))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, f'mitdb_{type}_data.npy'), all_data)
        np.save(os.path.join(save_path, f'mitdb_{type}_label.npy'), all_label)
        np.save(os.path.join(save_path, f'mitdb_{type}_group.npy'), all_group)
        np.save(os.path.join(save_path, f'mitdb_{type}_pid.npy'), all_pid)
        # np.save(os.path.join(save_path, f'mitdb_{type}_train_ind.npy'), train_ind)
        # np.save(os.path.join(save_path, f'mitdb_{type}_test_ind.npy'), test_ind)

        with open(os.path.join(save_path, f'mitdb_{type}_frame_annotation.json'), 'w') as anoot_file:
            json.dump(all_frame_annotation, anoot_file)