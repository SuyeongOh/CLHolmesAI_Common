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


AFIB_LABEL = '(AFIB'
AFL_LABEL = '(AFL'
AFL_Dataset = ['202', '203', '222']
AFIB_Dataset = ['201', '202', '203', '210', '219', '221', '222']
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
    def run(self, path: str, record_file: str):
        if path == '':
            return

        if not os.path.exists(path):
            return
        dataset_name = path.split('/')[0]
        save_path = f'parsed_data/{dataset_name}/'
        # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5']
        valid_lead = ['MLII']
        fs_out = 250

        all_pid = []
        all_data = []
        all_label = []
        all_group = []
        all_frame_annotation = {}
        all_afib_data = {}
        all_afl_data = {}
        with open(os.path.join(path, record_file), 'r') as fin:
            all_record_name = fin.read().strip().split('\n')

        for record_name in all_record_name:
            try:
                tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
                if not tmp_ann_res:
                    tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'pwave')
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


            if record_name in AFIB_Dataset:
                rhythm_label_pid = [(i, s) for i, s in enumerate(tmp_ann_res['aux_note']) if s.strip()]
                all_afib_data[record_name] = {}
                start_afib_flag = False
                start_afib_idx = 0
                record_afib_info = []
                for i, label in rhythm_label_pid:
                    if AFIB_LABEL in label :
                        start_afib_flag = True
                        start_afib_idx = i
                        continue
                    if start_afib_flag:
                        try:
                            afib_info = {}
                            afib_info['start_idx'] = start_afib_idx
                            afib_info['start_sample'] = all_frame_annotation[record_name][start_afib_idx]
                            afib_info['end_idx'] = i
                            afib_info['end_sample'] = all_frame_annotation[record_name][i]
                            record_afib_info.append(afib_info)
                            start_afib_flag = False
                        except Exception as e:
                            continue

                if start_afib_flag:
                    afib_info = {}
                    afib_info['start_idx'] = start_afib_idx
                    afib_info['start_sample'] = all_frame_annotation[record_name][start_afib_idx]
                    afib_info['end_idx'] = -1
                    afib_info['end_sample'] = 650000
                    record_afib_info.append(afib_info)
                all_afib_data[record_name] = record_afib_info

            if record_name in AFL_Dataset:
                rhythm_label_pid = [(i, s) for i, s in enumerate(tmp_ann_res['aux_note']) if s.strip()]
                all_afl_data[record_name] = {}
                start_afl_flag = False
                start_afl_idx = 0
                record_afl_info = []
                for i, label in rhythm_label_pid:
                    if AFL_LABEL in label :
                        start_afl_flag = True
                        start_afl_idx = i
                        continue
                    if start_afl_flag:
                        try:
                            afl_info = {}
                            afl_info['start_idx'] = start_afl_idx
                            afl_info['start_sample'] = all_frame_annotation[record_name][start_afl_idx]
                            afl_info['end_idx'] = i
                            afl_info['end_sample'] = all_frame_annotation[record_name][i]
                            record_afl_info.append(afl_info)
                            start_afl_flag = False
                        except Exception as e:
                            continue

                if start_afl_flag:
                    afl_info = {}
                    afl_info['start_idx'] = start_afl_idx
                    afl_info['start_sample'] = all_frame_annotation[record_name][start_afl_idx]
                    afl_info['end_idx'] = -1
                    afl_info['end_sample'] = 650000
                    record_afl_info.append(afl_info)
                all_afl_data[record_name] = record_afl_info

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

                    print('record_name:{}, lead:{}, fs:{}, cumcount: {}'.format(record_name, my_lead, fs, len(all_pid)))
            else:
                print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
                continue

        all_pid = np.array(all_pid)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        all_group = np.array(all_group)
        print(all_data.shape)
        print(Counter(all_label))
        print(Counter(all_group))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, f'data.npy'), all_data)
        np.save(os.path.join(save_path, f'label.npy'), all_label)
        np.save(os.path.join(save_path, f'group.npy'), all_group)
        np.save(os.path.join(save_path, f'pid.npy'), all_pid)

        with open(os.path.join(save_path, f'afib_annotation.json'), 'w') as afib_annot_file:
            json.dump(all_afib_data, afib_annot_file)

        with open(os.path.join(save_path, f'afl_annotation.json'), 'w') as afl_annot_file:
            json.dump(all_afl_data, afl_annot_file)

        with open(os.path.join(save_path, f'frame_annotation.json'), 'w') as anoot_file:
            json.dump(all_frame_annotation, anoot_file)
