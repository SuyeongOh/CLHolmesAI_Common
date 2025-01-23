import json
import os

import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import interp1d

from utils import data_utils

#p파 라벨은 p, rhythm symbol은 +
label_group_map = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', '.':'N',
                   'S':'S', 'A':'S', 'J':'S',  'a':'S',
                   'V': 'V', 'E': 'V',
                   'F':'F',
                   '/':'Q', 'f':'Q', 'Q':'Q',
                   'p':'p',
                   '+':'+'}


AFIB_LABEL = '(AFIB'
AFL_LABEL = '(AFL'

#https://github.com/hsd1503/PhysioNet : dataset parser 원본코드.
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
        dataset_name = path.split('/')[1]
        save_path = f'parsed_data/{dataset_name}/'
        # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5']
        #p-wave에선 ECG1 = MLII, ECG2 = V1
        valid_lead = ['MLII', 'ECG1']
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

        #Pwave dataset은 label의 확장자가 다름
        annot_footer = 'pwave'
        for file in os.listdir(path) :
            if file.endswith('.atr'):
                annot_footer = 'atr'
                break

        #record별 label 추출
        for record_name in all_record_name:
            #wfdb 이용 label extract
            try:
                tmp_ann_res = wfdb.rdann(path + '/' + record_name, annot_footer).__dict__
                tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
            except:
                print('read data failed')
                continue

            #tmp_data_res[0] : signal, lead별로 2차워 구성
            #tmp_data_res[1] : 기타 record 전체에 대한 dataset
            fs = tmp_data_res[1]['fs']
            siglen = tmp_data_res[1]['sig_len']

            #QRS Complex Parsing
            #all_frame_annotation : annotation에 대한 전체 정보
            try:
                annot_series = pd.Series(tmp_ann_res['symbol'], index=tmp_ann_res['sample'], name="annotations")
                annotations = annot_series.iloc[:].loc[annot_series.isin(label_group_map.keys())]
                frames_annotations_list = annotations.index.tolist()
                all_frame_annotation[record_name] = frames_annotations_list
                all_frame_annotation[f'{record_name}_FS'] = fs
            except Exception as e:
                print(f'file :: {record_name}, message :: {e}')

            #데이터는 beat label(nsvfq,,,) / rhythm label(afib, afl, normal ....)으로 나뉨
            rhythm_label_pid = [(i, s) for i, s in enumerate(tmp_ann_res['aux_note']) if s.strip()]

            #afl/afib data 파싱 과정. rhythmlabel의 위치를 파싱
            all_afib_data[record_name] = data_utils.rhythmLabelEpisodeFinder(
                AFIB_LABEL, record=record_name, rhythm_label_pid=rhythm_label_pid, all_frame_annotation=all_frame_annotation, siglen=siglen)
            all_afl_data[record_name] = data_utils.rhythmLabelEpisodeFinder(
                AFL_LABEL, record=record_name, rhythm_label_pid=rhythm_label_pid, all_frame_annotation=all_frame_annotation, siglen=siglen)

            ## total 1 second for each
            left_offset = int(1.0 * fs / 2)
            right_offset = int(fs) - int(1.0 * fs / 2)

            lead_in_data = tmp_data_res[1]['sig_name']
            my_lead_all = []

            # pid별 Dataset 정리 코드

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

        #numpy array 이용시 용이
        all_pid = np.array(all_pid)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        all_group = np.array(all_group)

        if 'atrial' in dataset_name:
            print()
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
