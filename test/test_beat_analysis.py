import neurokit2 as nk
import numpy as np
from ecg.ecg_beat_classify import BeatClassifierArchitecture
from scipy.signal import resample

from config import BATCH_SIZE
from utils.ecg_utils import min_max_normalization

resample_signal = []
result_class = []



def run(np_data: dict):
    # pid별로 QRS
    pid_list = np.unique(np_data['pid'])
    pid_beat_ppv = {}
    pid_beat_sense = {}
    pid_beat_fn = {}
    beat_model = BeatClassifierArchitecture().model
    beat_model.load_weights("weights/heartbeat_classifier_weights.h5")

    for pid in pid_list:
        pid_idxs = np.where(np_data['pid'] == pid)[0]
        onset = pid_idxs[0]
        offset = pid_idxs[-1]

        target_data = np_data['data'][onset:offset]
        target_label = np_data['group'][onset:offset]
        for beat in target_data:
            resample_beat = np.array(resample(beat, 125))
            preprocessed_beat = min_max_normalization(nk.signal_detrend(resample_beat))
            resample_signal.append(preprocessed_beat)


        resample_np_array = np.concatenate([arr[np.newaxis, :, np.newaxis] for arr in resample_signal], axis=0)

        # compare result_class/np_data['group']
        # a number of F = (V + A) - for holmesai
        # original map -> beat_class_map = {0: 'N', 1: 'S', 2: 'V', 3: 'A'}
        beat_class_map = {0: 'N', 1: 'S', 2: 'V', 3: 'A'}
        beat_class_correct = {'N': [], 'S': [], 'V': []}
        beat_class_correct_label = {}
        beat_class_incorrect = {'N_V': [], 'N_F': [], 'V_N': [], 'V_F': [], 'S_N': [], 'S_V': []}
        beat_class_incorrect_pid = {}
        beat_class_fp = {'N': [], 'V': [], 'S': []}
        beat_class_tn = {'V': [], 'S': []}
        beat_class_per_record = {}
        # Beat Analysis Model

        result_class = beat_model.predict(resample_np_array, batch_size=BATCH_SIZE)
        result_class = np.argmax(result_class, axis=1)

        for cls, label in zip(result_class, target_label):
            if beat_class_map[cls] == label:
                beat_class_correct[label].append(label)
            else:
                beat_class_fp[beat_class_map[cls]].append(beat_class_map[cls])
                if label == 'V' or label == 'S':
                    beat_class_tn[label].append(label)
                try:
                    beat_class_incorrect[f'{label}_{beat_class_map[cls]}'].append(cls)
                except:
                    print(f'create new dict {label}_{beat_class_map[cls]}')
                    beat_class_incorrect[f'{label}_{beat_class_map[cls]}'] = []
                    beat_class_incorrect[f'{label}_{beat_class_map[cls]}'].append(cls)
        total_class_count = {'N': 0, 'V': 0, 'S': 0}
        total_class_count['N'] = np.sum(np_data['group'] == 'N')
        total_class_count['V'] = np.sum(np_data['group'] == 'V')
        total_class_count['S'] = np.sum(np_data['group'] == 'S')

        total_analysis_class_count = {'N': 0, 'V': 0, 'S': 0}
        for cls in result_class:
            if cls == 0:
                total_analysis_class_count['N'] += 1
            elif cls == 2:
                total_analysis_class_count['V'] += 1
            elif cls == 1 or cls == 3:
                total_analysis_class_count['S'] += 1

        if total_class_count['N'] != 0:
            n_sense = len(beat_class_correct['N']) / total_class_count['N']
        else:
            n_sense = 0
        if total_class_count['V'] != 0:
            v_sense = len(beat_class_correct['V']) / total_class_count['V']
        else:
            v_sense = 0
        if total_class_count['S'] != 0:
            s_sense = len(beat_class_correct['S']) / total_class_count['S']
        else:
            s_sense = 0


        if total_analysis_class_count['N'] != 0:
            n_ppv = len(beat_class_correct['N']) / total_analysis_class_count['N']
        else:
            n_ppv = 0
        if total_analysis_class_count['V'] != 0:
            v_ppv = len(beat_class_correct['V']) / total_analysis_class_count['V']
        else:
            v_ppv = 0
        if total_analysis_class_count['S'] != 0:
            s_ppv = len(beat_class_correct['S']) / total_analysis_class_count['S']
        else:
            s_ppv = 0

        v_fn = len(beat_class_fp['V']) / (len(beat_class_tn['V']) + len(beat_class_fp['V']))
        s_fn = len(beat_class_fp['S']) / (len(beat_class_tn['S']) + len(beat_class_fp['S']))

        print(f'pid_{pid}  sensitivity : N={n_sense}, V={v_sense}, S={s_sense}')
        print(f'pid_{pid}  ppv : N={n_ppv}, V={v_ppv}, S={s_ppv}')

        # # Record별 출력
        # for cls in beat_class_correct:
        #     pid_list = beat_class_correct[cls]
        #     pid_keys = np.unique(beat_class_correct[cls])
        #     for pid in pid_keys:
        #         pid_count = pid_list.count(pid)
        #         print(f'correct - class, pid : ({cls}, {pid}, {pid_count})')
        #
        # for cls in beat_class_incorrect:
        #     pid_list = beat_class_incorrect[cls]
        #     pid_keys = np.unique(beat_class_incorrect[cls])
        #     for pid in pid_keys:
        #         pid_count = pid_list.count(pid)
        #         print(f'incorrect - class, pid : ({cls}, {pid}, {pid_count})')

        pid_beat_ppv[pid] = v_ppv
        pid_beat_sense[pid] = v_sense
        pid_beat_fn[pid] = v_fn

        resample_signal.clear()