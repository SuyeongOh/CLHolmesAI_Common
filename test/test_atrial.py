import config
from config import DURATION, FS, AF_THRESHOLD, BATCH_SIZE
from ecg.ecg_afib import DDNN

from scipy.signal import resample
import numpy as np


def run_afib(afib_data: dict, raw_signal: dict, raw_fs: dict):
    model = DDNN().ddnn()

    for pid in afib_data.keys():
        signal_split_10s = []
        raw_nparray = np.array(raw_signal[pid])
        afib_data_array = afib_data[pid]
        pid_fs = raw_fs[pid]
        resample_raw_signal = resample(raw_nparray, (int)(len(raw_nparray)/pid_fs * FS))
        afib_signal = np.zeros(len(resample_raw_signal))
        for record_afib_data in afib_data_array:
            record_afib_data['start_sample'] = (int)(record_afib_data['start_sample'] / pid_fs * FS)
            record_afib_data['end_sample'] = (int)(record_afib_data['end_sample'] / pid_fs * FS)
            afib_signal[record_afib_data['start_sample']:record_afib_data['end_sample']] = 1
        len_10s_sample = 250 * 10

        for idx in range(len(resample_raw_signal)//len_10s_sample):
            signal_10s = np.array(resample_raw_signal[idx*len_10s_sample : (idx+1)*len_10s_sample])
            signal_split_10s.append(signal_10s)

        if len(resample_raw_signal)%len_10s_sample != 0:
            signal_10s_last = np.array(
                resample_raw_signal[(idx+1)*len_10s_sample : ]).reshape(-1, 1)
            signal_10s_last = np.pad(signal_10s_last, ((0, 2500 - len(signal_10s_last)), (0, 0)), mode='constant',
                                     constant_values=0).flatten()
            signal_split_10s.append(signal_10s_last)

        ecg_pid_array = np.concatenate([arr[np.newaxis, :, np.newaxis] for arr in signal_split_10s], axis=0)

        confidence_AFL = model.predict(ecg_pid_array, batch_size=BATCH_SIZE)

        tmp_probability_AFL = confidence_AFL.copy()
        scaled_probability_AFL = np.where(confidence_AFL <= AF_THRESHOLD, 0.5 * confidence_AFL / AF_THRESHOLD,
                                          0.5 + 0.5 * (confidence_AFL - AF_THRESHOLD) / (1 - AF_THRESHOLD))

        tmp_probability_AFL[scaled_probability_AFL < 0.5] = 0  # AFIB
        tmp_probability_AFL[scaled_probability_AFL >= 0.5] = 1  # AFL

        #해당 10초구간이 AFIB인지 판별
        afib_signal_10s = np.zeros(len(tmp_probability_AFL))
        for i in range(len(afib_signal)//(FS * DURATION) + 1):
            if not i == (len(afib_signal)//(FS * DURATION) - 1):
                isAfib = np.isin(1, afib_signal[i*FS*DURATION:(i+1)*FS*DURATION])
            else:
                isAfib = np.isin(1, afib_signal[i*FS*DURATION:-1])

            if isAfib:
                afib_signal_10s[i] = 1
            else:
                afib_signal_10s[i] = 0

        t_count = np.count_nonzero(afib_signal_10s == 1)
        p_count = np.count_nonzero(tmp_probability_AFL == 0)
        #TP구간 갯수
        tp_count = np.sum((tmp_probability_AFL == 0) & (afib_signal_10s == 1))
        #에피소드는 추후 지원

        #sense
        afib_pid_sense = tp_count/t_count
        #ppv
        if not p_count:
            afib_pid_ppv = tp_count/p_count
        else:
            afib_pid_ppv = -1

        print(f'pid :: {pid}, true: {t_count}, positive: {p_count}, sense={afib_pid_sense}, ppv={afib_pid_ppv}')


def run_afl(afl_data: dict, raw_signal: dict, raw_fs: dict):
    model = DDNN().ddnn()

    for pid in afl_data.keys():
        signal_split_10s = []
        raw_nparray = np.array(raw_signal[pid])
        afl_data_array = afl_data[pid]
        pid_fs = raw_fs[pid]
        resample_raw_signal = resample(raw_nparray, (int)(len(raw_nparray)/pid_fs * FS))
        afl_signal = np.zeros(len(resample_raw_signal))
        for record_afl_data in afl_data_array:
            record_afl_data['start_sample'] = (int)(record_afl_data['start_sample'] / pid_fs * FS)
            record_afl_data['end_sample'] = (int)(record_afl_data['end_sample'] / pid_fs * FS)
            afl_signal[record_afl_data['start_sample']:record_afl_data['end_sample']] = 1
        len_10s_sample = 250 * 10

        for idx in range(len(resample_raw_signal)//len_10s_sample):
            signal_10s = np.array(resample_raw_signal[idx*len_10s_sample : (idx+1)*len_10s_sample])
            signal_split_10s.append(signal_10s)

        if len(resample_raw_signal)%len_10s_sample != 0:
            signal_10s_last = np.array(
                resample_raw_signal[(idx+1)*len_10s_sample : ]).reshape(-1, 1)
            signal_10s_last = np.pad(signal_10s_last, ((0, 2500 - len(signal_10s_last)), (0, 0)), mode='constant',
                                     constant_values=0).flatten()
            signal_split_10s.append(signal_10s_last)

        ecg_pid_array = np.concatenate([arr[np.newaxis, :, np.newaxis] for arr in signal_split_10s], axis=0)

        confidence_AFL = model.predict(ecg_pid_array, batch_size=BATCH_SIZE)

        tmp_probability_AFL = confidence_AFL.copy()
        scaled_probability_AFL = np.where(confidence_AFL <= AF_THRESHOLD, 0.5 * confidence_AFL / AF_THRESHOLD,
                                          0.5 + 0.5 * (confidence_AFL - AF_THRESHOLD) / (1 - AF_THRESHOLD))

        tmp_probability_AFL[scaled_probability_AFL < 0.5] = 0  # AFIB
        tmp_probability_AFL[scaled_probability_AFL >= 0.5] = 1  # AFL

        #해당 10초구간이 AFL인지 판별
        afl_signal_10s = np.zeros(len(tmp_probability_AFL))
        for i in range(len(afl_signal)//(FS * DURATION) + 1):
            if not i == (len(afl_signal)//(FS * DURATION) - 1):
                isAfl = np.isin(1, afl_signal[i*FS*DURATION:(i+1)*FS*DURATION])
            else:
                isAfl = np.isin(1, afl_signal[i*FS*DURATION:-1])

            if isAfl:
                afl_signal_10s[i] = 1
            else:
                afl_signal_10s[i] = 0

        t_count = np.count_nonzero(afl_signal_10s == 1)
        p_count = np.count_nonzero(tmp_probability_AFL == 1)
        #TP구간 갯수
        tp_count = np.sum((tmp_probability_AFL == 1) & (afl_signal_10s == 1))
        #에피소드는 추후 지원

        #sense
        afl_pid_sense = tp_count/t_count
        #ppv
        if not p_count:
            afl_pid_ppv = tp_count/p_count
        else:
            afl_pid_ppv = -1

        print(f'pid :: {pid}, true: {t_count}, positive: {p_count}, sense={afl_pid_sense}, ppv={afl_pid_ppv}')