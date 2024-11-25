import neurokit2 as nk
import numpy as np

from scipy.signal import resample, butter, medfilt, lfilter
from sympy import false

from scipy.interpolate import interp1d

from config import FS, ORDER, FILTERS_WIDTH, HIGHCUT2


def min_max_normalization(ecg10s):
    max = np.max(ecg10s)
    min = np.min(ecg10s)

    normalization_arr = (ecg10s - min) / (max - min)
    return normalization_arr

def zscore(ecg10s):
    mean = np.mean(ecg10s)
    std = np.std(ecg10s)

    ecg10s = (ecg10s - mean) / std
    return ecg10s

# 3. preprocessonECG
def getMedianFilters(duration):
    res = int(FS * duration)
    res += ((res % 2) - 1)  # needs to be an odd number
    return res

def LPF(data, highcut):  # cutoff 30 Hz
    nyq = 0.5 * FS  # 250 Hz / 2 = 125 Hz
    high = highcut / nyq  # 30 / 125 = 0.25 차단 주파수
    b, a = butter(ORDER, high, btype='lowpass')
    return lfilter(b, a, data)

def detrendonECG(ecg10s):
    mfa = np.zeros(len(FILTERS_WIDTH), dtype='int')

    for i, width in enumerate(FILTERS_WIDTH):
        mfa[i] = getMedianFilters(width)

    trend = ecg10s  # read orignal signal
    for mi in range(0, len(mfa)):
        trend = medfilt(trend, mfa[mi])      # finding trend
    return np.subtract(ecg10s, trend)

def denoiseAndNormalization(ecg10s):
    denoised_ecg = LPF(detrendonECG(ecg10s), HIGHCUT2)
    preprocessed_ecg10s = np.round(zscore(denoised_ecg), decimals=3)

    return preprocessed_ecg10s # 웹출력용 / 분석용


def extract_continuous_groups(arr, target_value):
    # 특정 값(target_value)이 연속되는 구간의 인덱스를 찾음
    is_target = (arr == target_value)
    diff = np.diff(is_target.astype(int))

    # 시작 인덱스 (onset)과 끝 인덱스 (offset)
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1

    # 시작이나 끝이 target_value로 연속될 경우 처리
    if is_target[0]:
        onsets = np.insert(onsets, 0, 0)
    if is_target[-1]:
        offsets = np.append(offsets, len(arr))

    return onsets, offsets

def extract_rPeak(arr, onsets, offsets):
    rPeak_list = []
    for onset, offset in zip(onsets, offsets):
        rPeak_idx = np.argmax(arr[onset:offset])
        rPeak_idx += onset
        rPeak_list.append(rPeak_idx)

    return rPeak_list

def find_rPeak_isClose(arr_predict, arr_gt, threshold):
    trueClose_list = []
    falseClose_list = []

    for predict in arr_predict:
        prev_tp_len = len(trueClose_list)
        for gt in arr_gt:
            if np.abs(predict - gt) <= threshold:
                trueClose_list.append(predict)
                break
        if len(trueClose_list) == prev_tp_len:
            falseClose_list.append(predict)

    return trueClose_list, falseClose_list


def resample_unequal(ts, fs_in, fs_out):
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
