import neurokit2 as nk
import numpy as np
from scipy.signal import resample, find_peaks

from ecg.ecg_delineate import ECGSegmentationArchitecture
from utils import ecg_utils
from config import FS, ORDER, FILTERS_WIDTH, HIGHCUT2, DURATION, BATCH_SIZE, SEC_TO_MS


#TEST_MODE : DNN / NK
TEST_MODE = 'DNN'
def run(np_data: dict, gt_data: dict, raw_signal: dict, raw_fs: dict):
    qrs_correct = {}

    sense_tag = 'sense'
    ppv_tag = 'ppv'

    # load model
    model = ECGSegmentationArchitecture().UNet1DPlusPlus()  # laod model architecture
    model.load_weights("weights/ecg_segmentation_weights.h5")  # load weights

    # pid별로 QRS
    pid_list = np.unique(np_data['pid'])

    hr_gt_list = []
    hr_pred_list = []
    for pid in pid_list:
        signal_split_10s = []
        qrs_correct[pid] = {}
        raw_nparray = np.array(raw_signal[pid])
        pid_fs = raw_fs[pid]
        resample_raw_signal = resample(raw_nparray, (int)(len(raw_nparray)/36 * 25))
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


        if TEST_MODE == 'NK':
            #Only use in Neurokit2(10sec list 2-dim signal to 1-dim)
            ecg_full = np.hstack(signal_split_10s)

            #Neurokit2 Find PID's R-peaks
            rpeaks = nk.ecg_findpeaks(ecg_full, sampling_rate=FS)
            signals, waves = nk.ecg_delineate(ecg_full, rpeaks=rpeaks, sampling_rate=FS)
            rPeak_list = rpeaks['ECG_R_Peaks']

        elif TEST_MODE == 'DNN':
            nparray_split_10s = np.stack(signal_split_10s, axis=0)

            all_ecg5s_1 = nparray_split_10s[:, :FS * DURATION // 2]
            all_ecg5s_2 = nparray_split_10s[:, FS * DURATION // 2:]
            all_ecg5s = np.concatenate((all_ecg5s_1, all_ecg5s_2), axis=0)
            del all_ecg5s_1, all_ecg5s_2

            segmentation_result_5s = model.predict(all_ecg5s, batch_size=BATCH_SIZE)  # (2N, 1250, 4)
            segmentation_result_5s = np.argmax(segmentation_result_5s, axis=-1)  # (2N, 1250)

            result_1 = segmentation_result_5s[:nparray_split_10s.shape[0]]  # (N, 1250)
            result_2 = segmentation_result_5s[nparray_split_10s.shape[0]:]  # (N, 1250)
            del segmentation_result_5s

            segmentation_result = np.concatenate((result_1, result_2), axis=1).astype(np.int8)  # (N, 2500)
            del result_1, result_2

            segmentation_result = segmentation_result.ravel()

            # QRS complex encoding = 2
            qrs_group_onset, qrs_group_offset = ecg_utils.extract_continuous_groups(segmentation_result, 2)

            # 10sec -> 1dim ECG data
            all_ecg5s = all_ecg5s.ravel()
            rPeak_list = ecg_utils.extract_rPeak(all_ecg5s, qrs_group_onset, qrs_group_offset)

            # rPeak sampling -> 250hz -> 360hz
            rPeak_list = np.array(rPeak_list)

        error_margin = 150
        # rPeak FS=250, label_data FS=360 (mit-bih기준 고정)
        # 150ms in 360hz 54 samples
        rPeak_gt = np.array(gt_data[pid])
        FS_gt = gt_data[f'{pid}_FS']
        rPeak_list_sec = rPeak_list / FS
        rPeak_gt_sec = rPeak_gt / FS_gt

        tp_list, fp_list = ecg_utils.find_rPeak_isClose(rPeak_list_sec, rPeak_gt_sec, error_margin / SEC_TO_MS)

        qrs_correct[pid]['tp_count'] = len(tp_list)
        qrs_correct[pid]['fn_count'] = len(fp_list)
        qrs_correct[pid][sense_tag] = len(tp_list) / len(rPeak_gt)
        qrs_correct[pid][ppv_tag] = len(tp_list) / len(rPeak_list)

        print(
            f'QRS Test Pid : {pid}, Sense={qrs_correct[pid][sense_tag]}, Ppv={qrs_correct[pid][ppv_tag]}, TP={len(tp_list)}, FP={len(fp_list)}')

        rri_gt = np.diff(rPeak_gt)
        rri_gt_avg = np.sum(rri_gt) / (pid_fs * len(rri_gt)) * SEC_TO_MS
        hr_gt = 60 * SEC_TO_MS / rri_gt_avg

        rri_list = np.diff(rPeak_list)
        rri_list_avg = np.sum(rri_list) / (FS * len(rri_list)) * SEC_TO_MS
        hr_pred = 60 * SEC_TO_MS / rri_list_avg

        hr_error = np.sqrt(np.abs(hr_gt - hr_pred) ** 2)
        print(f'HR Test Pid : {pid}, HR_DATASET={hr_gt}, HR_PRED={hr_pred}, RMSE={hr_error}')
        hr_gt_list.append(hr_gt)
        hr_pred_list.append(hr_pred)

    # HR RMS ERROR
    hr_rmse = np.sqrt(np.mean(np.abs(np.array(hr_gt_list) - np.array(hr_pred_list)) ** 2))
    print(f'HR RMSE = {hr_rmse}')
