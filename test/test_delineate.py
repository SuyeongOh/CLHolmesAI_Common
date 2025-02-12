import neurokit2 as nk
import numpy as np
from scipy.signal import resample

from config import  FS, DURATION, BATCH_SIZE, SEC_TO_MS
#from ecg.ecg_delineate import ECGSegmentationArchitecture
from scipy.signal import butter, lfilter, filtfilt, firwin, lfiltic, resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vg_beat_detectors import FastNVG, FastWHVG

from ecg.ecg_delineate import ECGSegmentation
from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.datamodel.mitdb_model import MitdbDataModel
from utils import ecg_utils, log_utils

#TEST_MODE : DNN / NK / NVG
TEST_MODE = 'NVG'

logger = log_utils.getCustomLogger(__name__)

class TestDeliniate:
    #model = ECGSegmentationArchitecture().UNet1DPlusPlus()

    def run(self):
        test_dataset = MitArrhythmiaLoader('delineate').load()
        ecg_segmentator = ECGSegmentation()
        #self.model.load_weights("weights/ecg_segmentation_weights.h5")  # load weights

        for dataset_name in test_dataset.keys():
            logger.info(f'Target Dataset {dataset_name}')
            dataModel = MitdbDataModel(test_dataset[dataset_name])
            qrs_correct = {}
            hr_gt_list = []
            hr_pred_list = []

            for pid in dataModel.getPidList():
                ecg_signal_pid = dataModel.getPidSignal(pid)
                ecg_fs = dataModel.getPidFs(pid)
                signal_split_10s = self.create_input(ecg_signal_pid, ecg_fs)
                #P 찾는것으로 추가 개선 예정
                rPeak_list = self.find_rPeak(signal_split_10s)
                segmentation_result = ecg_segmentator.run(signal_split_10s)
                qrs_correct[pid], hr_gt_pid, hr_pred_pid = self.calculate_Accuracy(pid, rPeak_list, dataModel.getPidRpeak(pid), ecg_fs)
                hr_gt_list.append(hr_gt_pid)
                hr_pred_list.append(hr_pred_pid)

            # HR RMS ERROR
            hr_rmse = np.sqrt(np.mean(np.abs(np.array(hr_gt_list) - np.array(hr_pred_list)) ** 2))
            logger.info(f'HR RMSE = {hr_rmse}')


    def create_input(self, ecg_signal, ecg_fs):
        signal_split_10s = []
        resample_ecg_signal = resample(ecg_signal, int(len(ecg_signal) / ecg_fs * FS))
        len_10s_sample = 250 * 10

        for idx in range(len(resample_ecg_signal) // len_10s_sample):
            signal_10s = np.array(resample_ecg_signal[idx * len_10s_sample: (idx + 1) * len_10s_sample])
            signal_split_10s.append(signal_10s)

        if len(resample_ecg_signal) % len_10s_sample != 0:
            signal_10s_last = np.array(
                resample_ecg_signal[(idx + 1) * len_10s_sample:]).reshape(-1, 1)
            signal_10s_last = np.pad(signal_10s_last, ((0, 2500 - len(signal_10s_last)), (0, 0)), mode='constant',
                                     constant_values=0).flatten()
            signal_split_10s.append(signal_10s_last)

        return signal_split_10s


    def find_rPeak(self, input_signal):
        if TEST_MODE == 'NK':
            # Only use in Neurokit2(10sec list 2-dim signal to 1-dim)
            ecg_full = np.hstack(input_signal)

            # Neurokit2 Find PID's R-peaks
            rpeaks = nk.ecg_findpeaks(ecg_full, sampling_rate=FS)
            signals, waves = nk.ecg_delineate(ecg_full, rpeaks=rpeaks, sampling_rate=FS)
            rPeak_list = rpeaks['ECG_R_Peaks']

        if TEST_MODE == 'NVG':
            detector = FastNVG(sampling_frequency=250)

            ecg_full = np.hstack(input_signal)

            rpeaks = detector.find_peaks(ecg_full)
            signals, waves = nk.ecg_delineate(ecg_full, rpeaks=rpeaks, sampling_rate=FS)
            rPeak_list = rpeaks

        # elif TEST_MODE == 'DNN':
        #     nparray_split_10s = np.stack(input_signal, axis=0)
        #
        #     all_ecg5s_1 = nparray_split_10s[:, :FS * DURATION // 2]
        #     all_ecg5s_2 = nparray_split_10s[:, FS * DURATION // 2:]
        #     all_ecg5s = np.concatenate((all_ecg5s_1, all_ecg5s_2), axis=0)
        #     del all_ecg5s_1, all_ecg5s_2
        #
        #     segmentation_result_5s = self.model.predict(all_ecg5s, batch_size=BATCH_SIZE)  # (2N, 1250, 4)
        #     segmentation_result_5s = np.argmax(segmentation_result_5s, axis=-1)  # (2N, 1250)
        #
        #     result_1 = segmentation_result_5s[:nparray_split_10s.shape[0]]  # (N, 1250)
        #     result_2 = segmentation_result_5s[nparray_split_10s.shape[0]:]  # (N, 1250)
        #     del segmentation_result_5s
        #
        #     segmentation_result = np.concatenate((result_1, result_2), axis=1).astype(np.int8)  # (N, 2500)
        #     del result_1, result_2
        #
        #     segmentation_result = segmentation_result.ravel()
        #
        #     # QRS complex encoding = 2
        #     qrs_group_onset, qrs_group_offset = ecg_utils.extract_continuous_groups(segmentation_result, 2)
        #
        #     # 10sec -> 1dim ECG data
        #     all_ecg5s = all_ecg5s.ravel()
        #     rPeak_list = ecg_utils.extract_rPeak(all_ecg5s, qrs_group_onset, qrs_group_offset)
        #
        #     # rPeak sampling -> 250hz -> 360hz
        #     rPeak_list = np.array(rPeak_list)
        else:
            return []

        return rPeak_list

    def calculate_Accuracy(self, pid, rPeak_list, gt_data, ecg_fs):
        error_margin = 150
        # rPeak FS=250, label_data FS=360 (mit-bih기준 고정)
        # 150ms in 360hz 54 samples

        acc = {}
        rPeak_gt = np.array(gt_data)
        rPeak_list_sec = rPeak_list / FS
        rPeak_gt_sec = rPeak_gt / ecg_fs

        tp_list, fp_list = ecg_utils.find_rPeak_isClose(rPeak_list_sec, rPeak_gt_sec, error_margin / SEC_TO_MS)

        acc['tp_count'] = len(tp_list)
        acc['fn_count'] = len(fp_list)
        acc['sense'] = len(tp_list) / len(rPeak_gt)
        acc['ppv'] = len(tp_list) / len(rPeak_list)

        sense = acc['sense']
        ppv = acc['ppv']
        logger.info(
            f'QRS Test Pid : {pid}, Sense={sense}, Ppv={ppv}, TP={len(tp_list)}, FP={len(fp_list)}')

        rri_gt = np.diff(rPeak_gt)
        rri_gt_avg = np.sum(rri_gt) / (ecg_fs * len(rri_gt)) * SEC_TO_MS
        hr_gt = 60 * SEC_TO_MS / rri_gt_avg

        rri_list = np.diff(rPeak_list)
        rri_list_avg = np.sum(rri_list) / (FS * len(rri_list)) * SEC_TO_MS
        hr_pred = 60 * SEC_TO_MS / rri_list_avg

        hr_error = np.sqrt(np.abs(hr_gt - hr_pred) ** 2)
        logger.info(f'HR Test Pid : {pid}, HR_DATASET={hr_gt}, HR_PRED={hr_pred}, RMSE={hr_error}')

        return acc, hr_gt, hr_pred
