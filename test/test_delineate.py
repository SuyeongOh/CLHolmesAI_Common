import os

import gdown
import neurokit2 as nk
import numpy as np
from scipy.signal import resample

from config import FS, DURATION, BATCH_SIZE, SEC_TO_MS
from ecg.ecg_delineate import ECGSegmentationArchitecture, UNetQRS, DataPostprocessor
from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.datamodel.BaseDataModel import BaseDataModel
from test.datamodel.mitdb_model import MitdbDataModel
from utils import ecg_utils, log_utils

import torch
import torch.nn as nn

from stockwell import st
#TEST_MODE : DNN / NK / DY
TEST_MODE = 'DY'

logger = log_utils.getCustomLogger(__name__)

class TestDeliniate:
    def __init__(self):
        if TEST_MODE == "DNN":
            self.model = ECGSegmentationArchitecture().UNet1DPlusPlus()
        elif TEST_MODE == "DY":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = UNetQRS(num_classes=4) # NONE P QRS T
        else:
            self.model = None

    def run(self):
        test_dataset = MitArrhythmiaLoader('delineate').load()
        if TEST_MODE == 'DNN':
            self.model.load_weights("weights/ecg_segmentation_weights.h5")  # load weights
        elif TEST_MODE == 'DY':
            if not os.path.exists("weights/best_binary_segmentation_model.pth"):
                print("Download best binary segmentation model...")
                url = "https://drive.google.com/uc?id=18PW8OdwSPu08_JgKtFrDojoP2c-v4A6t"
                gdown.download(url,"weights/best_binary_segmentation_model.pth", quiet=False)
            self.model.load_state_dict(torch.load("weights/best_binary_segmentation_model.pth"))
            self.model.to(self.device)
            self.model.eval()



        for dataset_name in test_dataset.keys():
            logger.info(f'Target Dataset {dataset_name}')
            dataModel = MitdbDataModel(test_dataset[dataset_name])
            qrs_correct = {}
            hr_gt_list = []
            hr_pred_list = []

            for pid in dataModel.getPidList():
                ecg_signal_pid = dataModel.getPidSignal(pid) # pid별 ECG 신호
                ecg_fs = dataModel.getPidFs(pid)
                signal_split_10s = self.create_input(ecg_signal_pid, ecg_fs)
                #P 찾는것으로 추가 개선 예정
                rPeak_list = self.find_rPeak(signal_split_10s)
                qrs_correct[pid], hr_gt_pid, hr_pred_pid = self.calculate_Accuracy(pid, rPeak_list, dataModel.getPidRpeak(pid), ecg_fs)
                hr_gt_list.append(hr_gt_pid)
                hr_pred_list.append(hr_pred_pid)

            # HR RMS ERROR
            hr_rmse = np.sqrt(np.mean(np.abs(np.array(hr_gt_list) - np.array(hr_pred_list)) ** 2))
            logger.info(f'HR RMSE = {hr_rmse}')


    def create_input(self, ecg_signal, ecg_fs):
        signal_split_10s = []
        if TEST_MODE == "DY":
            ecg_signal = ecg_utils.LPF_for_delinate_DY(ecg_signal, 30, ecg_fs)

        resample_ecg_signal = resample(ecg_signal, int(len(ecg_signal) / ecg_fs * FS))

        if TEST_MODE == "DY":
            resample_ecg_signal = ecg_utils.min_max_normalization(resample_ecg_signal)

        len_10s_sample = FS * 10

        for idx in range(len(resample_ecg_signal) // len_10s_sample):
            signal_10s = np.array(resample_ecg_signal[idx * len_10s_sample: (idx + 1) * len_10s_sample])
            signal_split_10s.append(signal_10s)

        if len(resample_ecg_signal) % len_10s_sample != 0:
            signal_10s_last = np.array(
                resample_ecg_signal[(idx + 1) * len_10s_sample:]).reshape(-1, 1)
            signal_10s_last = np.pad(signal_10s_last, ((0, len_10s_sample - len(signal_10s_last)), (0, 0)), mode='constant',
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

        elif TEST_MODE == 'DNN':
            nparray_split_10s = np.stack(input_signal, axis=0)

            all_ecg5s_1 = nparray_split_10s[:, :FS * DURATION // 2]
            all_ecg5s_2 = nparray_split_10s[:, FS * DURATION // 2:]
            all_ecg5s = np.concatenate((all_ecg5s_1, all_ecg5s_2), axis=0)
            del all_ecg5s_1, all_ecg5s_2

            segmentation_result_5s = self.model.predict(all_ecg5s, batch_size=BATCH_SIZE)  # (2N, 1250, 4)
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
        elif TEST_MODE == 'DY':
            nparray_split_10s = np.stack(input_signal, axis=0)
            # resampling 250Hz -> 100Hz
            st_rst = np.array([self.preprocess_st(signal) for signal in nparray_split_10s])
            # TODO : NEED TO MODIFY BATCH SIZE(16) TO FIT GPU MEMORY SIZE
            SPLIT_SIZE = len(st_rst) / 32
            batch_np_array = np.array_split(st_rst, SPLIT_SIZE)
            result_class_list = []
            with torch.no_grad():
                for batch in batch_np_array:
                    tensor_batch = torch.tensor(batch, dtype=torch.float32).cuda()
                    outputs = self.model(tensor_batch)
                    result_class_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            result_class = np.array(result_class_list).reshape(-1)
            qrs_group_onset, qrs_group_offset = ecg_utils.extract_continuous_groups(result_class, 2)
            rPeak_list = ecg_utils.extract_rPeak(nparray_split_10s.reshape(-1), qrs_group_onset, qrs_group_offset)
            rPeak_list = np.array(rPeak_list)
        else:
            return []

        return rPeak_list

    def stockwell_transform(self, signal, fmin, fmax, signal_length):
        """
        Stockwell Transform 수행 함수.

        :param signal: 입력 신호
        :param fmin: 최소 주파수
        :param fmax: 최대 주파수
        :param signal_length: 변환에 사용될 신호 길이
        :return: Stockwell 변환 결과 (복소수 배열)
        """
        df = 1.0 / signal_length
        fmin_samples = int(fmin / df)
        fmax_samples = int(fmax / df)
        return st.st(signal, fmin_samples, fmax_samples)


    def preprocess_st(self,signal):
        trans_signal = self.stockwell_transform(signal, 0, 15, 10)
        real_part = np.real(trans_signal)
        imag_part = np.imag(trans_signal)
        return np.stack((real_part, imag_part), axis=0)

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
