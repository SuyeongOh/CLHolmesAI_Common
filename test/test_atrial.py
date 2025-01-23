import logging
import torch
from sklearn.metrics import f1_score

from config import DURATION, FS, AF_THRESHOLD, BATCH_SIZE
import ecg.ecg_atrial as safer
from scipy.signal import resample
import numpy as np
from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.datamodel.mitdb_model import MitdbDataModel
from test.datamodel.accuracy_datamodel import AccuracyDataModel
from utils import log_utils

TAG_AFIB = 'afib'
TAG_AFL = 'afl'
weights_path = './weights/atrial_weights.pth'

logger = log_utils.getCustomLogger(__name__)

class TestAtrial:
    def __init__(self):
        self.atrial_model = safer.SigFormer_segmentation_rhythm()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Finetuning 모델 weight load
        self.atrial_model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        self.atrial_model.eval()  # 모델을 평가 모드로 설정
        self.atrial_model.to(self.device)

    def run(self):
        logger.info(f'Start Test Atrial !!')
        test_dataset = MitArrhythmiaLoader(type='atrial').load()
        logger.info(f'Dataset loaded. ')

        acc_total = {}
        for data_tag in test_dataset.keys():
            logger.info(f'Test Dataset - {data_tag}')

            dataModel = MitdbDataModel(test_dataset[data_tag])

            #AFIB/AFL 둘중 하나라도 있으면 진행
            acc_total[data_tag] = {}
            total_afib_data = []
            total_afl_data = []
            for pid in dataModel.getPidList():
                afibData = dataModel.getAfibData(pid=pid)
                aflData = dataModel.getAflData(pid=pid)

                if len(afibData) != 0 or len(aflData) != 0:
                    model_input, atrial_range = self.create_input(pid, dataModel)
                    model_input = torch.tensor(model_input, dtype=torch.float32).to(self.device)  # Tensor 변환
                    model_output, _ = self.atrial_model.forward_features(model_input)
                    model_output = model_output.detach().cpu().numpy()
                    acc = self.calculate_accuracy(model_output=model_output, atrial_range=atrial_range)

                    if acc[TAG_AFIB].t_count != 0:
                        logger.info(f'pid :: {pid}, AFIB\ntrue: {acc[TAG_AFIB].t_count}, '
                              f'positive: {acc[TAG_AFIB].p_count}, sense={acc[TAG_AFIB].getSense()}, ppv={acc[TAG_AFIB].getPpv()}')
                        total_afib_data.append(acc[TAG_AFIB])
                    if acc[TAG_AFL].t_count != 0:
                        logger.info(f'pid :: {pid}, AFL\ntrue: {acc[TAG_AFL].t_count}, '
                              f'positive: {acc[TAG_AFL].p_count}, sense={acc[TAG_AFL].getSense()}, ppv={acc[TAG_AFL].getPpv()}')
                        total_afl_data.append(acc[TAG_AFL])
            acc_total_afib = AccuracyDataModel()
            for afib_acc in total_afib_data:
                acc_total_afib.t_count += afib_acc.t_count
                acc_total_afib.p_count += afib_acc.p_count
                acc_total_afib.tp_count += afib_acc.tp_count
                acc_total_afib.total_count += afib_acc.total_count
            acc_total[data_tag][TAG_AFIB] = acc_total_afib

            acc_total_afl = AccuracyDataModel()
            for afl_acc in total_afl_data:
                acc_total_afl.t_count += afl_acc.t_count
                acc_total_afl.p_count += afl_acc.p_count
                acc_total_afl.tp_count += afl_acc.tp_count
                acc_total_afl.total_count += afl_acc.total_count
            acc_total[data_tag][TAG_AFL] = acc_total_afl

            logger.info(f'==== Dataset - {data_tag} summary====\n'
                        f'AFL - true: {acc_total[data_tag][TAG_AFL].t_count}'
                        f', 'f'positive: {acc_total[data_tag][TAG_AFL].p_count}'
                        f', sense={acc_total[data_tag][TAG_AFL].getSense()}'
                        f', ppv={acc_total[data_tag][TAG_AFL].getPpv()}'
                        )

            logger.info(f'AFIB - true: {acc_total[data_tag][TAG_AFIB].t_count}'
                        f', 'f'positive: {acc_total[data_tag][TAG_AFIB].p_count}'
                        f', sense={acc_total[data_tag][TAG_AFIB].getSense()}'
                        f', ppv={acc_total[data_tag][TAG_AFIB].getPpv()}'
                        )

        acc_afib_total_model = AccuracyDataModel()
        acc_afl_total_model = AccuracyDataModel()

        for acc_label in acc_total.keys():
            acc_afib_total_model.t_count += acc_total[acc_label][TAG_AFIB].t_count
            acc_afib_total_model.p_count += acc_total[acc_label][TAG_AFIB].p_count
            acc_afib_total_model.tp_count += acc_total[acc_label][TAG_AFIB].tp_count
            acc_afib_total_model.total_count += acc_total[acc_label][TAG_AFIB].total_count

        for acc_label in acc_total.keys():
            acc_afl_total_model.t_count += acc_total[acc_label][TAG_AFL].t_count
            acc_afl_total_model.p_count += acc_total[acc_label][TAG_AFL].p_count
            acc_afl_total_model.tp_count += acc_total[acc_label][TAG_AFL].tp_count
            acc_afl_total_model.total_count += acc_total[acc_label][TAG_AFL].total_count

        logger.info(f'==== Atrial summary====\n'
                    f'AFL - true: {acc_afl_total_model.t_count}'
                    f', 'f'positive: {acc_afl_total_model.p_count}'
                    f', sense={acc_afl_total_model.getSense()}'
                    f', ppv={acc_afl_total_model.getPpv()}')
        logger.info(f'AFIB - true: {acc_afib_total_model.t_count}'
                    f', 'f'positive: {acc_afib_total_model.p_count}'
                    f', sense={acc_afib_total_model.getSense()}'
                    f', ppv={acc_afib_total_model.getPpv()}')


    def create_input(self, pid: str, dataModel: MitdbDataModel):
        signal_split_10s = []

        raw_signal = dataModel.getPidSignal(pid)
        pid_fs = dataModel.getPidFs(pid)
        raw_nparray = np.array(raw_signal)
        afib_data_array = dataModel.getAfibData(pid)
        afl_data_array = dataModel.getAflData(pid)

        resample_raw_signal = resample(raw_nparray, int(len(raw_nparray) / pid_fs * FS))
        atrial_range = np.zeros(len(resample_raw_signal))

        for record_afib_data in afib_data_array:
            start_sample = int(record_afib_data['start_sample'] / pid_fs * FS)
            end_sample = int(record_afib_data['end_sample'] / pid_fs * FS)
            atrial_range[start_sample:end_sample] = 2
        for record_afl_data in afl_data_array:
            start_sample = int(record_afl_data['start_sample'] / pid_fs * FS)
            end_sample = int(record_afl_data['end_sample'] / pid_fs * FS)
            atrial_range[start_sample:end_sample] = 1

        len_10s_sample = 1000

        for idx in range(len(resample_raw_signal) // len_10s_sample):
            signal_10s = resample_raw_signal[idx * len_10s_sample: (idx + 1) * len_10s_sample]
            signal_split_10s.append(signal_10s)

        if len(resample_raw_signal) % len_10s_sample != 0:
            signal_10s_last = resample_raw_signal[(idx + 1) * len_10s_sample:]
            signal_10s_last = np.pad(signal_10s_last, (0, len_10s_sample - len(signal_10s_last)), mode='constant')
            signal_split_10s.append(signal_10s_last)

        ecg_pid_array = np.array([signal[np.newaxis, :] for signal in signal_split_10s])

        return ecg_pid_array, atrial_range


    def calculate_accuracy(self, model_output, atrial_range):
        # 모델 출력에서 예측값 계산 (argmax 사용)
        quant_output = np.argmax(model_output, axis=1)

        # 10초 단위로 AFL/AFIB 구간 체크
        atrial_signal_10s = np.zeros(len(quant_output))
        onset_idx = 0
        len_sample_10s = 1000

        # AFL(1) 구간 체크
        for i in range(len(atrial_signal_10s)):
            if i < len(atrial_signal_10s) - 1:
                isAfl = np.isin(1, atrial_range[onset_idx:(onset_idx + len_sample_10s)])
            else:
                isAfl = np.isin(1, atrial_range[onset_idx:])
            onset_idx += len_sample_10s
            if isAfl:
                atrial_signal_10s[i] = 1  # AFL 라벨
            else:
                atrial_signal_10s[i] = 0  # Normal/Other 라벨

        # AFIB(2) 구간 체크
        onset_idx = 0
        for i in range(len(atrial_signal_10s)):
            if i < len(atrial_signal_10s) - 1:
                isAfib = np.isin(2, atrial_range[onset_idx:(onset_idx + len_sample_10s)])
            else:
                isAfib = np.isin(2, atrial_range[onset_idx:])
            onset_idx += len_sample_10s
            if isAfib:
                atrial_signal_10s[i] = 2  # AFIB 라벨
        # AccuracyDataModel 생성
        result = {
            TAG_AFIB: AccuracyDataModel(
                t_count=np.count_nonzero(atrial_signal_10s == 2),
                p_count=np.count_nonzero(quant_output == 2),
                total_count=len(atrial_signal_10s),
                tp_count=np.sum((quant_output == 2) & (atrial_signal_10s == 2))
            ),
            TAG_AFL: AccuracyDataModel(
                t_count=np.count_nonzero(atrial_signal_10s == 1),
                p_count=np.count_nonzero(quant_output == 1),
                total_count=len(atrial_signal_10s),
                tp_count=np.sum((quant_output == 1) & (atrial_signal_10s == 1))
            )
        }

        return result
        #TODO 에피소드
