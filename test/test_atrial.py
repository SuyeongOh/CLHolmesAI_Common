import logging

from config import DURATION, FS, AF_THRESHOLD, BATCH_SIZE
from ecg.ecg_atrial import DDNN

from scipy.signal import resample
import numpy as np

from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.datamodel.mitdb_model import MitdbDataModel
from test.datamodel.accuracy_datamodel import AccuracyDataModel
from utils import log_utils

TAG_AFIB = 'afib'
TAG_AFL = 'afl'

logger = log_utils.getCustomLogger(__name__)

class TestAtrial:
    atrial_model = DDNN().ddnn()

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

                if (not afibData) or (not aflData):
                    model_input, atrial_range = self.create_input(pid, dataModel)
                    model_output = self.atrial_model.predict(x=model_input, batch_size=BATCH_SIZE)
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
            acc_total[data_tag][TAG_AFIB] = acc_total_afib

            acc_total_afl = AccuracyDataModel()
            for afl_acc in total_afl_data:
                acc_total_afl.t_count += afl_acc.t_count
                acc_total_afl.p_count += afl_acc.p_count
                acc_total_afl.tp_count += afl_acc.tp_count
            acc_total[data_tag][TAG_AFL] = acc_total_afl

            logger.info(f'==== Dataset - {data_tag} summary====\n'
                        f'AFL - true: {acc_total[data_tag][TAG_AFL].t_count}'
                        f', 'f'positive: {acc_total[data_tag][TAG_AFL].p_count}'
                        f', sense={acc_total[data_tag][TAG_AFL].getSense()}'
                        f', ppv={acc_total[data_tag][TAG_AFL].getPpv()}')
            logger.info(f'AFIB - true: {acc_total[data_tag][TAG_AFIB].t_count}'
                        f', 'f'positive: {acc_total[data_tag][TAG_AFIB].p_count}'
                        f', sense={acc_total[data_tag][TAG_AFIB].getSense()}'
                        f', ppv={acc_total[data_tag][TAG_AFIB].getPpv()}')

        acc_afib_total_model = AccuracyDataModel()
        acc_afl_total_model = AccuracyDataModel()

        for acc_label in acc_total.keys():
            acc_afib_total_model.t_count += acc_total[acc_label][TAG_AFIB].t_count
            acc_afib_total_model.p_count += acc_total[acc_label][TAG_AFIB].p_count
            acc_afib_total_model.tp_count += acc_total[acc_label][TAG_AFIB].tp_count

        for acc_label in acc_total.keys():
            acc_afl_total_model.t_count += acc_total[acc_label][TAG_AFL].t_count
            acc_afl_total_model.p_count += acc_total[acc_label][TAG_AFL].p_count
            acc_afl_total_model.tp_count += acc_total[acc_label][TAG_AFL].tp_count

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

        resample_raw_signal = resample(raw_nparray, (int)(len(raw_nparray) / pid_fs * FS))
        atrial_range = np.zeros(len(resample_raw_signal))

        #AFIB = 1, AFL = 2
        for record_afib_data in afib_data_array:
            record_afib_data['start_sample'] = (int)(record_afib_data['start_sample'] / pid_fs * FS)
            record_afib_data['end_sample'] = (int)(record_afib_data['end_sample'] / pid_fs * FS)
            atrial_range[record_afib_data['start_sample']:record_afib_data['end_sample']] = 1
        for record_afl_data in afl_data_array:
            record_afl_data['start_sample'] = (int)(record_afl_data['start_sample'] / pid_fs * FS)
            record_afl_data['end_sample'] = (int)(record_afl_data['end_sample'] / pid_fs * FS)
            atrial_range[record_afl_data['start_sample']:record_afl_data['end_sample']] = 2

        len_10s_sample = 250 * 10

        for idx in range(len(resample_raw_signal) // len_10s_sample):
            signal_10s = np.array(resample_raw_signal[idx * len_10s_sample: (idx + 1) * len_10s_sample])
            signal_split_10s.append(signal_10s)

        if len(resample_raw_signal) % len_10s_sample != 0:
            signal_10s_last = np.array(
                resample_raw_signal[(idx + 1) * len_10s_sample:]).reshape(-1, 1)
            signal_10s_last = np.pad(signal_10s_last, ((0, 2500 - len(signal_10s_last)), (0, 0)), mode='constant',
                                     constant_values=0).flatten()
            signal_split_10s.append(signal_10s_last)

        ecg_pid_array = np.concatenate([arr[np.newaxis, :, np.newaxis] for arr in signal_split_10s], axis=0)

        return ecg_pid_array, atrial_range


    def calculate_accuracy(self, model_output, atrial_range):
        quant_output = model_output.copy()
        filtered_output = np.where(model_output <= AF_THRESHOLD, 0.5 * model_output / AF_THRESHOLD,
                                   0.5 + 0.5 * (model_output - AF_THRESHOLD) / (1 - AF_THRESHOLD))

        quant_output[filtered_output < 0.5] = 0  # AFIB
        quant_output[filtered_output >= 0.5] = 1  # AFL

        # 10초단위로 AFL/AFIB 구간 체크
        atrial_signal_10s = np.zeros(len(quant_output))
        onset_idx = 0
        len_sample_10s = FS * DURATION
        for i, atrial_signal in enumerate(atrial_signal_10s):
            if not i < len(atrial_signal_10s) - 1:
                isAfl = np.isin(2, atrial_range[onset_idx: (onset_idx +len_sample_10s)])
            else:
                isAfl = np.isin(2, atrial_range[onset_idx:-1])
            onset_idx = onset_idx + len_sample_10s
            if isAfl:
                atrial_signal_10s[i] = 2
            else:
                atrial_signal_10s[i] = 0

        onset_idx = 0
        for i, atrial_signal in enumerate(atrial_signal_10s):
            if not i < len(atrial_signal_10s) - 1:
                isAfib = np.isin(1, atrial_range[onset_idx: (onset_idx +len_sample_10s)])
            else:
                isAfib = np.isin(1, atrial_range[onset_idx:-1])
            onset_idx = onset_idx + len_sample_10s
            if isAfib:
                atrial_signal_10s[i] = 1

        result = {TAG_AFIB: AccuracyDataModel(
            t_count=np.count_nonzero(atrial_signal_10s == 1),
            p_count=np.count_nonzero(quant_output == 0),
            tp_count=sum(q == 0 and a == 1 for q, a in zip(quant_output, atrial_signal_10s))
        ), TAG_AFL: AccuracyDataModel(
            t_count=np.count_nonzero(atrial_signal_10s == 2),
            p_count=np.count_nonzero(quant_output == 1),
            tp_count=sum(q == 1 and a == 2 for q, a in zip(quant_output, atrial_signal_10s))
        )}

        return result
        #TODO 에피소드
