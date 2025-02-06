import time
import numpy as np
from vg_beat_detectors import FastNVG
import json
from scipy.signal import filtfilt, resample
import heartpy as hp
from config import FS, DURATION, BATCH_SIZE, LEAD, NUM_COMPONENTS_CLASSES, AI_CPU_CORE

class ECGSegmentation:
    def __init__(self):
        self.name = "ECGSegmentation"

    # def __del__(self):
    #     print(self.name + " object disappears.")

    def ECGWaveAnalysis(self, ECG_dict):
        start_time = time.perf_counter()
        # 1. 10초 심전도를 list로 모두 가져오기
        all_ecg10s = np.array([ECG_dict[f"ecg_{i}"]["ecg"][:, np.newaxis] for i in range(len(ECG_dict))])

        # 2. ECG segmentation
        segmentation_result = []
        b, a = self.load_filter_coefficients()

        for idx, ecg_signal in enumerate(all_ecg10s):
            # R 피크 탐지
            detected_rpeaks = self.detect_r_peaks(ecg_signal.squeeze(), FS, b, a)

            # Q와 S 피크 탐지
            q_peaks, s_peaks = self.find_qs_peaks(ecg_signal.squeeze(), detected_rpeaks)

            # P와 T 피크 탐지
            t_peaks, p_peaks = self.find_p_t_peaks(ecg_signal.squeeze(), s_peaks, q_peaks)

            # Onset 및 Offset 탐지
            p_onsets, p_offsets = self.find_onset_offset(ecg_signal.squeeze(), p_peaks)
            t_onsets, t_offsets = self.find_onset_offset(ecg_signal.squeeze(), t_peaks)

            # 탐지 결과 저장
            segmentation_result.append({
                "r_peaks": detected_rpeaks,
                "q_peaks": q_peaks,
                "s_peaks": s_peaks,
                "p_peaks": p_peaks,
                "t_peaks": t_peaks,
                "p_onsets": p_onsets,
                "p_offsets": p_offsets,
                "t_onsets": t_onsets,
                "t_offsets": t_offsets,
            })

        # 3. 후처리 (스무딩 및 QTc 계산)
        smoothing = DataPostprocessor(segmentation_result)
        segmentation_result = smoothing.smooth()
        del smoothing

        end_time = time.perf_counter()
        execution_time1 = end_time - start_time

        print(f"2.1. Segmentation: {execution_time1} sec")
        start_time = time.perf_counter()

        for i, result in enumerate(segmentation_result):
            ECG_dict[f"ecg_{i}"]["segment"] = self.transformResult(result)
            ECG_dict[f"ecg_{i}"]["beat_qtc"] = self.calculateQTc(ECG_dict[f"ecg_{i}"]["segment"],
                                                                ECG_dict[f"ecg_{i}"]["ecg"])

        end_time = time.perf_counter()
        execution_time2 = end_time - start_time

        print(f"2.2. QTc: {execution_time2} sec")
        print(f"Segmentation time: {execution_time1 + execution_time2} sec\n")
        return ECG_dict


    def load_filter_coefficients(self, filename="optimized_filter_mit_arr.json"):
        try:
            with open(filename, "r") as file:
                filter_data = json.load(file)
            b_coeffs = np.array(filter_data["b_coefficients"])
            a_coeffs = np.array(filter_data["a_coefficients"])
            return b_coeffs, a_coeffs
        except FileNotFoundError:
            raise FileNotFoundError(f"Filter coefficients file '{filename}' not found.")

    def detect_r_peaks(self, signal, fs, b, a):
        detector = FastNVG(fs)
        filtered_signal = filtfilt(b, a, signal)
        baseline_removed = hp.remove_baseline_wander(filtered_signal, sample_rate=fs)
        rpeaks = detector.find_peaks(baseline_removed)

        # R 피크 주변 15포인트 내 더 큰 값 탐색
        refined_rpeaks = []
        for peak in rpeaks:
            search_start = max(0, peak - 15)
            search_end = min(len(signal), peak + 15)
            refined_rpeaks.append(search_start + np.argmax(signal[search_start:search_end]))

        return np.array(refined_rpeaks)


    def find_qs_peaks(self, signal, r_peaks, search_window=30):
        if not isinstance(r_peaks, (np.ndarray, list)):
            raise ValueError("r_peaks must be a numpy array or list.")
        r_peaks = np.array(r_peaks).squeeze()

        q_peaks = []
        s_peaks = []

        for r in r_peaks:
            q_start = max(0, r - search_window)
            q_end = r
            s_start = r
            s_end = min(len(signal), r + search_window)

            q_peaks.append(q_start + np.argmin(signal[q_start:q_end]))
            s_peaks.append(s_start + np.argmin(signal[s_start:s_end]))

        return np.array(q_peaks), np.array(s_peaks)

    def find_p_t_peaks(self, signal, s_peaks, q_peaks):
        """
        T 피크와 P 피크를 이전 S와 현재 Q 사이를 기준으로,
        T 피크는 왼쪽 2/3에서 최댓값
        """
        t_peaks = []
        p_peaks = []

        for i in range(1, len(s_peaks)):
            s = s_peaks[i - 1]
            q = q_peaks[i]
            t_end = s + 2 * (q - s) // 3  # T 피크는 S에서 2/3 지점까지
            p_start = t_end  # P 피크는 T 피크의 끝에서 시작

            if s < t_end < q:
                t_peaks.append(s + np.argmax(signal[s:t_end]))
                p_peaks.append(t_end + np.argmax(signal[t_end:q]))
            else:
                t_peaks.append(-1)  # 유효하지 않은 경우
                p_peaks.append(-1)  # 유효하지 않은 경우

        return np.array(t_peaks), np.array(p_peaks)
        
    def find_onset_offset(self, signal, peaks, search_window=30):
        onsets = []
        offsets = []

        for peak in peaks:
            # Onset 탐지
            onset_start = max(0, peak - search_window)
            onset_end = peak
            onset = onset_start + np.argmin(signal[onset_start:onset_end])
            onsets.append(onset)

            # Offset 탐지
            offset_start = peak
            offset_end = min(len(signal), peak + search_window)
            offset = offset_start + np.argmin(signal[offset_start:offset_end])
            offsets.append(offset)

        return np.array(onsets), np.array(offsets)



    def transformResult(self, segmentation_result):
        segments = []  # 연속된 구간을 저장할 리스트
        current_value = None
        onset = None

        for i, value in enumerate(segmentation_result):
            if current_value is None:
                current_value = value
                onset = i
            elif value != current_value:
                offset = i
                if current_value != 0:  # 0은 무시
                    if onset < offset:
                        segments.append((int(current_value), onset, offset))
                current_value = value
                onset = i

        # 마지막 구간 처리
        if current_value != 0:
            offset = len(segmentation_result) - 1
            if onset < offset:
                segments.append([int(current_value), onset, offset])

        return segments

    def calculateQTc(self, segments, ecg):
        qrs_and_t_segments = [segment for segment in segments if segment[0] != 1] # QRS (2), T (3) 만 추출

        # 1. 연속적인 QRS파와 T파 그룹화.
        # 맨 앞에 P파이면 o, 아니면 제외
        # 맨 뒤이 잘려 있을 수 있으므로 제외.
        # 1.1. 그루핑
        grouped_data = []
        group = []

        for segment in qrs_and_t_segments:
            label, onset, offset = segment

            if label == 2: # QRS
                group = [(2, onset, offset)]
            elif label == 3 and group: # T고 group에 QRS가 있으면, T 추가.
                group.append((3, onset, offset))
                grouped_data.append(group)
                group = []

        # 1.2. QRS-T 그룹 제외 판정
        # 해당 10초 심전도가 P파로 시작될 경우, 첫번째 심박의 QRS와 T는 안정적이므로 제외 X, 그게 아니면 제외
        # 맨 뒤는 안정성 보장으로 인해 무조건 제외
        if segments[0][0] == 0:
            grouped_data = grouped_data[:-1]
        else:
            grouped_data = grouped_data[1:-1]

        # 결과 출력
        QTI_list = []
        for group in grouped_data: # N개
            qrs, t = group
            t_offset, qrs_onset = t[2], qrs[1]
            QTI_list.append(t_offset - qrs_onset) # T_offset - Q_onset = QT interval

        QTI_list = np.array(QTI_list) / FS

        # 2. RRI
        QRS_tuples = [group[0] for group in grouped_data] # QRS (2)만 모아둠, N개
        R_indices = []
        for _, onset, offset in QRS_tuples:
            qrs = ecg[onset:offset]
            R_idx = np.argmax(qrs) + onset
            R_indices.append(R_idx) # N개
        RRI_list = np.diff(R_indices) / FS  # unit: sec, N-1개
        RRI_sqrt_list = np.sqrt(RRI_list) # N-1개

        if RRI_sqrt_list.size != 0:
            RRI_sqrt_list = np.append(RRI_sqrt_list, RRI_sqrt_list[-1])

        QTc_list = np.round(QTI_list / RRI_sqrt_list, 3)
        QTc_with_R_idx = [[float(QT_info[0]), int(QT_info[1])] for QT_info in zip(QTc_list, R_indices)] # (QTc, R_idx)
        return QTc_with_R_idx

    def main(self, ECG_dict):
        print("2. Segmentation")
        ECG_dict = self.ECGWaveAnalysis(ECG_dict)

        return ECG_dict


class DataPostprocessor:
    def __init__(self, y_pred):
        self.name = "Data_postprocessor"
        self.SPBon200BPM = 0.3 # s
        self.interval = FS * self.SPBon200BPM # 75 samples
        self.y_pred = y_pred

    # def __del__(self):
    #     print(self.name + " object disappears.")

    def smoothComponent(self, mask, mask_component, value):
        transitions_value_to_0 = np.where(np.diff(mask_component) == -value)[0]
        transitions_0_to_value = np.where(np.diff(mask_component) == value)[0]

        # 거리가 75보다 작은 구간을 p=1, qrs=2, t=3로 변경
        for start_idx in transitions_value_to_0:
            next_0_to_1 = transitions_0_to_value[transitions_0_to_value > start_idx]
            if next_0_to_1.size > 0:
                end_idx = next_0_to_1[0]
                if end_idx - start_idx < self.interval:
                    mask[start_idx:end_idx + 1] = value
        return mask

    def _smooth(self, mask):
        # 1. P
        mask_component = mask.copy()
        mask_component[mask_component != 1] = 0
        mask = self.smoothComponent(mask, mask_component, 1)

        # 2. QRS
        mask_component = mask.copy()
        mask_component[mask_component != 2] = 0
        mask = self.smoothComponent(mask, mask_component, 2)

        # 3. T
        mask_component = mask.copy()
        mask_component[mask_component != 3] = 0
        mask = self.smoothComponent(mask, mask_component, 3)
        return mask

    def smooth(self):
        return np.array(Parallel(AI_CPU_CORE)(delayed(self._smooth)(mask) for mask in self.y_pred))
