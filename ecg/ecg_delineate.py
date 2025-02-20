import time
import numpy as np
from joblib import Parallel, delayed
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

    def run(self, ecg_array):
        print("2. Segmentation")
        segmentation_result = self.ECGWaveAnalysis(ecg_array)

        return segmentation_result

    def ECGWaveAnalysis(self, ecg_array):
        start_time = time.perf_counter()
        # 1. 10초 심전도를 list로 모두 가져오기

        # 2. ECG segmentation
        segmentation_result = []
        b, a = self.load_filter_coefficients()

        for idx, ecg_signal in enumerate(ecg_array):
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

        end_time = time.perf_counter()
        execution_time1 = end_time - start_time

        print(f"2.1. Segmentation: {execution_time1} sec")

        return segmentation_result


    def load_filter_coefficients(self, filename="ecg/optimized_filter_mit_arr.json"):
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
            #TODO r peak index = 0 일때 해결 방법
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
