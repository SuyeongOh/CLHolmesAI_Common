import neurokit2 as nk
import numpy as np
from scipy.signal import resample

# 모델 관련 모듈
from ecg.ecg_beat_classify import BeatClassifierArchitecture, ResNeXtWithTransformerSE
from config import BATCH_SIZE, BEAT_MODEL_DICT
from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.datamodel.mitdb_model import MitdbDataModel
from utils.ecg_utils import min_max_normalization

# SWEEP 모드에 필요한 모듈
from stockwell import st
import torch
import gc

import pandas as pd
import openpyxl
# 임시 저장용 전역 변수
resample_signal = []

class TestBeatAnalysis:

    def run(self):
        """
        np_data의 각 PID에 대해 모델 평가를 진행합니다.

        - BEAT_MODEL_DICT['sweep']가 True이면, model_list에 명시된 모든 모델에 대해 평가합니다.
        - False이면, BEAT_MODEL_DICT['model']에 명시된 단일 모델에 대해 평가합니다.

        모델별 전처리/예측 방식:
          * 'beat_classifer': 리샘플링 → detrend → min_max_normalization 적용 + BeatClassifierArchitecture 모델
          * 'resxtranformer': Stockwell Transform 적용 후 실수/허수부 분리 + ResNeXtWithTransformerSE 모델

        평가 후 모델 객체를 삭제하고 GPU 메모리를 비웁니다.

        :param np_data: {'pid': array, 'data': array, 'group': array} 형태의 데이터
        :param raw_fs: (Stockwell 전처리 사용 시 필요) 각 pid의 sampling rate {pid: fs, ...}
        :return: pid별, 모델별 평가 결과 딕셔너리
        """

        test_dataset = MitArrhythmiaLoader('classify').load()

        for dataset_name in test_dataset.keys():
            dataModel = MitdbDataModel(test_dataset[dataset_name])
            pid_list = dataModel.getPidList()
            np_data = dataModel.np_data
            results = {pid: {} for pid in pid_list}

            # 모델 리스트 결정
            if BEAT_MODEL_DICT['sweep']:
                model_list = BEAT_MODEL_DICT['model_list']
            else:
                model_list = [BEAT_MODEL_DICT['model']]

            # 각 모델별 평가 진행
            for model_name in model_list:
                print(f"\n== 평가 시작: 모델 {model_name} ==")

                if model_name == 'beat_classifer':
                    beat_model = BeatClassifierArchitecture().model
                    beat_model.load_weights("weights/heartbeat_classifier_weights.h5")
                    # beat_class_map: beat_classifier 모델은 'A'가 포함될 수 있으므로,
                    # target_label과 union하여 평가하면, 만약 target에 'Q','F'가 없다면 'A'도 평가 대상에 포함됨.
                    beat_class_map = {0: 'N', 1: 'S', 2: 'V', 3: 'A'}
                    preprocess_func = lambda beat: min_max_normalization(
                        nk.signal_detrend(np.array(resample(beat, 125)))
                    )
                elif model_name == 'resxtranformer':
                    beat_model = ResNeXtWithTransformerSE(
                        num_classes=5,
                        transformer_layers=4,
                        transformer_heads=8,
                        dropout=0.4,
                        embed_dim=64,
                        cardinality=32,
                        bottleneck_width=4,
                        drop_prob=0.2,
                        reduction=16
                    ).cuda()
                    beat_model.load_state_dict(torch.load('weights/heartbeat_classifier_resxtransformer2.pth'))
                    beat_model.eval()
                    beat_class_map = {0: 'N', 1: 'S', 2: 'V', 3: 'Q', 4: 'F'}
                else:
                    raise ValueError(f"알 수 없는 모델 이름: {model_name}")

                for pid in pid_list:
                    fs = dataModel.getPidFs()
                    pid_idxs = np.where(np_data['pid'] == pid)[0]
                    onset, offset = pid_idxs[0], pid_idxs[-1]
                    target_data = np_data['data'][onset:offset]
                    target_label = np_data['group'][onset:offset]

                    resample_signal.clear()
                    for beat in target_data:
                        resample_signal.append(self.preprocess_func(beat))

                    if model_name == 'beat_classifer':
                        data_input = np.concatenate([arr[np.newaxis, :, np.newaxis] for arr in resample_signal], axis=0)
                        result_class = beat_model.predict(data_input, batch_size=BATCH_SIZE)
                        result_class = np.argmax(result_class, axis=1)
                    elif model_name == 'resxtranformer':
                        data_input = np.stack(resample_signal, axis=0)
                        SPLIT_SIZE = len(resample_signal) / BATCH_SIZE
                        batch_np_array = np.array_split(data_input, SPLIT_SIZE)
                        result_class_list = []
                        with torch.no_grad():
                            for batch in batch_np_array:
                                tensor_batch = torch.tensor(batch, dtype=torch.float).cuda()
                                outputs = beat_model(tensor_batch)
                                result_class_list.append(torch.argmax(outputs, dim=1).cpu().numpy())
                        result_class = np.concatenate(result_class_list, axis=0)
                    else:
                        result_class = None

                    # 평가 (동적으로 eval_keys가 결정됨)
                    sensitivity, ppv, fn = self.evaluate_result(result_class, target_label, beat_class_map)
                    results[pid][model_name] = {
                        'sensitivity': sensitivity,
                        'ppv': ppv,
                        'fn': fn
                    }
                    print(f"pid_{pid} - 모델 {model_name}: sensitivity {sensitivity}, PPV {ppv}, FN {fn}")

                    resample_signal.clear()
                # 메모리 릴리즈: 모델 객체 삭제, 캐시 비우기 및 가비지 컬렉션 수행
                torch.cuda.synchronize()
                del beat_model
                torch.cuda.empty_cache()
                gc.collect()

            self.save_results_to_excel(results)

    def preprocess_func(self, beat):
        trans_signal = self.stockwell_transform(beat, 0, 15, 10)
        real_part = np.real(trans_signal)
        imag_part = np.imag(trans_signal)
        return np.stack((real_part, imag_part), axis=0)

    def save_results_to_excel(self, results: dict, output_path: str = "model_comparison.xlsx"):
        # 추출할 라벨 순서
        labels = ['N', 'S', 'V', 'Q', 'A', 'U']
        metrics = ['F1', 'Se', 'PPV']
        rows = []

        # 라벨별, PID별 데이터 정리
        for label in labels:
            rows.append({'Label': label})  # 라벨 이름 추가
            for metric in metrics:
                metric_row = {'Label': f"  {metric}"}
                for pid, model_results in results.items():
                    for model_name, metrics_data in model_results.items():
                        # Sensitivity, PPV 데이터 가져오기
                        sensitivity = metrics_data['sensitivity'].get(label, 0)
                        ppv = metrics_data['ppv'].get(label, 0)
                        f1 = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0

                        # 저장할 값 선택
                        value = {
                            'F1': f1,
                            'Se': sensitivity,
                            'PPV': ppv,
                        }[metric]

                        # PID, Model 컬럼 데이터 추가
                        col_name = f"{pid} - {model_name}"
                        metric_row[col_name] = value
                rows.append(metric_row)
            rows.append({})  # 빈 줄 추가

        # DataFrame 생성
        df = pd.DataFrame(rows)

        # Excel 저장
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Comparison", index=False)
        print(f"평가 결과가 '{output_path}'에 저장되었습니다.")


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


    def evaluate_result(self, result_class, target_label, beat_class_map):
        """
        예측 결과와 실제 라벨을 비교하여 sensitivity, PPV, FN(실패 부정률)을 계산합니다.
        평가 대상은 target_label과 beat_class_map에 나타난 모든 라벨의 합집합(예, N, S, V, Q, F 등)입니다.

        :param result_class: 예측 결과 (클래스 인덱스 배열)
        :param target_label: 실제 라벨 리스트 (해당 PID 내)
        :param beat_class_map: {예측 클래스 인덱스: 라벨 문자} 매핑
        :return: sensitivity, ppv, fn (각각 딕셔너리)
        """
        # 평가 대상 라벨: target_label과 모델의 클래스 매핑의 합집합
        eval_keys = sorted(list(set(target_label).union(set(beat_class_map.values()))))

        # 초기화
        beat_class_correct = {key: [] for key in eval_keys}
        beat_class_fp = {key: [] for key in eval_keys}
        beat_class_tn = {key: [] for key in eval_keys}

        # 예측과 실제 비교
        for pred, label in zip(result_class, target_label):
            pred_label = beat_class_map[pred]
            if pred_label == label:
                # 실제 라벨이 평가 대상이면 기록
                if label in eval_keys:
                    beat_class_correct[label].append(label)
            else:
                # 예측된 라벨과 실제 라벨 모두 평가 대상이면 기록
                if pred_label in eval_keys:
                    beat_class_fp[pred_label].append(pred_label)
                if label in eval_keys:
                    beat_class_tn[label].append(label)

        # target_label에 기반한 라벨별 총 개수 (평가 대상에 포함된 라벨만)
        target_counts = {key: np.sum(np.array(target_label) == key) for key in eval_keys}

        # 예측 결과에 따른 각 라벨의 총 예측 개수
        total_analysis_class_count = {key: 0 for key in eval_keys}
        for pred in result_class:
            pred_label = beat_class_map[pred]
            if pred_label in total_analysis_class_count:
                total_analysis_class_count[pred_label] += 1

        sensitivity, ppv, fn = {}, {}, {}
        for key in eval_keys:
            count_correct = len(beat_class_correct[key])
            sensitivity[key] = (count_correct / target_counts[key]) if target_counts.get(key, 0) != 0 else 0
            ppv[key] = (count_correct / total_analysis_class_count[key]) if total_analysis_class_count.get(key,
                                                                                                           0) != 0 else 0
            fp_count = len(beat_class_fp[key])
            tn_count = len(beat_class_tn[key])
            fn[key] = (fp_count / (tn_count + fp_count)) if (tn_count + fp_count) != 0 else 0

        return sensitivity, ppv, fn

