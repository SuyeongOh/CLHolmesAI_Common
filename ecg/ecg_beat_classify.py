from datetime import time

from tensorflow.keras.layers import (Conv1D, MaxPool1D, Concatenate, BatchNormalization, Activation, Input, Add,
                                     GlobalAveragePooling1D, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from sklearn.cluster import KMeans

import numpy as np
import gc

from config import BATCH_SIZE, CLASSES_INDEX2, FS, LEAD, INPUT_SHAPE, NUM_BEAT_CLASSES


class BeatAnalysis:
    def __init__(self):
        self.name = "BeatAnalysis"

    def movingAvarageSmoothing(self, X, window_size=2):
        S = np.zeros(X.shape[0])
        for t in range(X.shape[0]):
            if t < X.shape[0] - window_size:
                S[t] = np.sum(X[t:t + window_size]) / window_size
            else:
                S[t] = S[t - 1]
        return S

    def findOptimalK(self, data, max_k=25, min_k=2):

        tmp_data = np.unique(data, axis=0)

        if 0 < tmp_data.shape[0] <= max_k:
            max_k = tmp_data.shape[0]
            del tmp_data
        elif tmp_data.shape[0] == 1:
            return 1, 1
        else:
            pass

        print("max_k:", max_k)

        sse_list = []
        optimal_k = 0
        labels_list = []

        print(f"Calculating K-means clustering... (K = {min_k} ~ {max_k})")
        if max_k == 1 or min_k == max_k:
            return 1, 1
        #elif
        else:

            for k in range(min_k, max_k + 1):
                # KMeans 클러스터링 수행
                print(k, end=", ")
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

                labels = kmeans.fit(data).labels_  # 결과값
                labels_list.append(labels)

                sse = kmeans.inertia_
                sse_list.append(sse)

            sse_list = np.array(sse_list)

            # 0. Signal smoothing
            ma_sse_list = self.movingAvarageSmoothing(sse_list)

            # 1. Scale into range with k_ragne
            scaled_sse_list = (ma_sse_list - np.min(ma_sse_list)) / (np.max(ma_sse_list) - np.min(ma_sse_list)) * len(
                ma_sse_list)

            # 2. Calculate gradient1
            first_diff_sse_list = np.diff(scaled_sse_list)

            # 3. Find the element index closest to gradient with -1.
            index = np.abs(first_diff_sse_list + 1).argmin()

            # 4. Correct optimal_k
            optimal_k = index + min_k + 1 # 6 + 2 = 8
            # print()
            # print(min(labels_list[index + 1]), max(labels_list[index + 1]), optimal_k)


            return labels_list[index + 1], optimal_k

    def beatClassifier(self, ECG_dict):
        # 1. Beat Classification
        start_time = time.perf_counter()

        number_heartbeats_list, all_Heartbeats_list = [], []
        for i, ecg_i in enumerate(ECG_dict.values()):
            Hearbeats_list = ecg_i["class_analysis"]["beat_amp"]
            n_heartbeats = len(Hearbeats_list)

            if n_heartbeats == 0:
                n_heartbeats = 1
                #input shape으로 진행되어야함
                Hearbeats_list = [np.zeros([FS//2, LEAD])]

            number_heartbeats_list.append(n_heartbeats)  # ECG10s에 포함된 R-peak 갯수
            all_Heartbeats_list.extend(Hearbeats_list)  # ECG10s에 포함된 심박파형들
        # print("The number of heaetbeats:", sum(N_Heartbeats_list))
        all_Heartbeats_list = np.array(all_Heartbeats_list)



        # 1. 심박 파형 분류 모델 빌드 및 가중치 로드
        model = BeatClassifierArchitecture().model
        model.load_weights("weights/heartbeat_classifier_weights.h5")
        result_class = model.predict(all_Heartbeats_list, batch_size=BATCH_SIZE)

        # 메모리 해제
        del model
        clear_session()
        gc.collect()

        result_class = np.argmax(result_class, axis=1)

        end_time = time.perf_counter()
        execution_time1 = end_time - start_time
        print(f"4.1. Beat classification: {execution_time1} sec")

        # 2.Beat Subclassification using K-means
        start_time = time.perf_counter()

        # 마지막 축 제거 : (N, 125, 1) --> (N, 125)
        all_Heartbeats_list = all_Heartbeats_list.reshape(all_Heartbeats_list.shape[0], all_Heartbeats_list.shape[1])

        # N
        N_indices = np.where(result_class == CLASSES_INDEX2["N"])[0]
        # class N의 심박 파형 수가 있을 때 --> 2개 이상인지 판정
        Num_N = len(N_indices)
        print("N 수:", Num_N)
        if len(N_indices) > 0:
            N_Heatbeats_list = all_Heartbeats_list[N_indices]

            # class N의 심박 파형 수가 2개 이상 일 때 --> K-means clustering
            if Num_N >= 2:
                # 최적의 K 값 및 실루엣 계수 점수 계산
                SUBCLASSES_N_list, optimal_k = self.findOptimalK(N_Heatbeats_list)
                if optimal_k == 1:
                    print("\nN - number of subclass group:", 1)
                    info_N_pairs = [[idx, 1101] for idx in N_indices]
                else:
                    print("\nN - number of subclass group:", optimal_k)
                    info_N_pairs = [[idx, 1100 + int(subcls) + 1] for idx, subcls in zip(N_indices, SUBCLASSES_N_list)]
                del N_Heatbeats_list, SUBCLASSES_N_list
            # class N의 심박 파형 수가 1개 일 때 --> 1개 그룹으로 처리함.
            else:
                print("\nN - number of subclass group:", 1)
                info_N_pairs = [[idx, 1101] for idx in N_indices]
        else:
            info_N_pairs = []

        # S
        S_indices = np.where(result_class == CLASSES_INDEX2["S"])[0]
        Num_S = len(S_indices)
        print("\nS 수:", Num_S)
        # class S의 심박 파형 수가 있을 때 --> 2개 이상인지 판정
        if len(S_indices) > 0:
            S_Heatbeats_list = all_Heartbeats_list[S_indices]

            # class S의 심박 파형 수가 2개 이상 일 때 --> K-means clustering
            if Num_S >= 2:
                # 최적의 K 값 및 실루엣 계수 점수 계산
                SUBCLASSES_S_list, optimal_k = self.findOptimalK(S_Heatbeats_list)
                if optimal_k == 1:
                    print("\nS - number of subclass group:", 1)
                    info_S_pairs = [[idx, 1201] for idx in S_indices]
                else:
                    print("\nS - number of subclass group:", optimal_k)
                    info_S_pairs = [[idx, 1200 + int(subcls) + 1] for idx, subcls in zip(S_indices, SUBCLASSES_S_list)]
                del S_Heatbeats_list, SUBCLASSES_S_list
            # class S의 심박 파형 수가 1개 일 때 --> 1개 그룹으로 처리함.
            else:
                print("\nS - number of subclass group:", 1)
                info_S_pairs = [[idx, 1201] for idx in S_indices]
        else:
            info_S_pairs = []

        # V
        V_indices = np.where(result_class == CLASSES_INDEX2["V"])[0]
        Num_V = len(V_indices)
        print("\nV 수:", Num_V)
        # class V의 심박 파형 수가 있을 때 --> 2개 이상인지 판정
        if len(V_indices) > 0:
            V_Heatbeats_list = all_Heartbeats_list[V_indices]

            # class V의 심박 파형 수가 2개 이상 일 때 --> K-means clustering
            if Num_V >= 2:
                # 최적의 K 값 및 실루엣 계수 점수 계산
                SUBCLASSES_V_list, optimal_k = self.findOptimalK(V_Heatbeats_list)
                if optimal_k == 1:
                    print("\nV - number of subclass group:", 1)
                    info_V_pairs = [[idx, 1301] for idx in V_indices]
                else:
                    print("\nV - number of subclass group:", optimal_k)
                    info_V_pairs = [[idx, 1300 + int(subcls) + 1] for idx, subcls in zip(V_indices, SUBCLASSES_V_list)]
                del V_Heatbeats_list, SUBCLASSES_V_list
            # class V의 심박 파형 수가 1개 일 때 --> 1개 그룹으로 처리함.
            else:
                print("\nV - number of subclass group:", 1)
                info_V_pairs = [[idx, 1301] for idx in V_indices]
        else:
            info_V_pairs = []

        # A
        A_indices = np.where(result_class == CLASSES_INDEX2["A"])[0]
        Num_A = len(A_indices)
        print("\nA 수:", Num_A)
        if len(A_indices) > 0:
            # A - 14XX
            info_A_pairs = [[idx, 1401] for idx in A_indices]
        else:
            info_A_pairs = []
        del result_class

        # Total
        info_pairs = info_N_pairs + info_S_pairs + info_V_pairs + info_A_pairs
        del info_N_pairs, info_S_pairs, info_V_pairs, info_A_pairs
        info_pairs.sort(key=lambda x: x[0])  # idx 기준으로 오름차순 정렬

        # 전체 N, S, V, A 갯수
        total_num_beats_dict = {"N": Num_N,
                                "S": Num_S,
                                "V": Num_V,
                                "A": Num_A,
                                }

        # 3. Update results: heartbeat == 0 --> 1401 으로 보정
        cnt = 0
        for i, num_beats in enumerate(number_heartbeats_list):
            beats_cls_info = info_pairs[cnt:cnt + num_beats]
            ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_class"] = [cls_info[1] for cls_info in beats_cls_info]

            if 1 <= len(ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_indice"]) <= 2:
                # ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_class"] = [("A", 1) for _ in range(num_beats)]
                ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_class"] = [1401 for _ in range(num_beats)]

            elif len(ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_indice"]) == 0:
                ECG_dict[f"ecg_{i}"]["class_analysis"]["beat_class"] = []
            cnt += num_beats
        print(f"Total number of QRS: {cnt}")
        print(total_num_beats_dict)
        end_time = time.perf_counter()
        execution_time2 = end_time - start_time
        print(f"4.2. Beat sub-classification: {execution_time2} sec")
        print(f"Beat analysis: {execution_time1 + execution_time2} sec\n")
        return ECG_dict, total_num_beats_dict


class BeatClassifierArchitecture:
    def __init__(self, nb_filters=32, use_residual=True, use_bottleneck=True, depth=10, kernel_size=41):
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        self.model = self.buildModel()

    def inceptionModule(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=self.nb_filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def shortcutLayer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    def buildModel(self):
        input_layer = Input(INPUT_SHAPE)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self.inceptionModule(x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcutLayer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        output_layer = Dense(NUM_BEAT_CLASSES, activation='softmax')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        return model
