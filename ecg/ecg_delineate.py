from datetime import time
import numpy as np

from tensorflow.keras.layers import (Conv1D, MaxPool1D, Concatenate, BatchNormalization, Activation, Input, Add,
                                     GlobalAveragePooling1D, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from sklearn.cluster import KMeans

from config import FS, DURATION, BATCH_SIZE

class ECGSegmentation:
    def __init__(self):
        self.name = "ECGSegmentation"


    # def __del__(self):
    #     print(self.name + " object disappears.")

    def ECGWaveAnalysis(self, ECG_dict):
        start_time = time.perf_counter()
        # 1. 10초 심전도를 list로 모두 가져오기
        # (N, 2500,) -> (N, 2500, 1) -> (N, 1250, 1), (N, 1250, 1) -> (2N, 1250, 1)
        all_ecg10s = np.array([ECG_dict[f"ecg_{i}"]["ecg"][:, np.newaxis] for i in range(len(ECG_dict))])
        all_ecg5s_1 = all_ecg10s[:, :FS * DURATION // 2]
        all_ecg5s_2 = all_ecg10s[:, FS * DURATION // 2:]
        all_ecg5s = np.concatenate((all_ecg5s_1, all_ecg5s_2), axis=0)
        del all_ecg5s_1, all_ecg5s_2

        # 2. ECG segmentation
        model = ECGSegmentationArchitecture().UNet1DPlusPlus() # laod model architecture
        model.load_weights("weights/ecg_segmentation_weights.h5") # load weights
        segmentation_result_5s = model.predict(all_ecg5s, batch_size=BATCH_SIZE) # (2N, 1250, 4)
        segmentation_result_5s = np.argmax(segmentation_result_5s, axis=-1)  # (2N, 1250)

        result_1 = segmentation_result_5s[:all_ecg10s.shape[0]] # (N, 1250)
        result_2 = segmentation_result_5s[all_ecg10s.shape[0]:] # (N, 1250)
        del segmentation_result_5s

        segmentation_result = np.concatenate((result_1, result_2), axis=1).astype(np.int8) # (N, 2500)
        del result_1, result_2
        del all_ecg10s

        # 메모리 해제
        del model
        clear_session()
        gc.collect()

        # 2. 후처리
        # 2.1. 스무딩 작업, QTc 계산
        smoothing = DataPostprocessor(segmentation_result)
        segmentation_result = smoothing.smooth()
        del smoothing

        end_time = time.perf_counter()
        execution_time1 = end_time - start_time

        print(f"2.1. Segmentation: {execution_time1} sec")
        start_time = time.perf_counter()

        for i in range(len(ECG_dict)):
            ECG_dict[f"ecg_{i}"]["segment"] = self.transformResult(segmentation_result[i])
            ECG_dict[f"ecg_{i}"]["beat_qtc"] = self.calculateQTc(ECG_dict[f"ecg_{i}"]["segment"],
                                                                 ECG_dict[f"ecg_{i}"]["ecg"]
                                                                 )

        end_time = time.perf_counter()
        execution_time2 = end_time - start_time

        print(f"2.2. QTc: {execution_time2} sec")
        print(f"Segmentation time: {execution_time1 + execution_time2} sec\n")
        return ECG_dict

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


class ECGSegmentationArchitecture:
    def __init__(self):
        self.name = "ECGsegmentation"
        self.num_filter = 16
        self.kernel_size = 9
        self.last_kernel_size = 1
        self.strides = 2

        self.pad_and_crop_size = 7

    def UNet1D(self):

        inputs = Input((FS*DURATION//2, LEAD))  # length of 5 sec signal is 1,250
        x = ZeroPadding1D(padding = self.pad_and_crop_size)(inputs)

        # Encoder
        x = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        # kernel_initializer : kernel가중치 행렬의 초기화 프로그램입니다( 참조 keras.initializers). 기본값은 'glorot_uniform'입니다.
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        o1 = Activation('relu')(x)
        x = MaxPooling1D(2)(o1)

        x = Conv1D(self.num_filter*2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        o2 = Activation('relu')(x)
        x = MaxPooling1D(2)(o2)

        x = Conv1D(self.num_filter*4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        o3 = Activation('relu')(x)
        x = MaxPooling1D(2)(o3)

        x = Conv1D(self.num_filter*8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        o4 = Activation('relu')(x)
        x = MaxPooling1D(2)(o4)

        x = Conv1D(self.num_filter*16, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*16, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)

        # Decoder
        x = Conv1DTranspose(self.num_filter*8, self.kernel_size-1, strides=self.strides, padding='same')(x)
        x = concatenate([x, o4])
        x = Conv1D(self.num_filter*8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)

        x = Conv1DTranspose(self.num_filter*4, self.kernel_size-1, strides=self.strides, padding='same')(x)
        x = concatenate([x, o3])
        x = Conv1D(self.num_filter*4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)

        x = Conv1DTranspose(self.num_filter*2, self.kernel_size-1, strides=self.strides, padding='same')(x)
        x = concatenate([x, o2])
        x = Conv1D(self.num_filter*2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter*2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)

        x = Conv1DTranspose(self.num_filter, self.kernel_size-1, strides=self.strides, padding='same')(x)
        x = concatenate([x, o1])
        x = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.3)(x)
        x = Activation('relu')(x)

        x = Conv1D(NUM_COMPONENTS_CLASSES, self.last_kernel_size, activation='softmax', kernel_initializer='he_normal')(x)
        outputs = Cropping1D(cropping = self.pad_and_crop_size)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def UNet1DPlusPlus(self):
        inputs = Input((FS * DURATION // 2, LEAD))  # length of 5 sec signal is 1,250
        x = ZeroPadding1D(padding=self.pad_and_crop_size)(inputs)

        # Encoder
        """
        x(0, 0) --> x1
        """
        x00 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x)
        x00 = BatchNormalization()(x00)
        x00 = SpatialDropout1D(0.3)(x00)
        x00 = Activation('relu')(x00)
        x00 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x00)
        x00 = BatchNormalization()(x00)
        x00 = SpatialDropout1D(0.3)(x00)
        o00 = Activation('relu')(x00)
        x00 = MaxPooling1D(2)(o00)

        """
        x(1, 0) --> x2
        """
        x10 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x00)
        x10 = BatchNormalization()(x10)
        x10 = SpatialDropout1D(0.3)(x10)
        x10 = Activation('relu')(x10)
        x10 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x10)
        x10 = BatchNormalization()(x10)
        x10 = SpatialDropout1D(0.3)(x10)
        o10 = Activation('relu')(x10)
        x10 = MaxPooling1D(2)(o10)

        """
        x(0, 1) = [o00, up_o10] --> x1
        """
        up_o10 = Conv1DTranspose(self.num_filter, self.kernel_size - 1, strides=self.strides, padding='same')(o10)
        x01 = concatenate([up_o10, o00])
        x01 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x01)
        x01 = BatchNormalization()(x01)
        x01 = SpatialDropout1D(0.3)(x01)
        x01 = Activation('relu')(x01)
        x01 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x01)
        x01 = BatchNormalization()(x01)
        x01 = SpatialDropout1D(0.3)(x01)
        x01 = Activation('relu')(x01)

        """
        x(2, 0) --> x4
        """
        x20 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x10)
        x20 = BatchNormalization()(x20)
        x20 = SpatialDropout1D(0.3)(x20)
        x20 = Activation('relu')(x20)
        x20 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x20)
        x20 = BatchNormalization()(x20)
        x20 = SpatialDropout1D(0.3)(x20)
        o20 = Activation('relu')(x20)
        x20 = MaxPooling1D(2)(o20)

        """
        x(1, 1) = [o10, up_o20] --> x2
        """
        up_o20 = Conv1DTranspose(self.num_filter * 2, self.kernel_size - 1, strides=self.strides, padding='same')(o20)
        x11 = concatenate([up_o20, o10])
        x11 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x11)
        x11 = BatchNormalization()(x11)
        x11 = SpatialDropout1D(0.3)(x11)
        x11 = Activation('relu')(x11)
        x11 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x11)
        x11 = BatchNormalization()(x11)
        x11 = SpatialDropout1D(0.3)(x11)
        x11 = Activation('relu')(x11)

        """
        x(0, 2) = [x01, up_x11] --> x1
        """
        up_x11 = Conv1DTranspose(self.num_filter, self.kernel_size - 1, strides=self.strides, padding='same')(x11)
        x02 = concatenate([up_x11, x01])
        x02 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x02)
        x02 = BatchNormalization()(x02)
        x02 = SpatialDropout1D(0.3)(x02)
        x02 = Activation('relu')(x02)
        x02 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x02)
        x02 = BatchNormalization()(x02)
        x02 = SpatialDropout1D(0.3)(x02)
        x02 = Activation('relu')(x02)

        """
        x(3, 0)--> x8
        """
        x30 = Conv1D(self.num_filter * 8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x20)
        x30 = BatchNormalization()(x30)
        x30 = SpatialDropout1D(0.3)(x30)
        x30 = Activation('relu')(x30)
        x30 = Conv1D(self.num_filter * 8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x30)
        x30 = BatchNormalization()(x30)
        x30 = SpatialDropout1D(0.3)(x30)
        o30 = Activation('relu')(x30)
        x30 = MaxPooling1D(2)(o30)

        """
        x(2, 1) = [o20, up_o30] --> x4
        """
        up_o30 = Conv1DTranspose(self.num_filter * 4, self.kernel_size - 1, strides=self.strides, padding='same')(o30)
        x21 = concatenate([up_o30, o20])
        x21 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x21)
        x21 = BatchNormalization()(x21)
        x21 = SpatialDropout1D(0.3)(x21)
        x21 = Activation('relu')(x21)
        x21 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x21)
        x21 = BatchNormalization()(x21)
        x21 = SpatialDropout1D(0.3)(x21)
        x21 = Activation('relu')(x21)

        """
        x(1, 2) = [x11, up_x21] --> x2
        """
        up_x21 = Conv1DTranspose(self.num_filter * 2, self.kernel_size - 1, strides=self.strides, padding='same')(x21)
        x12 = concatenate([up_x21, x11])
        x12 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x12)
        x12 = BatchNormalization()(x12)
        x12 = SpatialDropout1D(0.3)(x12)
        x12 = Activation('relu')(x12)
        x12 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x12)
        x12 = BatchNormalization()(x12)
        x12 = SpatialDropout1D(0.3)(x12)
        x12 = Activation('relu')(x12)

        """
        x(0, 3) = [x02, up_x12] --> x1
        """
        up_x12 = Conv1DTranspose(self.num_filter, self.kernel_size - 1, strides=self.strides, padding='same')(x12)
        x03 = concatenate([up_x12, x02])
        x03 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x03)
        x03 = BatchNormalization()(x03)
        x03 = SpatialDropout1D(0.3)(x03)
        x03 = Activation('relu')(x03)
        x03 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x03)
        x03 = BatchNormalization()(x03)
        x03 = SpatialDropout1D(0.3)(x03)
        x03 = Activation('relu')(x03)

        """
        x(4, 0)--> x16
        """
        x40 = Conv1D(self.num_filter * 16, self.kernel_size, kernel_initializer='he_normal', padding='same')(x30)
        x40 = BatchNormalization()(x40)
        x40 = SpatialDropout1D(0.3)(x40)
        x40 = Activation('relu')(x40)
        x40 = Conv1D(self.num_filter * 16, self.kernel_size, kernel_initializer='he_normal', padding='same')(x40)
        x40 = BatchNormalization()(x40)
        x40 = SpatialDropout1D(0.3)(x40)
        x40 = Activation('relu')(x40)

        # Decoder
        """
        x(3, 1) = [o30, up_x40] --> x8
        """
        up_x40 = Conv1DTranspose(self.num_filter * 8, self.kernel_size - 1, strides=self.strides, padding='same')(x40)
        x31 = concatenate([up_x40, o30])
        x31 = Conv1D(self.num_filter * 8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x31)
        x31 = BatchNormalization()(x31)
        x31 = SpatialDropout1D(0.3)(x31)
        x31 = Activation('relu')(x31)
        x31 = Conv1D(self.num_filter * 8, self.kernel_size, kernel_initializer='he_normal', padding='same')(x31)
        x31 = BatchNormalization()(x31)
        x31 = SpatialDropout1D(0.3)(x31)
        x31 = Activation('relu')(x31)

        """
        x(2, 2) = [up_x31, x21] --> x4
        """
        up_x31 = Conv1DTranspose(self.num_filter * 4, self.kernel_size - 1, strides=self.strides, padding='same')(x31)
        x22 = concatenate([up_x31, x21])
        x22 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x22)
        x22 = BatchNormalization()(x22)
        x22 = SpatialDropout1D(0.3)(x22)
        x22 = Activation('relu')(x22)
        x22 = Conv1D(self.num_filter * 4, self.kernel_size, kernel_initializer='he_normal', padding='same')(x22)
        x22 = BatchNormalization()(x22)
        x22 = SpatialDropout1D(0.3)(x22)
        x22 = Activation('relu')(x22)

        """
        x(1, 3) = [up_x22, x12] --> x2
        """
        x22 = Conv1DTranspose(self.num_filter * 2, self.kernel_size - 1, strides=self.strides, padding='same')(x22)
        x13 = concatenate([x22, x12])
        x13 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x13)
        x13 = BatchNormalization()(x13)
        x13 = SpatialDropout1D(0.3)(x13)
        x13 = Activation('relu')(x13)
        x13 = Conv1D(self.num_filter * 2, self.kernel_size, kernel_initializer='he_normal', padding='same')(x13)
        x13 = BatchNormalization()(x13)
        x13 = SpatialDropout1D(0.3)(x13)
        x13 = Activation('relu')(x13)

        # x0, 1
        """
        x(0, 4) = [up_x13, x03] --> x1
        """
        up_x13 = Conv1DTranspose(self.num_filter, self.kernel_size - 1, strides=self.strides, padding='same')(x13)
        x04 = concatenate([up_x13, x03])
        x04 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x04)
        x04 = BatchNormalization()(x04)
        x04 = SpatialDropout1D(0.3)(x04)
        x04 = Activation('relu')(x04)
        x04 = Conv1D(self.num_filter, self.kernel_size, kernel_initializer='he_normal', padding='same')(x04)
        x04 = BatchNormalization()(x04)
        x04 = SpatialDropout1D(0.3)(x04)
        x04 = Activation('relu')(x04)

        x04 = Conv1D(NUM_COMPONENTS_CLASSES, self.last_kernel_size, activation='softmax', kernel_initializer='he_normal')(x04)
        outputs = Cropping1D(cropping=self.pad_and_crop_size)(x04)

        model = Model(inputs=inputs, outputs=outputs)
        return model




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
