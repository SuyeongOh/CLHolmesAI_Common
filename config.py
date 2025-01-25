# DB code
RECORD_DURATION_DAYS = {"7": 7, "8": 14} # 공통 코드 [7]은 7일, [8]은 14일

DAILY_10SEC_COUNTS = int(24 * 60 * 60 / 10)

# Evironment
AI_CPU_CORE = 32
SEC_TO_MS = 1000
BATCH_SIZE = 512
FS_MITBIH = 360
FS = 100 # DY : 100 / DNN: 250
DURATION = 10

FILTERS_WIDTH = [0.2, 0.6] # unit: sec (s)
HIGHCUT1 = 100 # LPF highcut1: 100 Hz
HIGHCUT2 = 60 # LPF highcut2: 60 Hz
ORDER = 10 # LPF Butterworth filter order: 10

LEAD = 1

CROP_RATE = 2 / 5

NUM_BEAT_CLASSES = 4 # N, S, V, A

#INPUT_SHAPE = (FS, LEAD) # Heartbeat shape: (250, 1) = 1 s
INPUT_SHAPE = (FS//2, LEAD) # Heartbeat shape: (125, 1) = 0.5 s
CLASSES_INDEX1 = {
    0: 'N',
    1: 'S',
    2: 'V',
    3: 'A',
}

CLASSES_INDEX2 = {
    'N': 0,
    'S': 1,
    'V': 2,
    'A': 3,
}

CLASSES_INDEX3 = {
    1: 'N',
    2: 'S',
    3: 'V',
    4: 'A',
}

# 3.2. Heartbeat sub-classifier
N_SUBCLASSES = 20

# 4. ECG segmentation
NUM_COMPONENTS_CLASSES = 4 # P, QRS, T, Background

# 5. Anomaly Detector
ANOMALY_THRESHOLD = 0.50073004

# 6. Diagnostic Module
DIFF_CONDITION = 0.1 # Unit: sec (s) # 20 samples. 0.08 --> 0.1 로 수정

# Sinus_rhythm: SR, ST, SB, SA

HR_BRADY, HR_TACHY = 60, 100 # unit: BMP for SR, ST, SB
PAUSE_CONDITION = 2 # unit: sec (s) for SA

# AVB
AVB_1ST_DEGREE_CONDITION = 0.2 # unit: sec (s). 0.2 s = 50 samples

# AA
AF_THRESHOLD = 0.5454725

# 7. Report
ARRHYTHMIA_LIST = ["sinus_rhythm", "sinus_tachycardia", "sinus_bradycardia", "sinus_pause",
                   "pac_isolated", "pac_couplet", "pac_triplet", "pac_bigeminy", "pac_trigeminy",
                   "pvc_isolated", "pvc_couplet", "ventricular_tachycardia", "pvc_bigeminy", "pvc_trigeminy",
                   "atrial_fibrillation", "atrial_flutter",
                   "first_degree_atrioventricular_block",
                   "second_degree_atrioventricular_block_type_I",
                   "second_degree_atrioventricular_block_type_II",
                   "third_degree_atrioventricular_block"
                   ] # 20

ARRHYTHMIA_DICT = {"sinus_rhythm": 0,
                   "sinus_tachycardia": 1,
                   "sinus_bradycardia": 2,
                   "sinus_pause": 3,
                   "pac_isolated": 4,
                   "pac_couplet": 5,
                   "pac_triplet": 6,
                   "pac_bigeminy": 7,
                   "pac_trigeminy": 8,
                   "pvc_isolated": 9,
                   "pvc_couplet": 10,
                   "pvc_triplet": 11,
                   "ventricular_tachycardia": 11,
                   "pvc_bigeminy": 12,
                   "pvc_trigeminy": 13,
                   "atrial_fibrillation": 14,
                   "atrial_flutter": 15,
                   "first_degree_atrioventricular_block": 16,
                   "second_degree_atrioventricular_block_type_I": 17,
                   "second_degree_atrioventricular_block_type_II": 18,
                   "third_degree_atrioventricular_block": 19}

ARRHYTHMIA_REASON_DICT = {"sinus_rhythm": " ",
                          "sinus_tachycardia": "Heart rate is over 100 BPM. ",
                          "sinus_bradycardia": "Heart rate is below 60 BPM." ,
                          "sinus_pause": "T-P interval is over 2 seconds. ",
                          "pac_isolated": "Single occurrence of an ectopic beat (S). ",
                          "pac_couplet": "Two consecutive ectopic beats (S). ",
                          "pac_triplet": "Three or more consecutive premature beats (S). ",
                          "pac_bigeminy": "An ectopic occurs after every normal beat (S) ",
                          "pac_trigeminy": "An ectopic occurs after every two normal beats (S) ",
                          "pvc_isolated": "Single occurrence of an ectopic beat (V). ",
                          "pvc_couplet": "Two consecutive ectopic beats (V). ",
                          "pvc_triplet": "Three or more consecutive premature beats (V). ",
                          "ventricular_tachycardia": "Three or more consecutive premature beats (V). ",
                          "pvc_bigeminy": "An ectopic occurs after every normal beat (V) ",
                          "pvc_trigeminy": "An ectopic occurs after every two normal beats (V) ",
                          "atrial_fibrillation": "R-R intervals are irregular and the P-wave is missing. ",
                          "atrial_flutter": "F-wave is present. ",
                          "first_degree_atrioventricular_block": "P-R interval is over 0.2 seconds. ",
                          "second_degree_atrioventricular_block_type_I": "PR interval gradually increases and the QRS-complex is dropped. ",
                          "second_degree_atrioventricular_block_type_II": "PR interval is constant and QRS-complex is dropped. ",
                          "third_degree_atrioventricular_block": "AV dissociation. "}


ABNORMAL_QTc = {0: 0.43, 1: 0.45} # 0: 남자, 1: 여자 . 단위: sec (s)


AF_THRESHOLD = 0.5454725

##### BEAT MODEL DICT
BEAT_MODEL_DICT = {
    'sweep' : True,     # Ture : SWEEP, False: Specific model
    'model' : 'sweep',  # if Sweep flag is False plz note specific model name
    'model_list' : [
        'resxtranformer', # New Model
        'beat_classifer', # OLD Model
    ]
}
