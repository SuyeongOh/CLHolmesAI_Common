import argparse
import json
import os
import sys
import time

import numpy as np
import wfdb
from sympy.codegen.ast import continue_

from test import test_delineate, test_beat_analysis, test_atrial
from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader
from test.test_atrial import TestAtrial

ROOT_DATA_PATH = 'data/data'
BASE_DATA_PATH = 'data/data/parsing'
DATA_LABEL_ARRHYTHMIA = 'arrhythmia'
DATA_LABEL_STRESS = 'stress'

TAG_LEAD2 = 'MLII'

# slice 1s waveform for
if __name__ == "__main__":
    MitArrhythmiaLoader().load()

    parser = argparse.ArgumentParser(description="Test Type Parser")

    parser.add_argument("--m", type=str, required=False, help="Input Test Module(delineate/atrial/classify")

    args = parser.parse_args()

    start_time = time.time()

    if args.m:
        if args.m == 'delineate':
            NotImplemented
            #test_delineate.run(np_data=np_data, gt_data=frame_json_data, raw_signal=raw_record, raw_fs=raw_record_fs)
        elif args.m == 'atrial':
            NotImplemented
            #test_beat_analysis.run(np_data=np_data)

        elif args.m == 'classify':
            TestAtrial().run()
        else:
            print("The Type(--m) is not Exist")
            sys.exit(1)
    else:
        #test_delineate.run(np_data=np_data, gt_data=frame_json_data, raw_signal=raw_record, raw_fs=raw_record_fs)
        #test_beat_analysis.run(np_data=np_data)
        TestAtrial().run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"소요 시간: {elapsed_time:.6f}초")
