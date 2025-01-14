import argparse
import json
import logging
import os
import sys
import time

from test.test_atrial import TestAtrial
from utils import log_utils

logger = log_utils.getCustomLogger(__name__)
logger.setLevel(logging.DEBUG)

# slice 1s waveform for
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Type Parser")

    parser.add_argument("--m", type=str, required=False, help="Input Test Module(delineate/atrial/classify")

    args = parser.parse_args()

    start_time = time.time()

    if args.m:
        if args.m == 'delineate':
            NotImplemented
            #test_delineate.run(np_data=np_data, gt_data=frame_json_data, raw_signal=raw_record, raw_fs=raw_record_fs)
        elif args.m == 'atrial':
            TestAtrial().run()

        elif args.m == 'classify':
            NotImplemented
            #test_beat_analysis.run(np_data=np_data)
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
