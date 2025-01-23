import argparse
import logging
import sys
import time

from test.test_atrial import TestAtrial
#from test.test_beat_analysis import TestBeatAnalysis
#from test.test_delineate import TestDeliniate
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
            TestDeliniate().run()
        elif args.m == 'atrial':
            TestAtrial().run()

        elif args.m == 'classify':
            TestBeatAnalysis().run()
        else:
            print("The Type(--m) is not Exist")
            sys.exit(1)
    else:
        #TestDeliniate().run()
        #TestBeatAnalysis().run()
        TestAtrial().run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"소요 시간: {elapsed_time:.6f}초")
