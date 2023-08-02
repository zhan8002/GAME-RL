import glob
import os.path
import re
import sys
import numpy as np

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
SAMPLE_PATH = os.path.join(module_path, "samples")
sequence_length = 200

def fetch_file(sample_path):
    apisequence = np.load(sample_path)
    return apisequence


def get_available_sha256():
    samplelist = []
    for fp in glob.glob(os.path.join(SAMPLE_PATH, "*")):
        fn = os.path.split(fp)[-1]
        samplelist.append(fn)
    # no files found in SAMLPE_PATH with sha256 names
    assert len(samplelist) > 0
    return samplelist

def cal_payloads_rate(sequence):
    payloads_id = np.where(sequence[:sequence_length, 8] != 100)[0]

    num_payloads = len(payloads_id)

    rate = num_payloads/sequence_length

    return rate