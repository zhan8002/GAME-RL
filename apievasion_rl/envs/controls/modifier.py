import array
import json
import os
import random
import subprocess
import sys
import tempfile
from os import listdir
from os.path import isfile, join
import re

import numpy as np
import pandas as pd

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
APIHASH = np.load(os.path.join(module_path,"APIHash.npy"), allow_pickle=True).item()
APIINDEX = pd.read_csv(os.path.join(module_path,"api_312.csv"), header=None, index_col= 1, squeeze=True).to_dict()


sequence_length = 200


def modify_sample(sequence, actions, insert_ord):
    hook_index = actions[0] + 1
    pad_index = actions[1] + 1

    # padded location
    hook_api = APIINDEX[hook_index[0]]
    hook_hash = APIHASH[hook_api]

    hook_api_idx = []
    for i, api in enumerate(sequence.tolist()):
        if api[8] == 100 and (np.array(api[:8]) == hook_hash).all():
            hook_api_idx.append(i)

    # all api
    pad_api = APIINDEX[pad_index[0]] 


    pad_hash = APIHASH[pad_api]

    label_pad_hash = np.insert(pad_hash, 8, -100*insert_ord)

    padded = sequence
    hook_api_idx.reverse()

    for pad_loc in hook_api_idx:
        padded = np.insert(padded, pad_loc + 1, label_pad_hash, axis=0)

    return padded
