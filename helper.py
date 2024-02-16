# Project:  CensorshipDissent
# Filename: helper.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

import numpy as np
import os
import os.path as osp


def dump_np(fname, arr):
    """
    Writes a numpy array to file.
    """
    os.makedirs(osp.split(fname)[0], exist_ok=True)
    with open(fname, 'wb') as f:
        np.save(f, arr)


def load_np(fname):
    """
    Reads a numpy array from file.
    """
    with open(fname, 'rb') as f:
        return np.load(f)
