# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def fvecs_read_norm(fname):
    tmp = ivecs_read(fname).view('float32')
    return normalize(tmp, axis=1, norm='max')

def ivecs_read_norm(fname):
    tmp = ivecs_read(fname)
    return normalize(tmp, axis=1, norm='max')


def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    xb = fvecs_read("sift1M/sift_base.fvecs")
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

