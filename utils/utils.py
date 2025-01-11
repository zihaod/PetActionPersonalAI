# -*- coding: utf-8 -*-

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import sklearn
import os
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def read_json_data(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data
