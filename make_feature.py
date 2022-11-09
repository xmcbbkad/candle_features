# coding: utf-8
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_LIST = [
    't0_close',
    't1_open',
    't1_high',
    't1_low',
    't1_close',
    't2_open',
    't2_high',
    't2_low',
    't2_close',
    't3_open',
    't3_high',
    't3_low',
    't3_close',
    't4_open',
    't4_high',
    't4_low',
    't4_close',
]

def make_feature(input_file):
    df_5mins = pd.read_csv(input_file, sep='\t')
    
    

    df_featue = pd.DataFrame(data=None, columns=FEATURE_LIST)






