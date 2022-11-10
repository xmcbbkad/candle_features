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

def t0_close(df, row_id):
    return df.loc[row_id]['close']

def t1_open(df, row_id):
    return df.loc[row_id-1]['open']

def t1_high(df, row_id):
    return df.loc[row_id-1]['high']

def t1_low(df, row_id):
    return df.loc[row_id-1]['low']

def t1_close(df, row_id):
    return df.loc[row_id-1]['close']

def t2_open(df, row_id):
    return df.loc[row_id-2]['open']

def t2_high(df, row_id):
    return df.loc[row_id-2]['high']

def t2_low(df, row_id):
    return df.loc[row_id-2]['low']

def t2_close(df, row_id):
    return df.loc[row_id-2]['close']

def t3_open(df, row_id):
    return df.loc[row_id-3]['open']

def t3_high(df, row_id):
    return df.loc[row_id-3]['high']

def t3_low(df, row_id):
    return df.loc[row_id-3]['low']

def t3_close(df, row_id):
    return df.loc[row_id-3]['close']

def t4_open(df, row_id):
    return df.loc[row_id-4]['open']

def t4_high(df, row_id):
    return df.loc[row_id-4]['high']

def t4_low(df, row_id):
    return df.loc[row_id-4]['low']

def t4_close(df, row_id):
    return df.loc[row_id-4]['close']




def make_feature(input_file, output_file):
    df_5mins = pd.read_csv(input_file, sep='\t')
    df_features = pd.DataFrame(data=None, columns=FEATURE_LIST)

    for row_id in range(10, df_5mins.shape[0]):
        features = [eval(func)(df_5mins, row_id)  for func in FEATURE_LIST ] 
        df_features.loc[len(df_features)] = features
    df_features.to_csv(output_file, sep='\t')

if __name__ == '__main__':
    make_feature(input_file='./2022-08-30_TSLA_tiger_5mins.csv', output_file='./2022-08-30_TSLA_tiger_5mins_features.csv')
