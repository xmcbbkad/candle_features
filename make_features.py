# coding: utf-8
import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_LIST = [
    't0_close',
    't0_close_ratio_last',
    't1_open',
    't1_open_ratio_last',
    't1_high',
    't1_high_ratio_last',
    't1_low',
    't1_low_ratio_last',
    't1_close',
    't1_close_ratio_last',
    't2_open',
    't2_open_ratio_last',
    't2_high',
    't2_high_ratio_last',
    't2_low',
    't2_low_ratio_last',
    't2_close',
    't2_close_ratio_last',
    't3_open',
    't3_open_ratio_last',
    't3_high',
    't3_high_ratio_last',
    't3_low',
    't3_low_ratio_last',
    't3_close',
    't3_close_ratio_last',
    't4_open',
    't4_open_ratio_last',
    't4_high',
    't4_high_ratio_last',
    't4_low',
    't4_low_ratio_last',
    't4_close',
    't4_close_ratio_last',
    't5_open',
    't5_open_ratio_last',
    't5_high',
    't5_high_ratio_last',
    't5_low',
    't5_low_ratio_last',
    't5_close',
    't5_close_ratio_last',
    't6_open',
    't6_open_ratio_last',
    't6_high',
    't6_high_ratio_last',
    't6_low',
    't6_low_ratio_last',
    't6_close',
    't6_close_ratio_last',
    't7_open',
    't7_open_ratio_last',
    't7_high',
    't7_high_ratio_last',
    't7_low',
    't7_low_ratio_last',
    't7_close',
    't7_close_ratio_last',
    't8_open',
    't8_open_ratio_last',
    't8_high',
    't8_high_ratio_last',
    't8_low',
    't8_low_ratio_last',
    't8_close',
    't8_close_ratio_last',
    't9_open',
    't9_open_ratio_last',
    't9_high',
    't9_high_ratio_last',
    't9_low',
    't9_low_ratio_last',
    't9_close',
    't9_close_ratio_last',
]

def t0_close(df, row_id):
    return df.loc[row_id]['close']

def t0_close_ratio_last(df, row_id):
    return df.loc[row_id]['close']/df.loc[row_id-1]['close']

def t1_open(df, row_id):
    return df.loc[row_id-1]['open']

def t1_open_ratio_last(df, row_id):
    return df.loc[row_id-1]['open']/df.loc[row_id-2]['close']

def t1_high(df, row_id):
    return df.loc[row_id-1]['high']

def t1_high_ratio_last(df, row_id):
    return df.loc[row_id-1]['high']/df.loc[row_id-2]['close']

def t1_low(df, row_id):
    return df.loc[row_id-1]['low']

def t1_low_ratio_last(df, row_id):
    return df.loc[row_id-1]['low']/df.loc[row_id-2]['close']

def t1_close(df, row_id):
    return df.loc[row_id-1]['close']

def t1_close_ratio_last(df, row_id):
    return df.loc[row_id-1]['close']/df.loc[row_id-2]['close']

def t2_open(df, row_id):
    return df.loc[row_id-2]['open']

def t2_open_ratio_last(df, row_id):
    return df.loc[row_id-2]['open']/df.loc[row_id-3]['close']

def t2_high(df, row_id):
    return df.loc[row_id-2]['high']

def t2_high_ratio_last(df, row_id):
    return df.loc[row_id-2]['high']/df.loc[row_id-3]['close']

def t2_low(df, row_id):
    return df.loc[row_id-2]['low']

def t2_low_ratio_last(df, row_id):
    return df.loc[row_id-2]['low']/df.loc[row_id-3]['close']

def t2_close(df, row_id):
    return df.loc[row_id-2]['close']

def t2_close_ratio_last(df, row_id):
    return df.loc[row_id-2]['close']/df.loc[row_id-3]['close']

def t3_open(df, row_id):
    return df.loc[row_id-3]['open']

def t3_open_ratio_last(df, row_id):
    return df.loc[row_id-3]['open']/df.loc[row_id-4]['close']

def t3_high(df, row_id):
    return df.loc[row_id-3]['high']

def t3_high_ratio_last(df, row_id):
    return df.loc[row_id-3]['high']/df.loc[row_id-4]['close']

def t3_low(df, row_id):
    return df.loc[row_id-3]['low']

def t3_low_ratio_last(df, row_id):
    return df.loc[row_id-3]['low']/df.loc[row_id-4]['close']

def t3_close(df, row_id):
    return df.loc[row_id-3]['close']

def t3_close_ratio_last(df, row_id):
    return df.loc[row_id-3]['close']/df.loc[row_id-4]['close']

def t4_open(df, row_id):
    return df.loc[row_id-4]['open']

def t4_open_ratio_last(df, row_id):
    return df.loc[row_id-4]['open']/df.loc[row_id-5]['close']

def t4_high(df, row_id):
    return df.loc[row_id-4]['high']

def t4_high_ratio_last(df, row_id):
    return df.loc[row_id-4]['high']/df.loc[row_id-5]['close']

def t4_low(df, row_id):
    return df.loc[row_id-4]['low']

def t4_low_ratio_last(df, row_id):
    return df.loc[row_id-4]['low']/df.loc[row_id-5]['close']

def t4_close(df, row_id):
    return df.loc[row_id-4]['close']

def t4_close_ratio_last(df, row_id):
    return df.loc[row_id-4]['close']/df.loc[row_id-5]['close']

def t5_open(df, row_id):
    return df.loc[row_id-5]['open']

def t5_open_ratio_last(df, row_id):
    return df.loc[row_id-5]['open']/df.loc[row_id-6]['close']

def t5_high(df, row_id):
    return df.loc[row_id-5]['high']

def t5_high_ratio_last(df, row_id):
    return df.loc[row_id-5]['high']/df.loc[row_id-6]['close']

def t5_low(df, row_id):
    return df.loc[row_id-5]['low']

def t5_low_ratio_last(df, row_id):
    return df.loc[row_id-5]['low']/df.loc[row_id-6]['close']

def t5_close(df, row_id):
    return df.loc[row_id-5]['close']

def t5_close_ratio_last(df, row_id):
    return df.loc[row_id-5]['close']/df.loc[row_id-6]['close']

def t6_open(df, row_id):
    return df.loc[row_id-6]['open']

def t6_open_ratio_last(df, row_id):
    return df.loc[row_id-6]['open']/df.loc[row_id-7]['close']

def t6_high(df, row_id):
    return df.loc[row_id-6]['high']

def t6_high_ratio_last(df, row_id):
    return df.loc[row_id-6]['high']/df.loc[row_id-7]['close']

def t6_low(df, row_id):
    return df.loc[row_id-6]['low']

def t6_low_ratio_last(df, row_id):
    return df.loc[row_id-6]['low']/df.loc[row_id-7]['close']

def t6_close(df, row_id):
    return df.loc[row_id-6]['close']

def t6_close_ratio_last(df, row_id):
    return df.loc[row_id-6]['close']/df.loc[row_id-7]['close']

def t7_open(df, row_id):
    return df.loc[row_id-7]['open']

def t7_open_ratio_last(df, row_id):
    return df.loc[row_id-7]['open']/df.loc[row_id-8]['close']

def t7_high(df, row_id):
    return df.loc[row_id-7]['high']

def t7_high_ratio_last(df, row_id):
    return df.loc[row_id-7]['high']/df.loc[row_id-8]['close']

def t7_low(df, row_id):
    return df.loc[row_id-7]['low']

def t7_low_ratio_last(df, row_id):
    return df.loc[row_id-7]['low']/df.loc[row_id-8]['close']

def t7_close(df, row_id):
    return df.loc[row_id-7]['close']

def t7_close_ratio_last(df, row_id):
    return df.loc[row_id-7]['close']/df.loc[row_id-8]['close']

def t8_open(df, row_id):
    return df.loc[row_id-8]['open']

def t8_open_ratio_last(df, row_id):
    return df.loc[row_id-8]['open']/df.loc[row_id-9]['close']

def t8_high(df, row_id):
    return df.loc[row_id-8]['high']

def t8_high_ratio_last(df, row_id):
    return df.loc[row_id-8]['high']/df.loc[row_id-9]['close']

def t8_low(df, row_id):
    return df.loc[row_id-8]['low']

def t8_low_ratio_last(df, row_id):
    return df.loc[row_id-8]['low']/df.loc[row_id-9]['close']

def t8_close(df, row_id):
    return df.loc[row_id-8]['close']

def t8_close_ratio_last(df, row_id):
    return df.loc[row_id-8]['close']/df.loc[row_id-9]['close']

def t9_open(df, row_id):
    return df.loc[row_id-9]['open']

def t9_open_ratio_last(df, row_id):
    return df.loc[row_id-9]['open']/df.loc[row_id-10]['close']

def t9_high(df, row_id):
    return df.loc[row_id-9]['high']

def t9_high_ratio_last(df, row_id):
    return df.loc[row_id-9]['high']/df.loc[row_id-10]['close']

def t9_low(df, row_id):
    return df.loc[row_id-9]['low']

def t9_low_ratio_last(df, row_id):
    return df.loc[row_id-9]['low']/df.loc[row_id-10]['close']

def t9_close(df, row_id):
    return df.loc[row_id-9]['close']

def t9_close_ratio_last(df, row_id):
    return df.loc[row_id-9]['close']/df.loc[row_id-10]['close']








def make_feature_file(input_file, output_file):
    df_5mins = pd.read_csv(input_file, sep='\t')
    df_features = pd.DataFrame(data=None, columns=FEATURE_LIST)

    for row_id in range(10, df_5mins.shape[0]):
        features = [eval(func)(df_5mins, row_id)  for func in FEATURE_LIST ] 
        df_features.loc[len(df_features)] = features
    df_features.to_csv(output_file, sep='\t')

def make_feature_dir(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        logger.info(filename)
        make_feature_file(input_file=os.path.join(input_dir, filename), output_file=os.path.join(output_dir, filename))

if __name__ == '__main__':
    make_feature_dir(input_dir='/Users/xiaokunfan/code/data/TSLA_5mins_ohlc', output_dir='/Users/xiaokunfan/code/data/TSLA_5mins_features')
    #make_feature_file(input_file='./2022-08-30_TSLA_tiger_5mins.csv', output_file='./2022-08-30_TSLA_tiger_5mins_features.csv')
