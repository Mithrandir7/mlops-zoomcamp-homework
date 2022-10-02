#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import sys
import os

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

year = int(sys.argv[1]) # 2021
month = int(sys.argv[2]) # 2

### Pasar a buckets de s3 ###
input_file = f's3://nyc-tlc-mio/trip data/fhv_tripdata_2021-02.parquet'
output_file = f's3://nyc-tlc-mio/duration-prediction/taxi_type=fhv/year={year:04d}/month={month:02d}/predict.parquet'

df = read_data(input_file)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print('predicted mean duration: ', np.mean(y_pred))

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = df[['ride_id', 'duration']].copy()

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
