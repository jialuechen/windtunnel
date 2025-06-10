import pandas as pd
import numpy as np

def load_dataset(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, parse_dates=['datetime'])
    return df.sort_values(['symbol','datetime']).reset_index(drop=True)

def process_data(df, history_len, pred_len):
    feature_cols = [c for c in df.columns if c not in ['symbol','datetime']]
    symbols = df['symbol'].unique()
    data = {s: df[df['symbol']==s][feature_cols].values for s in symbols}
    sequences = []
    length = len(next(iter(data.values())))
    for t in range(history_len, length-pred_len+1):
        cond = np.stack([data[s][t-history_len:t] for s in symbols],axis=0)
        target = data[symbols[0]][t:t+pred_len]
        mask = np.ones((pred_len,),dtype=bool)
        sequences.append((cond,target,mask))
    return sequences