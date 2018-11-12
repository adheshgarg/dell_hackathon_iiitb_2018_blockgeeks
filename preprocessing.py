
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

def process(df):
    
    from sklearn.preprocessing import Imputer
    df['lead_time'] = Imputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    df = df.dropna()
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        df[col] = Imputer(missing_values=-99).fit_transform(
                                    df[col].values.reshape(-1, 1))
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'xcs_pro']:
        df[col] = (df[col] == 'Yes').astype(int)
     
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    return df



cols=range(0,23)
train = pd.read_csv('data/kaggle/Kaggle_Training_Dataset_v2.csv', usecols=cols)
test = pd.read_csv('data/kaggle/Kaggle_Test_Dataset_v2.csv', usecols=cols)
df = process(train.append(test))

sample = df.sample(5000)
X_sample = sample.drop('xcs_pro',axis=1).values
y_sample = sample['xcs_pro'].values

df.round(6).to_csv('data/kaggle.csv',index=False)