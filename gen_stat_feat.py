#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: gen_stat_feat.py
@time: 2019-05-17 10:54
@description: 特征生成
"""
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('input/CAX_MortgageModeling_Train.csv')
df_test = pd.read_csv('input/CAX_MortgageModeling_Test.csv')

df_train['RESULT'] = df_train['RESULT'].apply(lambda x: 1 if x == 'FUNDED' else 0)
df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)
print(df.shape)

numerical_cols = ['PROPERTY VALUE', 'MORTGAGE PAYMENT', 'GDS',
                  'LTV', 'TDS', 'AMORTIZATION', 'MORTGAGE AMOUNT',
                  'RATE', 'TERM', 'INCOME', 'CREDIT SCORE']
categorical_cols = ['MORTGAGE PURPOSE', 'PAYMENT FREQUENCY',
                    'PROPERTY TYPE', 'AGE RANGE', 'GENDER',
                    'INCOME TYPE', 'NAICS CODE']

print(len(numerical_cols) + len(categorical_cols))
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=categorical_cols)

mean_cols = ['FSA']
for col in mean_cols:
    print(df[col].value_counts())
    buildYear_nums = dict(df[col].value_counts())
    df[col + '_nums'] = df[col].apply(lambda x: buildYear_nums[x])
    for fea in tqdm(numerical_cols):
        grouped_df = df.groupby(col).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
        prefix = col + '_'
        grouped_df.columns = [prefix + '_'.join(col).strip() for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        # print(grouped_df)
        df = pd.merge(df, grouped_df, on=col, how='left')
# print(df.shape)

# # 生成数据
no_features = ['Unique_ID', 'MORTGAGE NUMBER', 'FSA', 'RESULT'] + categorical_cols
features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(df_train)], df[len(df_train):]
df.head(100).to_csv('input/df.csv', index=False)

print(train.shape, test.shape)


def load_data():
    return train, test, no_features, features
