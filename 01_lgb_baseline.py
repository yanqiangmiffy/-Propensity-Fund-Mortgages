#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: 01_lgb_baseline.py
@time: 2019-05-17 10:54
@description:
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from gen_stat_feat import load_data
from sklearn.metrics import f1_score


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


# ----------加载数据------------
train, test, no_features, features = load_data()
#
X = train[features].values
y = train['RESULT'].values
test_data = test[features].values

# ----------lgb------------
valid_id = []  # 训练集
valid_index = []
valid_list = []
valid_pred_list = []

res_list = []  # 结果
scores_list = []

print("start：********************************")
start = time.time()
# kf = KFold(n_splits=5, shuffle=True, random_state=2019)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]
    valid_id.extend(list(train.Unique_ID[test_index].values))
    # 数据结构
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': {'auc'},
    #     'num_leaves': 900,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.6,
    #     'bagging_fraction': 0.7,
    #     'bagging_freq': 5,
    #     'reg_alpha': 0.3,
    #     'reg_lambda': 0.3,
    #     # 'min_data_in_leaf': 18,
    #     'min_sum_hessian_in_leaf': 0.001,
    # }
    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        # 'max_depth': 10,
        # 'min_child_weight': 6,
        'num_leaves': 90,
        'learning_rate': 0.02,  # 0.05
        # 'feature_fraction': 0.7,
        # 'bagging_fraction': 0.7,
        # 'bagging_freq': 5,
        # 'lambda_l1':0.25,
        # 'lambda_l2':0.5,
        # 'scale_pos_weight':10.0/1.0, #14309.0 / 691.0, #不设置
        # 'num_threads':4,
    }

    print('................Start training {} fold..........................'.format(k + 1))
    # train
    evals_result = {}

    lgb_clf = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        feval=lgb_f1_score,
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        feature_name=features,
                        evals_result=evals_result)
    lgb.plot_metric(evals_result, metric='f1')
    plt.show()
    # 验证集测试
    print('................Start predict .........................')
    valid_pred = lgb_clf.predict(x_valid)
    valid_index.extend(list(test_index))
    valid_list.extend(list(y_valid))
    valid_pred_list.extend(list(valid_pred))
    score = roc_auc_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = lgb_clf.predict(test_data)
    res_list.append(pred)

lgb.plot_importance(lgb_clf, max_num_features=20)
plt.show()

### 特征选择
df = pd.DataFrame(train[features].columns.tolist(), columns=['feature'])
df['importance'] = list(lgb_clf.feature_importance())  # 特征分数
df = df.sort_values(by='importance', ascending=False)
print(list(df['feature'].values))
# 特征排序
df.to_csv("output/feature_score.csv", index=None)  # 保存分数

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 线下平均分数
mean_score = np.mean(scores_list)
print("lgb mean score:", mean_score)
filepath = 'output/lgb_' + str(mean_score) + '.csv'  #

# 提交结果 5折平均
print("lgb 提交结果...")
res = np.array(res_list)
r = res.mean(axis=0)
result = pd.read_csv('input/CAX_MortgageModeling_SubmissionFormat.csv')
result['Result_Predicted'] = r
result['Result_Predicted'] = result['Result_Predicted'].apply(lambda x: 'FUNDED' if x > 0.5 else 'NOT FUNDED')
result.to_csv(filepath, index=None, sep=",")
# # 训练集结果
# print("训练集结果")
# raw_df = pd.read_csv('input/CAX_MortgageModeling_Train.csv')
# valid_df = pd.DataFrame()
# valid_df['Unique_ID'] = valid_id
# valid_df['pred_tradeMoney'] = valid_pred_list
# full_df = pd.merge(raw_df, valid_df, on="ID")
# full_df.to_csv('output/lgb_df.csv', index=None)
