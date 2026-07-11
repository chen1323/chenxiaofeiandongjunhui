# -*- coding: utf-8 -*-
"""
Single-file version for reproducing get_prediction.py -> main_2018 only.
This file keeps the logic aligned with the original code path:
get_prediction.py -> FunctionPred.get_full_prediction_ensemble -> FunctionSet helpers

It only includes the functions that are actually used to generate:
    result/main_2018.csv
"""

import pandas as pd
import numpy as np
from itertools import product
import random

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ndcg_score,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV

from imblearn.ensemble import (
    RUSBoostClassifier,
    EasyEnsembleClassifier,
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
)
import xgboost as xgb

def get_metric_cla(y_test, y_pred, y_score):
    return {
        'accuracy': [accuracy_score(y_test, y_pred)],
        'precision': [precision_score(y_test, y_pred, average='weighted')],
        'recall': [recall_score(y_test, y_pred, average='weighted')],
        'f1': [f1_score(y_test, y_pred, average='weighted')],
        'AUC': [roc_auc_score(y_test, y_score)],
        'PR_AUC': [average_precision_score(y_test, y_score)],
        'NDCG': [ndcg_score([list(y_pred.astype(int))], [list(y_test.astype(int))], k=y_test.sum())],
    }


def train_and_predict(X_train, y_train, X_test, y_test, m_param, sample_weight=None):
    print(m_param)
    models = {
        # regression
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'GBDT': GradientBoostingRegressor(),
        'XGboost': xgb.XGBRegressor(),
        'RF': RandomForestRegressor(),
        # classification
        'GBDT_c': GradientBoostingClassifier(),
        'XGboost_c': xgb.XGBClassifier(),
        'Logit_c': CalibratedClassifierCV(LogisticRegression(class_weight='balanced', random_state=42), method='sigmoid', cv=5),
        'GaussianNB_c': GaussianNB(),
        'RF_c': RandomForestClassifier(),
        'RUSboost_c': RUSBoostClassifier(),
        'EasyEns_c': EasyEnsembleClassifier(),
        'BalancedBagging_c': BalancedBaggingClassifier(),
        'BalancedRF_c': BalancedRandomForestClassifier(),
    }

    model_name = list(m_param.keys())[0]
    hyperparameters = m_param[model_name]
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose one of {list(models.keys())}")

    selected_model = models[model_name]
    selected_model.set_params(**hyperparameters)
    if sample_weight is not None and model_name == 'Logit_c':
        selected_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        selected_model.fit(X_train, y_train)

    y_pred = selected_model.predict(X_test)
    y_score = pd.DataFrame(selected_model.predict_proba(X_test), index=X_test.index)[1]

    metric = get_metric_cla(y_test, y_pred, y_score)
    metric = pd.DataFrame(metric, index=[str(m_param)])
    return metric


def train_and_pred_value(X_train, y_train, X_test, y_test, m_param, sample_weight=None):
    random.seed(0)
    print(m_param)
    models = {
        # regression
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'GBDT': GradientBoostingRegressor(),
        'XGboost': xgb.XGBRegressor(),
        'RF': RandomForestRegressor(),
        # classification
        'GBDT_c': GradientBoostingClassifier(),
        'XGboost_c': xgb.XGBClassifier(),
        'Logit_c': CalibratedClassifierCV(LogisticRegression(class_weight='balanced', random_state=42), method='sigmoid', cv=5),
        'GaussianNB_c': GaussianNB(),
        'RF_c': RandomForestClassifier(),
        'RUSboost_c': RUSBoostClassifier(),
        'EasyEns_c': EasyEnsembleClassifier(),
        'BalancedBagging_c': BalancedBaggingClassifier(),
        'BalancedRF_c': BalancedRandomForestClassifier(),
    }

    model_name = list(m_param.keys())[0]
    hyperparameters = m_param[model_name]
    if model_name not in models:
        raise ValueError(f"Invalid model name. Choose one of {list(models.keys())}")

    selected_model = models[model_name]
    selected_model.set_params(**hyperparameters)
    if sample_weight is not None and model_name == 'Logit_c':
        selected_model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        selected_model.fit(X_train, y_train)

    y_pred = selected_model.predict(X_test)
    y_score = pd.DataFrame(selected_model.predict_proba(X_test))[1]
    return y_pred, y_score


def get_rank_decile(s):
    s_pos = s[s > 0]
    s_pos = pd.qcut(s_pos.rank(method="first"), q=10, labels=range(1, 11))
    s_zero = s[s == 0]
    s_zero = s_zero.replace(0, 11)
    res = pd.concat([s_pos, s_zero])
    res = res.loc[s.index]
    return res


def get_full_prediction_ensemble(data0, train_test, param_pool, drop10='original', robust='main'):
    """
    data0: input dataset including train and test set, predictors and target variables
    train_test:
        1: 2015-2016 train, 2017 test
        2: 2015-2017 train, 2018 test
        3: 2015-2016 train, 2018 test
    param_pool: hyperparameter pool in dict format
    drop10: whether drop samples placed top 10 schools
    robust: whether remove some predictors 
    """
    data = data0.copy()
    if drop10 == 'drop10':
        data = data[(data['Placerank'] > 10) | (data['Placerank'] == 0)]

    c = data.groupby('year').apply(lambda x: get_rank_decile(x.Placerank)).reset_index()[['ID', 'Placerank']].set_index('ID')
    data['Placerank'] = c

    if robust == 'removemanipulate':
        data = data.drop([
            'has PhD honor',
            'number of papers in progress', 'number of presentations',
            'number of teaching experiences',
            'number of reviewers',
            'number of working experiences', 'provide abstract',
        ], axis=1)

    if train_test == 1:
        train = data[data.year < 2017]
        test = data[data.year == 2017]
    if train_test == 2:
        train = data[data.year < 2018]
        test = data[data.year == 2018]
    if train_test == 3:
        train = data[data.year < 2017]
        test = data[data.year == 2018]

    if robust == 'embedding3072':
        X_trainB = train.loc[:, '0_cv':'3071_cv']
        X_testB = test.loc[:, '0_cv':'3071_cv']

        X_trainE = train.loc[:, '0_dt':'3071_dt']
        X_testE = test.loc[:, '0_dt':'3071_dt']
    else:
        X_trainB = train.loc[:, '0_cv':'255_cv']
        X_testB = test.loc[:, '0_cv':'255_cv']

        X_trainE = train.loc[:, '0_dt':'255_dt']
        X_testE = test.loc[:, '0_dt':'255_dt']

    X_trainC = train.loc[:, 'gender':'multi_language']
    X_testC = test.loc[:, 'gender':'multi_language']

    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']

    X_trainF = train[['Placerank']]
    X_testF = test[['Placerank']]

    y_train00 = train[[
        'pub_top_5pct', 'pub_top_10pct', 'pub_top_20pct', 'pub_top_30pct',
        'job_top_5pct', 'job_top_10pct', 'job_top_20pct', 'job_top_30pct',
        'pub_w_top_5pct', 'pub_w_top_10pct', 'pub_w_top_20pct', 'pub_w_top_30pct',
    ]]
    y_test = test[[
        'pub_top_5pct', 'pub_top_10pct', 'pub_top_20pct', 'pub_top_30pct',
        'job_top_5pct', 'job_top_10pct', 'job_top_20pct', 'job_top_30pct',
        'pub_w_top_5pct', 'pub_w_top_10pct', 'pub_w_top_20pct', 'pub_w_top_30pct',
    ]]

    m_param_list = []
    for m in list(param_pool.keys()):
        param_pro = product(*param_pool[m].values())
        for params in param_pro:
            params_dict = dict(zip(param_pool[m].keys(), params))
            m_param_list.append({m: params_dict})

    test_full = pd.DataFrame()

    for x in ['B', 'C', 'D', 'E', 'F']:
        if x == 'B':
            X_train00 = X_trainB.copy()
            X_test = X_testB.copy()
        if x == 'C':
            X_train00 = X_trainC.copy()
            X_test = X_testC.copy()
        if x == 'D':
            X_train00 = X_trainD.copy()
            X_test = X_testD.copy()
        if x == 'E':
            X_train00 = X_trainE.copy()
            X_test = X_testE.copy()
        if x == 'F':
            X_train00 = X_trainF.copy()
            X_test = X_testF.copy()

        X_train0, X_val, y_train0, y_val = train_test_split(
            X_train00, y_train00, test_size=0.2, random_state=42
        )

        for target in [
            'pub_top_5pct', 'pub_top_10pct', 'pub_top_20pct', 'pub_top_30pct',
            'job_top_5pct', 'job_top_10pct', 'job_top_20pct', 'job_top_30pct',
            'pub_w_top_5pct', 'pub_w_top_10pct', 'pub_w_top_20pct', 'pub_w_top_30pct',
        ]:
            if (drop10 == 'drop10') & ('job' in target):
                continue
            print(x, target)
            
            # IPW Calculation (Fixed for Perfect Separation & Trimming Bug)
            # 1. Use only structured covariates for PS model to avoid perfect separation
            X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1).loc[X_train0.index]
            D = data.loc[X_train0.index, 'research_oriented']
            propensity_model = LogisticRegression(random_state=42, max_iter=1000)
            propensity_model.fit(X_ps_train, D)
            prop_scores = propensity_model.predict_proba(X_ps_train)[:, 1]
            
            # 2. Stabilized weights
            p_marginal = D.mean()
            weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
            
            # 3. Weight Capping (99th percentile) instead of trimming
            p99 = np.percentile(weights, 99)
            weights = np.clip(weights, 0, p99)
            
            # 4. Print weight stats
            print(f"IPW Weight Stats [{x} {target}]: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}")
            
            X_train, y_train = X_train0.copy(), y_train0[target].copy()
            sample_weight = weights

            vali = pd.DataFrame()
            for m_param in m_param_list:
                temp, temp_score = train_and_pred_value(X_train, y_train, X_val, y_val[target], m_param, sample_weight=sample_weight)
                vali['pred_' + str(list(m_param.values())[0])] = temp
                vali['score_' + str(list(m_param.values())[0])] = temp_score
            vali['true'] = y_val[target].tolist()

            # choose best hyperparameter by validation AUC
            final = pd.DataFrame()
            for m_param in m_param_list:
                temp = train_and_predict(X_train, y_train, X_val, y_val[target], m_param, sample_weight=sample_weight)
                final = pd.concat([final, temp])

            final['model'] = [list(eval(i).keys())[0] for i in list(final.index)]
            best = final.groupby('model')['PR_AUC'].idxmax()
            best = best.dropna()
            print(best)

            # test predictions with best param
            best_res = pd.DataFrame(y_test[target])
            for i in range(len(best)):
                res, score = train_and_pred_value(
                    X_train, y_train, X_test, y_test[target], eval(best.iloc[i]), sample_weight=sample_weight
                )
                best_res[best.iloc[i]] = res
                best_res[best.iloc[i] + '_score'] = list(score)
            best_res.columns = ['y_vali', 'pred', 'score']
            best_res['x'] = x
            best_res['target'] = target

            test_full = pd.concat([test_full, best_res])

    pred_full = pd.DataFrame()
    for x in ['B', 'C', 'D', 'E', 'F']:
        for target in [
            'pub_top_5pct', 'pub_top_10pct', 'pub_top_20pct', 'pub_top_30pct',
            'job_top_5pct', 'job_top_10pct', 'job_top_20pct', 'job_top_30pct',
            'pub_w_top_5pct', 'pub_w_top_10pct', 'pub_w_top_20pct', 'pub_w_top_30pct',
        ]:
            if (drop10 == 'drop10') & ('job' in target):
                continue
            temp = test_full[(test_full.x == x) & (test_full.target == target)]
            temp = temp.rename({'y_vali': x + '_' + target, 'pred': x + '_' + target + '_pred', 'score': x + '_' + target + '_score'}, axis=1)
            pred_full = pred_full.join(temp.iloc[:, :3], how='outer')

    for x in ['BC', 'BCD', 'BCDE', 'BCDEF', 'CD', 'CDE', 'CDEF', 'CE', 'CEF']:
        for target in [
            'pub_top_5pct', 'pub_top_10pct', 'pub_top_20pct', 'pub_top_30pct',
            'job_top_5pct', 'job_top_10pct', 'job_top_20pct', 'job_top_30pct',
            'pub_w_top_5pct', 'pub_w_top_10pct', 'pub_w_top_20pct', 'pub_w_top_30pct',
        ]:
            if (drop10 == 'drop10') & ('job' in target):
                continue
            test_set = test_full[(test_full['x'].isin(list(x))) & (test_full['target'] == target)]
            test_set = test_set.reset_index()
            test_x = test_set[['ID', 'score', 'x']].pivot(index='ID', columns='x', values='score')
            pred = test_x.mean(axis=1)
            pred_full[x + '_' + target + '_score'] = pred
            pred = (pred >= 0.5).astype(int)
            pred_full[x + '_' + target + '_pred'] = pred
            pred_full[x + '_' + target] = test[target]

    return pred_full

def ndcg_at_k(y_true, y_score, k):
    temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    
    # DCG@K
    temp_pred = temp.sort_values('y_score', ascending=False).reset_index(drop=True)
    temp_pred['gain'] = 2 ** temp_pred['y_true'] - 1
    temp_pred['discount'] = np.log2(np.arange(len(temp_pred)) + 2)
    dcg_k = (temp_pred.loc[:k-1, 'gain'] / temp_pred.loc[:k-1, 'discount']).sum()
    
    # IDCG@K
    temp_ideal = temp.sort_values('y_true', ascending=False).reset_index(drop=True)
    temp_ideal['gain'] = 2 ** temp_ideal['y_true'] - 1
    temp_ideal['discount'] = np.log2(np.arange(len(temp_ideal)) + 2)
    idcg_k = (temp_ideal.loc[:k-1, 'gain'] / temp_ideal.loc[:k-1, 'discount']).sum()
    
    return dcg_k / idcg_k if idcg_k > 0 else 0.0
def precision_at_k(y_true, y_score, k):
    temp = pd.DataFrame([y_true,y_score]).T
    temp.columns = ['y_true','y_score']
    temp = temp.sort_values('y_score', ascending = False)
    TP = temp.iloc[:k,:].y_true.sum()
    
    # order = np.argsort(y_score)[::-1]
    # y_true = np.take(y_true, order[:k])
    # recall = np.sum(y_true) / np.sum(y_true != 0)
    return TP/k

def recall_at_k(y_true, y_score, k):
    temp = pd.DataFrame([y_true,y_score]).T
    temp.columns = ['y_true','y_score']
    temp = temp.sort_values('y_score', ascending = False)
    TP = temp.iloc[:k,:].y_true.sum()
    
    # order = np.argsort(y_score)[::-1]
    # y_true = np.take(y_true, order[:k])
    # recall = np.sum(y_true) / np.sum(y_true != 0)
    return TP/y_true.sum()

def get_metric_test(y_true, y_pred, y_score):
    
    return pd.Series({
            'true_pos': y_true.mean(),
            'accuracy':accuracy_score(y_true, y_pred),
            'precision':precision_score(y_true, y_pred),#, average='weighted'
            'recall':recall_score(y_true, y_pred),#, average='weighted'
            'f1':f1_score(y_true, y_pred),#, average='weighted'3
            'AUC':roc_auc_score(y_true, y_score),
            'NDCG':ndcg_at_k(y_true, y_score, k = y_true.sum()),
           'precision@K':recall_at_k(y_true,y_score, k=y_true.sum()),
            'LIFT': recall_at_k(y_true,y_score, k=y_true.sum())/y_true.mean() * 100,
            'recall@K': recall_at_k(y_true,y_score, k=y_true.sum()),
        })

def get_accuracy(data):
    allres = pd.DataFrame()
    for x in ['B','C','D','E','F','BC','BCD','BCDE','BCDEF','CD','CDE','CDEF','CE','CEF']:
        for target in ['pub_top_5pct','pub_top_10pct','pub_top_20pct','pub_top_30pct',
                       'job_top_5pct','job_top_10pct','job_top_20pct','job_top_30pct',
                       'pub_w_top_5pct','pub_w_top_10pct','pub_w_top_20pct','pub_w_top_30pct']:
            if 'F' in x and 'job' in target:
                continue

            y_true = data[x + '_' + target]
            y_pred = data[x + '_' + target + '_pred']
            y_score = data[x + '_' + target + '_score']

            sub_res = pd.DataFrame([get_metric_test(y_true, y_pred, y_score)])
            sub_res['x'] = x
            sub_res['target'] = target
            allres = pd.concat([allres, sub_res])

    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'AUC', 'NDCG', 'precision@K']
    allres[metric_cols] = allres[metric_cols] * 100

    custom_order = ['pub_top_5pct','pub_top_10pct','pub_top_20pct','pub_top_30pct',
                    'job_top_5pct','job_top_10pct','job_top_20pct','job_top_30pct',
                    'pub_w_top_5pct','pub_w_top_10pct','pub_w_top_20pct','pub_w_top_30pct']
    custom_order2 = ['B','C','D','E','F','BC','BCD','BCDE','BCDEF','CD','CDE','CDEF','CE','CEF']

    allres['target'] = pd.Categorical(allres['target'], categories=custom_order, ordered=True)
    allres['x'] = pd.Categorical(allres['x'], categories=custom_order2, ordered=True)
    allres = allres.sort_values(by=['target', 'x'])

    return allres
if __name__ == '__main__':
    data = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)    
    res = get_full_prediction_ensemble(
        data,
        2,
        {'Logit_c': {'estimator__C': [0.001, 0.01, 0.1, 1]}}
    )
    res.to_csv('output_prediction_main_2018.csv')
    print('Saved: result_main_2018.csv')
    
    acc = get_accuracy(res)
    acc.to_csv('output_accuracy_main_2018.csv', index=False)






