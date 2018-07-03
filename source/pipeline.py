#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16
"""

# TODO  Save the CV result for reference

import datetime as dt

import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from config import PIPELINE, PARAMS_GRID


def fit_pipeline(x, y, predict=False, x_score=None, submission=None, fit_params=False):
    """
    Main container for the model fitting.
    PIPELINE and PARAMS_GRID are pre-defined in this script and will be called in this function.

    Parameters
    ----------
    x : pd.DataFrame
        X in the training base
    y : pd.DataFrame
        Target in the training base
    predict : bool, default False
        If True, the fitted model will be called again to give prediction on the validation/scoring set
    x_score : pd.DataFrame, default None
        The validation/scoring set, ignored if `predict=False`
    submission : string, default None
        The name of the submission file, ignored if `predict=False`
        The scoring result will be saved as '../submission/{submission}.csv'
        If `submission=None`, the timestamp will be used as the name.
        Timestamp format is %Y%m%d_%H%M%S
    fit_params : bool, default False
        If True, PARAMS_GRID will be used to find the best hyper-parameters.
                 In this case, a 5-fold CV will be used to find the best hyper-parameters
        If False, the default parameters set in PIPELINE will be directly used as the model

    Returns
    -------
    None
    """
    if fit_params:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2028)
        PIPELINE.fit(x_train, y_train)
        clf = GridSearchCV(
            PIPELINE, PARAMS_GRID,
            scoring='roc_auc',
            return_train_score=True,
            cv=5, verbose=2)
        clf.fit(x_train, y_train)
        print(clf.best_params_)
        best_clf = clf.best_estimator_
        y_test_pred = best_clf.predict_proba(x_test)[:, 1]
        print(f'AUC on testing set: {roc_auc_score(y_score=y_test_pred, y_true=y_test)}')
    else:
        best_clf = PIPELINE
        best_clf.fit(x, y)
        predict = True
    if predict:
        print(f'AUC on whole set: {roc_auc_score(y_score=best_clf.predict_proba(x)[:, 1], y_true=y)}')
        y_pred = best_clf.predict_proba(x_score)[:, 1]
        result_df = pd.DataFrame({'TARGET': y_pred})
        result_df.index = x_score.index
        if submission is None:
            submission = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_df.to_csv(f'../submission/{submission}.csv', index=True)
    return best_clf


def main():
    data = pd.read_csv('./data/_final_train.csv')
    y = data['TARGET']
    y = LabelEncoder().fit_transform(y)
    del data['TARGET']
    x = data.set_index(keys='SK_ID_CURR', drop=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2136)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    fit_pipeline(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
