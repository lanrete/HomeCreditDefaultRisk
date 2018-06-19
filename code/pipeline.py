#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16
"""
# TODO  Save the result

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import chi2, SelectPercentile

from pipeline_utility.general_transformer import ColumnTypeExtract, GeneralImputer
from pipeline_utility.category_transformer import CategoryToIntTransformer

from lightgbm import LGBMClassifier


PIPELINE = Pipeline([
    ('Features', FeatureUnion([
        ('Categorical', Pipeline([
            ('Extract', ColumnTypeExtract('Category')),
            ('Impute', GeneralImputer(value='Missing')),
            ('ToInt', CategoryToIntTransformer()),
            ('One-Hot', OneHotEncoder(handle_unknown='ignore')),
        ])),
        ('Numerical', Pipeline([
            ('Extract', ColumnTypeExtract('Numerical')),
            # ('Impute', GeneralImputer(func=np.median)),
        ]))
    ])),
    # ('Select Feature', SelectPercentile(score_func=chi2, percentile=50)),
    ('Classifier', LGBMClassifier())
])

params_grid = [
    # {
    #     'Features__Numerical__Impute__func': [np.median, np.mean],
    #     'Classifier__num_leaves': [31, 63, 127],
    #     'Classifier__learning_rate': [0.1, 0.3],
    #     'Classifier__n_estimators': [100, 50],
    # },
    {
        # 'Features__Numerical__Impute__func': [np.median, np.mean],
        # 'Select Feature__percentile': [10, 25, 50, 80, 100],

        'Classifier__num_leaves': [31, 63],
        'Classifier__learning_rate': [0.03, 0.01],
        'Classifier__n_estimators': [200, 500],
    },
]


def fit_pipeline_parameters(x_train, y_train, x_test, y_test, predict=False, x_score=None, submission=None):
    PIPELINE.fit(x_train, y_train)
    clf = GridSearchCV(
        PIPELINE, params_grid,
        scoring='roc_auc',
        cv=5, verbose=1)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    best_clf = clf.best_estimator_
    y_test_pred = best_clf.predict_proba(x_test)[:, 1]
    print(f'AUC on testing set: {roc_auc_score(y_score=y_test_pred, y_true=y_test)}')
    if predict:
        y_pred = best_clf.predict_proba(x_score)[:, 1]
        result_df = pd.DataFrame({'TARGET': y_pred})
        result_df.index = x_score.index
        result_df.to_csv(f'../submission/{submission}.csv', index=True)


def main():
    data = pd.read_csv('./data/_final_train.csv')
    y = data['TARGET']
    y = LabelEncoder().fit_transform(y)
    del data['TARGET']
    x = data.set_index(keys='SK_ID_CURR', drop=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2136)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    fit_pipeline_parameters(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
