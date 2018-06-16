#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pipeline_utility.general_transformer import ColumnTypeExtract, GeneralImputer
from pipeline_utility.category_transformer import CategoryToIntTransformer

from lightgbm import LGBMClassifier


data = pd.read_csv('./data/application_train.csv')
y = data['TARGET']
y = LabelEncoder().fit_transform(y)
del data['TARGET']
x = data.iloc[:, 1:]  # type:pd.DataFrame

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2136)
assert len(x_train) == len(y_train)
assert len(x_test) == len(y_test)

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
            ('Impute', GeneralImputer(func=np.median)),
        ]))
    ])),
    ('Classifier', LGBMClassifier())
])

params_grid = {
    'Features__Numerical__Impute__func': [np.median, np.mean],
    'Classifier__num_leaves': [31, 63, 127],
    'Classifier__learning_rate': [0.1, 0.3, 0.03, 0.01],
    'Classifier__n_estimators': [100, 200, 500, 50],
}


if __name__ == '__main__':
    PIPELINE.fit(x_train, y_train)
    clf = GridSearchCV(
        PIPELINE, params_grid,
        scoring='roc_auc',
        cv=5, verbose=3)
    clf.fit(x_train, y_train)
    best_clf = clf.best_estimator_

    y_test_pred = best_clf.predict_proba(x_test)[:, 1]
    print(f'AUC on testing set: {roc_auc_score(y_score=y_test_pred, y_true=y_test)}')
