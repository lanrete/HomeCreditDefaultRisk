#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/21

Hold the transformers to select features within the pipeline
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer, minmax_scale
from sklearn.feature_selection import f_classif


class ModelBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector to select features based on the `feature_importances_` from a fitted estimators.

    The x and y will be used to fit the given estimators once, the features will be sorted by the `feature_importances_`
    and top features will be kept as the transform result.
    """

    def __init__(self, clf, number_features=None, percent_features=None):
        """

        Parameters
        ----------
        clf : sklearn estimator
            The estimator used to calculate the `feature_importances_`on complete dataset.
            Estimator given must have attributes `feature_importances_` after fitted.
        number_features : int, default None
            Number of features to keep after feature selection, must be in (0, # total features]
        percent_features : float, default None
            Percentage of features to keep after feature selection, must be in (0, 1]
            If original dataset have 100 features and `percent_features=0.25`, then 25 features with
            highest `feature_importances_` will be selected
        """
        self.clf = clf
        self.selected_columns = None
        if number_features:
            assert number_features > 0
            self.number_features = number_features
            self.percent_features = None
            self.select_type = 'Number'
        elif percent_features:
            assert 0 < percent_features <= 1
            self.number_features = None
            self.percent_features = percent_features
            self.select_type = 'Percent'
        else:
            raise ValueError('Either number_features or percent_features should be given')

    def fit(self, X, y=None, **fit_params):
        _, total_columns = X.shape
        self.clf.fit(X, y, **fit_params)
        importance = self.clf.feature_importances_  # type: np.ndarray
        column_index = importance.argsort()

        if self.select_type == 'Percent':
            self.number_features = int(total_columns * self.percent_features)
        if self.number_features > total_columns:
            self.number_features = total_columns

        self.selected_columns = column_index[-self.number_features:]
        return self

    def transform(self, X, y=None, **transform_params):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_columns]
        else:
            return X[:, self.selected_columns]


class ByGroupSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select one feature from each feature groups.
    Feature group are defined as features generated from aggregation at different level.
    For example, when aggregating history information, we might have the credit limit in past 6 month,
    for these credit limits, we might generate mulitple features when aggregating, namely:
        1) credit_limit_6m_mean
        2) credit_limit_6m_max
        3) credit_limit_6m_min
    Since these features all came from same source but just aggregated differently, we would like to
    choose only one feature from these 3 trying to reduce overfitting.

    The feature groups are detected here by name by below rules:
        1) Feature name should start with _, which is a flag for aggregated features
        2) Feature name actually follow below pattern:
            _[Source]_[OriginalName]_[AggMethod]
        3) We extract the source+original_name part as the group key,
           If there are multiple features in one group, we will choose one feature from the group

    Method used to select feature:
    We use sklearn built-in function to select the features,
    since sklearn doesn't work with missing data,
    the features are first imputed by `Imputer()`, which will fill in the median of non-missing value,
    we use `f_classif` function to select feature.
    """

    def __init__(self):
        self.column_list = []
        self.imputer = Imputer()

    def fit(self, X, y=None, **fit_params):
        col_df = pd.DataFrame({
            'Column': X.columns
        })
        col_df['Suffix'] = col_df['Column'].apply(
            lambda col: '_'.join(col.split('_')[:-1])
        )
        suffix_count = col_df['Suffix'].value_counts()
        col_df['Need Selection'] = col_df['Suffix'].apply(
            lambda suffix: suffix_count[suffix] > 1 and suffix.startswith('_')
        )
        selection_df = col_df[col_df['Need Selection']].copy()

        part_x = X[selection_df['Column']]
        processed_part_x = self.imputer.fit_transform(part_x)
        processed_part_x = minmax_scale(processed_part_x)
        score = f_classif(processed_part_x, y)[0]
        selection_df['Score'] = score

        selected_features = selection_df.groupby(by=['Suffix']).apply(
            lambda part_df: part_df.loc[part_df['Score'].idxmax(), 'Column']
        )

        ret_list = list(selected_features)
        ret_list.extend(col_df.loc[~col_df['Need Selection'], 'Column'])

        self.column_list = ret_list
        return self

    def transform(self, X, y=None, **transform_params):
        return X[self.column_list]


class RemoveAllMissing(BaseEstimator, TransformerMixin):
    """
    A simple transformer to remove features with all missing value from the dataset
    These features could be generated during the aggregating process
    """

    def __init__(self):
        self.column_list = []

    def fit(self, X, y=None, **fit_params):
        # Reset the column_list here everytime the transformer is being fitted.
        # Otherwise, transformer will actually duplicates the columns since column_list
        # keep expanding
        self.column_list = []
        for col in X.columns:
            if any(X[col].notnull()):
                self.column_list.append(col)
        return self

    def transform(self, X, y=None, **transform_params):
        ret_X = X.copy()
        return ret_X[self.column_list]
