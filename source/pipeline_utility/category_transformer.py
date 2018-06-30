#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Created by Lanrete on 2018/6/16
"""

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator


class CategoryToIntTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer will take every columns in the DataFrame and convert them into integers.
    So the object can be used later for OneHotEncoder.

    Idealy, this transformer won't be necessary after the release of CategoricalEncoder which is currently
    under development in Scikit-Learn dev version 0.20
    """
    def __init__(self):
        self.category = {}

    def fit(self, X, y=None, **fit_params):
        for _ in X.columns:
            X[_] = X[_].astype('category')
            self.category[_] = X[_].cat.categories
        return self

    def transform(self, X, y=None, **trans_params):
        for _ in X.columns:
            X[_] = pd.Categorical(X[_], categories=self.category[_])
            X[_] = X[_].cat.codes
            X[_] = np.where(X[_] < 0, len(self.category[_]) + 2, X[_])
        return X
