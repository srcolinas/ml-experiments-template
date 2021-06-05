"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import data


EstimatorConfig = t.List[t.Dict[str, t.Any]]


def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        params = step["params"]
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "random-forest-regressor": RandomForestRegressor,
        "decision-tree-regressor": DecisionTreeRegressor,
        "linear-regressor": LinearRegression,
        "ridge-regressor": Ridge,
        "average-price-per-neighborhood-regressor": AveragePricePerNeighborhoodRegressor,
        "age-extractor": AgeExtractor,
        "categorical-encoder": CategoricalEncoder,
        "standard-scaler": StandardScaler,
        "discretizer": Discretizer,
        "averager": AveragePricePerNeighborhoodExtractor,
    }


class Discretizer(BaseEstimator, TransformerMixin):
    def __init__(self, *, bins_per_column: t.Mapping[str, int], strategy: str):
        self.bins_per_column = bins_per_column
        self.strategy = strategy

    def fit(self, X, y):
        X = X.copy()
        self.n_features_in_ = X.shape[1]
        self.original_column_order_ = X.columns.tolist()
        self.columns_, n_bins = zip(*self.bins_per_column.items())
        self.new_column_order_ = self.columns_ + tuple(
            name
            for name in self.original_column_order_
            if name not in self.bins_per_column
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    KBinsDiscretizer(
                        n_bins=n_bins, encode="ordinal", strategy=self.strategy
                    ),
                    self.columns_,
                ),
            ],
            remainder="passthrough",
        )
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        X = pd.DataFrame(
            self._column_transformer.transform(X), columns=self.new_column_order_
        )
        return X


def _get_crosser(
    *,
    columns: t.Sequence[int],
):
    transformer = ColumnTransformer(
        ("crosser", PolynomialFeatures(interaction_only=True), columns),
        remainder="passthrough",
    )
    return transformer


class AgeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAddAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        one_hot: bool = False,
        force_dense_array: bool = False,
        additional_pass_through_columns: t.Optional[t.Sequence[str]] = None,
        additional_categories: t.Optional[t.Mapping[str, t.Sequence[str]]] = None,
        to_ignore: t.Optional[t.Sequence[str]] = None,
    ):
        self.one_hot = one_hot
        self.force_dense_array = force_dense_array
        self.additional_pass_through_columns = additional_pass_through_columns
        self.additional_categories = additional_categories
        self.to_ignore = to_ignore

    def fit(self, X, y=None):
        X = X.copy()
        self.n_features_in_ = X.shape[1]

        self.categorical_column_names_, self.categories_ = self._get_categories_params()
        self.pass_through_columns_ = self._get_pass_through_columns()
        encoder_cls = (
            partial(OneHotEncoder, drop="first", sparse=not self.force_dense_array)
            if self.one_hot
            else OrdinalEncoder
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    encoder_cls(
                        categories=self.categories_,
                    ),
                    self.categorical_column_names_,
                ),
                ("pass-numeric", "passthrough", self.pass_through_columns_),
            ],
            remainder="drop",
        )
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def _get_categories_params(self):
        categories_mapping = data.get_categorical_variables_values_mapping()
        if self.additional_categories is not None:
            for k, v in self.additional_categories.items():
                categories_mapping[k] = v

        if self.to_ignore is not None:
            for k in self.ignore:
                categories_mapping.pop(k, None)

        categorical_column_names, categories = zip(
            *((k, tuple(v)) for k, v in categories_mapping.items())
        )
        return categorical_column_names, categories

    def _get_pass_through_columns(self):
        pass_through_columns = data.get_numeric_column_names()
        if self.additional_pass_through_columns is not None:
            pass_through_columns = (
                pass_through_columns + self.additional_pass_through_columns
            )
        to_ignore = set(self.to_ignore) if self.to_ignore is not None else set()
        if self.additional_categories is not None:
            to_ignore = to_ignore.union(self.additional_categories.keys())
        pass_through_columns = tuple(
            v
            for v in pass_through_columns if v not in to_ignore
        )
        return pass_through_columns

    def transform(self, X):
        return self._column_transformer.transform(X)


class AveragePricePerNeighborhoodRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        """Computes the mode of the price per neighbor on training data."""
        df = pd.DataFrame({"Neighborhood": X["Neighborhood"], "price": y})
        self.means_ = df.groupby("Neighborhood").mean().to_dict()["price"]
        self.global_mean_ = y.mean()
        return self

    def predict(self, X):
        """Predicts the mode computed in the fit method."""

        def get_average(x):
            if x in self.means_:
                return self.means_[x]
            else:
                return self.global_mean_

        y_pred = X["Neighborhood"].apply(get_average)
        return y_pred


class AveragePricePerNeighborhoodExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        df = pd.DataFrame({"Neighborhood": X["Neighborhood"], "price": y})
        self.means_ = df.groupby("Neighborhood").mean().to_dict()["price"]
        self.global_mean_ = y.mean()
        return self

    def transform(self, X):
        X = X.copy()

        def get_average(x):
            if x in self.means_:
                return self.means_[x]
            else:
                return self.global_mean_

        X["AveragePriceInNeihborhood"] = X["Neighborhood"].apply(get_average)
        return X
