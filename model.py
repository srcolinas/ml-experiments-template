"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


EstimatorConfig = t.List[t.Dict[str, t.Any]]


def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        hparams = step.get("hparams", {})
        estimator = estimator_mapping[name](**hparams)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "random-forest-regressor": RandomForestRegressor,
        "linear-regressor": LinearRegression,
        "age-extractor": AgeExtractor,
        "ignore-and-encode-transformer": IgnoreAndEncodeTransformer,
        "one-hot-encoder": OneHotEncoder,
        "average-per-neighborhood-baseline": AveragePerNeighborhoodBaseline,
    }


class AgeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAddAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        return X


class IgnoreAndEncodeTransformer(BaseEstimator, TransformerMixin):
    _categorical_columns = (
        "MSSubClass,MSZoning,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,"
        + "Neighborhood,Condition1,Condition2,BldgType,HouseStyle,RoofStyle,RoofMatl,"
        + "Exterior1st,MasVnrType,Foundation,Heating,Electrical,GarageType,PavedDrive,"
        + "MiscFeature,SaleType,SaleCondition,OverallQual,OverallCond,ExterQual,"
        + "ExterCond,BsmtQual,BsmtCond,BsmtFinType1,HeatingQC,PoolQC,Fence,KitchenQual,"
        + "Functional,FireplaceQu,GarageFinish,GarageQual,GarageCond,BsmtExposure,"
        + "BsmtFinType2,Exterior2nd,MoSold"
    ).split(",")

    _binary_columns = "Street,CentralAir".split(",")

    _float_columns = (
        "LotFrontage,LotArea,MasVnrArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,"
        + "2ndFlrSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,MiscVal,LowQualFinSF,"
        + "GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,"
        + "TotRmsAbvGrd,Fireplaces,GarageCars,GarageArea,WoodDeckSF,OpenPorchSF,"
        + "HouseAge,RemodAddAge,GarageAge"
    ).split(",")

    _ignored_columns = "YrSold,YearBuilt,YearRemodAdd,GarageYrBlt".split(",")

    _ordinal_encoder_categories = (
        ("MSZoning", np.array(["RL", "FV", "RM", "RH", "C,(all)"])),
        ("Street", np.array(["Pave", "Grvl"])),
        ("Alley", np.array(["NA", "Pave", "Grvl"])),
        ("LotShape", np.array(["Reg", "IR1", "IR3", "IR2"])),
        ("LandContour", np.array(["Lvl", "Bnk", "HLS", "Low"])),
        ("Utilities", np.array(["AllPub", "NoSeWa"])),
        ("LotConfig", np.array(["Inside", "Corner", "FR2", "CulDSac", "FR3"])),
        ("LandSlope", np.array(["Gtl", "Mod", "Sev"])),
        (
            "Neighborhood",
            np.array(
                [
                    "SawyerW",
                    "Sawyer",
                    "Gilbert",
                    "NridgHt",
                    "SWISU",
                    "Edwards",
                    "NWAmes",
                    "NAmes",
                    "Somerst",
                    "StoneBr",
                    "OldTown",
                    "ClearCr",
                    "Mitchel",
                    "MeadowV",
                    "Timber" "Veenker",
                    "BrkSide",
                    "CollgCr",
                    "BrDale",
                    "NoRidge",
                    "IDOTRR",
                    "Crawfor" "Blmngtn",
                    "NPkVill",
                ]
            ),
        ),
        (
            "Condition1",
            np.array(
                [
                    "Norm",
                    "PosN",
                    "RRNn",
                    "Feedr",
                    "RRAn",
                    "Artery",
                    "RRAe",
                    "PosA",
                    "RRNe",
                ]
            ),
        ),
        ("Condition2", np.array(["Norm", "RRAn", "Feedr", "RRNn", "Artery", "PosA"])),
        ("BldgType", np.array(["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])),
        (
            "HouseStyle",
            np.array(
                [
                    "2Story",
                    "1Story",
                    "1.5Fin",
                    "SLvl",
                    "SFoyer",
                    "2.5Fin",
                    "2.5Unf",
                    "1.5Unf",
                ]
            ),
        ),
        ("RoofStyle", np.array(["Hip", "Gable", "Mansard", "Flat", "Gambrel", "Shed"])),
        ("RoofMatl", np.array(["CompShg", "WdShake", "Tar&Grv", "WdShngl", "ClyTile"])),
        (
            "Exterior1st",
            np.array(
                [
                    "HdBoard",
                    "Plywood",
                    "VinylSd",
                    "Wd,Sdng",
                    "CemntBd",
                    "MetalSd",
                    "BrkFace",
                    "WdShing",
                    "Stucco",
                    "AsbShng",
                    "Stone",
                    "CBlock",
                    "ImStucc",
                    "BrkComm",
                    "AsphShn",
                ]
            ),
        ),
        (
            "Exterior2nd",
            np.array(
                [
                    "HdBoard",
                    "VinylSd",
                    "Wd,Sdng",
                    "CmentBd",
                    "MetalSd",
                    "ImStucc",
                    "Wd,Shng",
                    "BrkFace",
                    "Plywood",
                    "Stucco",
                    "AsbShng",
                    "Brk,Cmn",
                    "Stone",
                    "CBlock",
                    "AsphShn",
                    "Other",
                ]
            ),
        ),
        ("MasVnrType", np.array(["BrkFace", "None", "Stone", "BrkCmn"])),
        ("ExterQual", np.array(["TA", "Gd", "Ex", "Fa"])),
        ("ExterCond", np.array(["TA", "Gd", "Fa", "Ex", "Po"])),
        (
            "Foundation",
            np.array(["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"]),
        ),
        ("BsmtQual", np.array(["Gd", "TA", "Ex", "NB", "Fa"])),
        ("BsmtCond", np.array(["TA", "Fa", "Gd", "NB", "Po"])),
        ("BsmtExposure", np.array(["No", "Mn", "Av", "Gd", "NB"])),
        ("BsmtFinType1", np.array(["GLQ", "BLQ", "Unf", "Rec", "ALQ", "LwQ", "NB"])),
        ("BsmtFinType2", np.array(["Unf", "LwQ", "Rec", "BLQ", "GLQ", "NB", "ALQ"])),
        ("Heating", np.array(["GasA", "GasW", "Wall", "Grav", "Floor", "OthW"])),
        ("HeatingQC", np.array(["Ex", "Gd", "TA", "Fa", "Po"])),
        ("CentralAir", np.array(["Y", "N"])),
        ("Electrical", np.array(["SBrkr", "FuseF", "FuseA", "FuseP", "Mix"])),
        ("KitchenQual", np.array(["Gd", "TA", "Ex", "Fa"])),
        ("Functional", np.array(["Typ", "Min2", "Min1", "Mod", "Maj1", "Maj2"])),
        ("FireplaceQu", np.array(["Ex", "No,FP", "TA", "Gd", "Fa", "Po"])),
        (
            "GarageType",
            np.array(
                ["Attchd", "BuiltIn", "Detchd", "No,Ga", "Basment", "CarPort", "2Types"]
            ),
        ),
        ("GarageFinish", np.array(["Unf", "RFn", "No,Ga", "Fin"])),
        ("GarageQual", np.array(["TA", "No,Ga", "Fa", "Gd", "Ex", "Po"])),
        ("GarageCond", np.array(["TA", "No,Ga", "Fa", "Gd", "Ex", "Po"])),
        ("PavedDrive", np.array(["Y", "N", "P"])),
        ("PoolQC", np.array(["NP", "Fa", "Gd", "Ex"])),
        ("Fence", np.array(["NF", "MnPrv", "GdPrv", "GdWo", "MnWw"])),
        ("MiscFeature", np.array(["No,MF", "Shed", "TenC", "Gar2"])),
        (
            "SaleType",
            np.array(
                ["WD", "New", "COD", "Con", "ConLD", "ConLw", "ConLI", "Oth", "CWD"]
            ),
        ),
        (
            "SaleCondition",
            np.array(["Normal", "Partial", "Family", "Abnorml", "Alloca", "AdjLand"]),
        ),
    )

    def __init__(self):
        column_names, categories = zip(*type(self)._ordinal_encoder_categories)
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("droper", "drop", type(self)._ignored_columns),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories="auto",
                        handle_unknown="use_encoded_value",
                        unknown_value=100,
                    ),
                    type(self)._binary_columns + type(self)._categorical_columns,
                ),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)


class AveragePerNeighborhoodBaseline(BaseEstimator, RegressorMixin):
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
