"""
In this module we store prepare the sataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_pool_quality,
            _fix_misc_feature,
            _fix_fireplace_quality,
            _fix_garage_variables,
            _fix_lot_frontage,
            _fix_alley,
            _fix_fence,
            _fix_masvnr_variables,
            _fix_electrical,
            _fix_basement_variables,
            _fix_unhandled_nulls,
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def _fix_pool_quality(df):
    num_total_nulls = df["PoolQC"].isna().sum()
    num_nulls_when_poolarea_is_zero = df[df["PoolArea"] == 0]["PoolQC"].isna().sum()
    assert num_nulls_when_poolarea_is_zero == num_total_nulls
    num_nulls_when_poolarea_is_not_zero = df[df["PoolArea"] != 0]["PoolQC"].isna().sum()
    assert num_nulls_when_poolarea_is_not_zero == 0
    df["PoolQC"] = df["PoolQC"].fillna("NP")
    return df


def _fix_misc_feature(df):
    num_total_nulls = df["MiscFeature"].isna().sum()
    num_nulls_when_miscval_is_zero = df[df["MiscVal"] == 0]["MiscFeature"].isna().sum()
    num_nulls_when_miscval_is_not_zero = (
        df[df["MiscVal"] != 0]["MiscFeature"].isna().sum()
    )
    assert num_nulls_when_miscval_is_zero == num_total_nulls
    assert num_nulls_when_miscval_is_not_zero == 0
    df["MiscFeature"] = df["MiscFeature"].fillna("No MF")
    return df


def _fix_fireplace_quality(df):
    num_total_nulls = df["FireplaceQu"].isna().sum()
    num_nulls_when_fireplaces_is_zero = (
        df[df["Fireplaces"] == 0]["FireplaceQu"].isna().sum()
    )
    num_nulls_when_fireplaces_is_not_zero = (
        df[df["Fireplaces"] != 0]["FireplaceQu"].isna().sum()
    )
    assert num_nulls_when_fireplaces_is_zero == num_total_nulls
    assert num_nulls_when_fireplaces_is_not_zero == 0
    df["FireplaceQu"] = df["FireplaceQu"].fillna("No FP")
    return df


def _fix_garage_variables(df):
    num_area_zeros = (df["GarageArea"] == 0).sum()
    num_cars_zeros = (df["GarageCars"] == 0).sum()
    num_both_zeros = ((df["GarageArea"] == 0) & (df["GarageCars"] == 0.0)).sum()
    assert num_both_zeros == num_area_zeros == num_cars_zeros
    for colname in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        num_total_nulls = df[colname].isna().sum()
        num_nulls_when_area_and_cars_capacity_is_zero = (
            df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)][colname]
            .isna()
            .sum()
        )
        num_nulls_when_area_and_cars_capacity_is_not_zero = (
            df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)][colname]
            .isna()
            .sum()
        )
        assert num_total_nulls == num_nulls_when_area_and_cars_capacity_is_zero
        assert num_nulls_when_area_and_cars_capacity_is_not_zero == 0
        df[colname] = df[colname].fillna("No Ga")

    num_total_nulls = df["GarageYrBlt"].isna().sum()
    num_nulls_when_area_and_cars_is_zero = (
        df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)]["GarageYrBlt"]
        .isna()
        .sum()
    )
    num_nulls_when_area_and_cars_is_not_zero = (
        df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)]["GarageYrBlt"]
        .isna()
        .sum()
    )
    assert num_nulls_when_area_and_cars_is_zero == num_total_nulls
    assert num_nulls_when_area_and_cars_is_not_zero == 0
    df["GarageYrBlt"].where(
        ~df["GarageYrBlt"].isna(), other=df["YrSold"] + 1, inplace=True
    )

    return df


def _fix_lot_frontage(df):
    assert (df["LotFrontage"] == 0).sum() == 0
    df["LotFrontage"].fillna(0, inplace=True)
    return df


def _fix_alley(df):
    df["Alley"].fillna("NA", inplace=True)
    return df


def _fix_fence(df):
    df["Fence"].fillna("NF", inplace=True)
    return df


def _fix_masvnr_variables(df):
    df = df.dropna(subset=["MasVnrType", "MasVnrArea"])
    df = df[~((df["MasVnrType"] == "None") & (df["MasVnrArea"] != 0.0))]
    return df


def _fix_electrical(df):
    df.dropna(subset=["Electrical"], inplace=True)
    return df


def _fix_basement_variables(df):
    colnames = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    cond = ~(
        df["BsmtQual"].isna()
        & df["BsmtCond"].isna()
        & df["BsmtExposure"].isna()
        & df["BsmtFinType1"].isna()
        & df["BsmtFinType2"].isna()
    )
    for c in colnames:
        df[c].where(cond, other="NB", inplace=True)
    return df


def _fix_unhandled_nulls(df):
    df.dropna(inplace=True)
    return df


def get_categorical_column_names() -> t.List[str]:
    return (
        "MSSubClass,MSZoning,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,"
        + "Neighborhood,Condition1,Condition2,BldgType,HouseStyle,RoofStyle,RoofMatl,"
        + "Exterior1st,MasVnrType,Foundation,Heating,Electrical,GarageType,PavedDrive,"
        + "MiscFeature,SaleType,SaleCondition,OverallQual,OverallCond,ExterQual,"
        + "ExterCond,BsmtQual,BsmtCond,BsmtFinType1,HeatingQC,PoolQC,Fence,KitchenQual,"
        + "Functional,FireplaceQu,GarageFinish,GarageQual,GarageCond,BsmtExposure,"
        + "BsmtFinType2,Exterior2nd,MoSold"
    ).split(",")


def get_binary_column_names() -> t.List[str]:
    return "Street,CentralAir".split(",")


def get_numeric_column_names() -> t.List[str]:
    return (
        "LotFrontage,LotArea,MasVnrArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,"
        + "2ndFlrSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,MiscVal,LowQualFinSF,"
        + "GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,"
        + "TotRmsAbvGrd,Fireplaces,GarageCars,GarageArea,WoodDeckSF,OpenPorchSF"
    ).split(",")


def get_column_names() -> t.List[str]:
    return (
        "MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,"
        + "LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,"
        + "OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,"
        + "MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,"
        + "BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,"
        + "Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,"
        + "HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,"
        + "FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,"
        + "GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,"
        + "PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition"
    ).split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
        "MSSubClass": (
            "20",
            "30",
            "40",
            "45",
            "50",
            "60",
            "70",
            "75",
            "80",
            "85",
            "90",
            "120",
            "150",
            "160",
            "180",
            "190",
        ),
        "MSZoning": ("RL", "FV", "RM", "RH", "C (all)"),
        "Street": ("Pave", "Grvl"),
        "Alley": ("NA", "Pave", "Grvl"),
        "LotShape": ("Reg", "IR1", "IR3", "IR2"),
        "LandContour": ("Lvl", "Bnk", "HLS", "Low"),
        "Utilities": ("AllPub", "NoSeWa"),
        "LotConfig": ("Inside", "Corner", "FR2", "CulDSac", "FR3"),
        "LandSlope": ("Gtl", "Mod", "Sev"),
        "Neighborhood": (
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
            "Timber",
            "Veenker",
            "BrkSide",
            "CollgCr",
            "BrDale",
            "NoRidge",
            "IDOTRR",
            "Crawfor",
            "Blmngtn",
            "NPkVill",
            "Blueste",
        ),
        "Condition1": (
            "Norm",
            "PosN",
            "RRNn",
            "Feedr",
            "RRAn",
            "Artery",
            "RRAe",
            "PosA",
            "RRNe",
        ),
        "Condition2": ("Norm", "RRAn", "Feedr", "RRNn", "Artery", "PosA", "PosN"),
        "BldgType": ("1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"),
        "HouseStyle": (
            "2Story",
            "1Story",
            "1.5Fin",
            "SLvl",
            "SFoyer",
            "2.5Fin",
            "2.5Unf",
            "1.5Unf",
        ),
        "OverallQual": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
        "OverallCond": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
        "RoofStyle": ("Hip", "Gable", "Mansard", "Flat", "Gambrel", "Shed"),
        "RoofMatl": (
            "CompShg",
            "WdShake",
            "Tar&Grv",
            "WdShngl",
            "ClyTile",
            "Roll",
            "Metal",
            "Membran",
        ),
        "Exterior1st": (
            "HdBoard",
            "Plywood",
            "VinylSd",
            "Wd Sdng",
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
        ),
        "Exterior2nd": (
            "HdBoard",
            "VinylSd",
            "Wd Sdng",
            "CmentBd",
            "MetalSd",
            "ImStucc",
            "Wd Shng",
            "BrkFace",
            "Plywood",
            "Stucco",
            "AsbShng",
            "Brk Cmn",
            "Stone",
            "CBlock",
            "AsphShn",
            "Other",
        ),
        "MasVnrType": ("BrkFace", "None", "Stone", "BrkCmn"),
        "ExterQual": ("TA", "Gd", "Ex", "Fa"),
        "ExterCond": ("TA", "Gd", "Fa", "Ex", "Po"),
        "Foundation": ("PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"),
        "BsmtQual": ("Gd", "TA", "Ex", "NB", "Fa"),
        "BsmtCond": ("TA", "Fa", "Gd", "NB", "Po"),
        "BsmtExposure": ("No", "Mn", "Av", "Gd", "NB"),
        "BsmtFinType1": ("GLQ", "BLQ", "Unf", "Rec", "ALQ", "LwQ", "NB"),
        "BsmtFinType2": ("Unf", "LwQ", "Rec", "BLQ", "GLQ", "NB", "ALQ"),
        "Heating": ("GasA", "GasW", "Wall", "Grav", "Floor", "OthW"),
        "HeatingQC": ("Ex", "Gd", "TA", "Fa", "Po"),
        "CentralAir": ("Y", "N"),
        "Electrical": ("SBrkr", "FuseF", "FuseA", "FuseP", "Mix"),
        "KitchenQual": ("Gd", "TA", "Ex", "Fa"),
        "Functional": ("Typ", "Min2", "Min1", "Mod", "Maj1", "Maj2", "Sev", "Sal"),
        "FireplaceQu": ("Ex", "No FP", "TA", "Gd", "Fa", "Po"),
        "GarageType": (
            "Attchd",
            "BuiltIn",
            "Detchd",
            "No Ga",
            "Basment",
            "CarPort",
            "2Types",
        ),
        "GarageFinish": ("Unf", "RFn", "No Ga", "Fin"),
        "GarageQual": ("TA", "No Ga", "Fa", "Gd", "Ex", "Po"),
        "GarageCond": ("TA", "No Ga", "Fa", "Gd", "Ex", "Po"),
        "PavedDrive": ("Y", "N", "P"),
        "PoolQC": ("NP", "Fa", "Gd", "Ex"),
        "Fence": ("NF", "MnPrv", "GdPrv", "GdWo", "MnWw"),
        "MiscFeature": ("No MF", "Shed", "TenC", "Gar2", "Othr"),
        "SaleType": (
            "WD",
            "New",
            "COD",
            "Con",
            "ConLD",
            "ConLw",
            "ConLI",
            "Oth",
            "CWD",
        ),
        "MoSold": ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"),
        "SaleCondition": (
            "Normal",
            "Partial",
            "Family",
            "Abnorml",
            "Alloca",
            "AdjLand",
        ),
    }
