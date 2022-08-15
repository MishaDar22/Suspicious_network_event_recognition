import pandas as pd
import typing as tp

from sklearn import preprocessing


# Data preparation
def preprocessing_cat_features_le(cat_features: list, df: pd.DataFrame) -> tp.Tuple[pd.DataFrame, dict]:
    df = df.copy()

    maps = {}
    for feature in cat_features:
        le = preprocessing.LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        maps[feature] = le
    return df, maps


def preprocessing_data(train: pd.DataFrame, test: pd.DataFrame, target: str, black_list: list) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame, dict, list]:

    """
    Function for preprocessing a training dataset with a target variable and a test dataset without a target variable

    :param train: a training dataset with a target variable.
    :param test: a test dataset without a target variable.
    :param target: Predictive variable
    :param black_list: A list with the features that won't be used in predictions.
    :return: Pre-processed training and test dataset, maps for features and
    features, that will be used for predictions.
    """
    df_full = pd.concat([train, test])
    features_num = df_full.select_dtypes('number').columns.tolist()
    features_cat = df_full.select_dtypes(exclude='number').columns.tolist()
    black_list.append(target)
    features_num = [feature for feature in features_num if feature not in black_list]
    features_cat = [feature for feature in features_cat if feature not in black_list]
    new_df_full, maps = preprocessing_cat_features_le(features_cat, df_full)
    features = features_cat+features_num

    df_test = new_df_full.loc[new_df_full.notified.isna()][features]
    df_train = new_df_full.loc[new_df_full.notified.notna()][features+[target]]
    return df_train, df_test, maps, features
