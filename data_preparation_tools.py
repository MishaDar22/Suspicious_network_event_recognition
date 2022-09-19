import pandas as pd
import typing as tp

from sklearn import preprocessing


# Data preparation
def preprocessing_cat_features_le(cat_features: list, df: pd.DataFrame) -> tp.Tuple[pd.DataFrame, dict]:
    df = df.copy()

    maps = {}
    for feature in cat_features:
        le = preprocessing.LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        maps[feature] = le
    return df, maps


def preprocessing_data(train: pd.DataFrame, test: pd.DataFrame, target: str, var_for_preprocessing: dict) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    """
    Function for preprocessing a training dataset with a target variable and a test dataset without a target variable

    :param train: a training dataset with a target variable.
    :param test: a test dataset without a target variable.
    :param target: Predictive variable
    :param black_list: A list with the features that won't be used in predictions.
    :param var_for_preprocessing: The variables that will be preprocessed.
    :return: Pre-processed training and test dataset, maps for features and
    features, that will be used for predictions.
    """

    # Data imputation
    train = train.fillna(-1)
    test = test.fillna(-1)

    # Join training and test dataset
    df_full = pd.concat([train, test])

    # Add some feature to black list, witch will be removed from main dataset
    black_list = var_for_preprocessing.get("feature_black_list")
    if black_list:
        black_list.append(target)
    else:
        black_list = [target]

    for var_type, var in var_for_preprocessing.items():
        if var_type == "categorical_features" or var_type == "numeric_categorical_features":
            for feature, n in var:
                set_top_n_categories_in_variable(df_full, feature, n)
        elif var_type == "continuous_numeric_features":
            df_full[var]= df_full[var].apply(lambda x: (x - x.mean())/x.std())
        # One Hot Encoding
        elif var_type == "one_hot_encoding":
            df_bin_var = pd.get_dummies(df_full[var], drop_first=True)
            black_list.extend(var)
            df_full = pd.concat([df_full, df_bin_var], axis=1)

    cat_features = list(set(df_full.select_dtypes(exclude='number').columns.tolist())-set(black_list))

    new_df_full, maps = preprocessing_cat_features_le(cat_features, df_full)
    features = list(set(new_df_full.columns.tolist())-set(black_list))

    df_test = new_df_full.loc[new_df_full.notified.isna()][features]
    df_train = new_df_full.loc[new_df_full.notified.notna()][features+[target]]
    return df_train, df_test, maps, features


def set_top_n_categories_in_variable(df: pd.DataFrame, feature: str, n_cat: int):
    top_n_var = df[feature].value_counts().nlargest(n_cat).index
    df[feature] = df[feature].where(df[feature].isin(top_n_var), other='Other')
