import pandas as pd
import numpy as np
import typing as tp

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlflow_tracking import run_experiment


###############################
# Random Forest
###############################
# Cross-Validation wrapper
def cv_random_forest(train_df: pd.DataFrame, target: str, features: list, n_folds: int = 5, random_state: int = 2022,
                     debug: bool = False, save_models=True, mlflow_tracking: bool = False,
                     exp_name: str = 'some_exp', *args, **kwargs) -> tp.Tuple[list, list, list, list, list]:
    """
    Cross-Validation wrapper for imbalanced dataset using Random Forest algorithm for predictions

    :param train_df: a training dataset with selected and preprocessed features
    :param target: a target variable
    :param features: selected features for training model and making predictions
    :param n_folds: number of folds. Must be at least 2
    :param random_state: random state
    :param debug: display progress and results of each fold
    :param save_models: to save model of each fold
    :param mlflow_tracking: to use mlflow for tracking parameters and metrics
    :param exp_name: an experiment name in mlflow tracking
    :param kwargs: parameters for Random Forest
    :return: train_results: score for a training dataset; valid_results: score for a valid dataset;
     predictions for each fold of validation dataset; record indexes for each fold of validation dataset;
     model of each iteration
    """

    # Stratified K-Folds cross-validator for imbalanced dataset
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # lists to store the results
    valid_results = []
    train_results = []
    predictions = []
    indexes = []
    models = []

    # Model validation loop on successive folds
    for train, test in skf.split(train_df[features], train_df[target]):
        # Estimator preparation
        clf = RandomForestClassifier(*args, **kwargs, random_state=random_state, n_jobs=-1)
        if debug:
            print(clf)
        # Training the model
        clf.fit(train_df.iloc[train][features], train_df.iloc[train][target])

        # Prepare predictions for the training and test dataset
        # NOTE Sklearn will return two columns of probabilities for both classes
        preds_train = clf.predict_proba(train_df.iloc[train][features])[:, 1]
        preds = clf.predict_proba(train_df.iloc[test][features])[:, 1]

        # Save prediction information for current iteration
        predictions.append(preds.tolist().copy())

        # and indexes in the original data frame
        indexes.append(train_df.iloc[test].index.tolist().copy())

        # Counting the fit using the ROC-AUC metric
        train_score = roc_auc_score(train_df[target].iloc[train], preds_train)
        test_score = roc_auc_score(train_df[target].iloc[test], preds)

        # Saving the results to the lists
        train_results.append(train_score)
        valid_results.append(test_score)

        # Optionally displaying results about each iteration
        if debug:
            print("Train AUC:", train_score,
                  "Valid AUC:", test_score)

        # Saving current model
        if save_models:
            models.append(clf)

    # Using mlflow for tracking results
    if mlflow_tracking:
        metrics = {'Valid_AUC': np.mean(valid_results) * 100,
                   'Train_AUC': np.mean(train_results) * 100}
        unselected_features = set(train_df.columns.tolist()) - set(features)

        run_experiment(exp_name, 'Random Forest', kwargs, metrics, features, list(unselected_features))

    return train_results, valid_results, predictions, indexes, models


# Making prediction for competition using RF
def make_prediction_rf(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list,
                       target: str, file: str, random_state: int = 2022, save=False,
                       *args, **kwargs) -> np.ndarray:
    """
    Function for making prediction for competition using Random Forest

    :param train_df: a training dataset with selected and preprocessed features
    :param test_df: a test dataset with selected and preprocessed features without target variable
    :param features: selected features for training model and making predictions
    :param target: a target variable
    :param file: file name for saving the forecast results for competition
    :param random_state: random state
    :param save: to save results to text file
    :return: numpy array with the forecast results for competition
    """
    clf = RandomForestClassifier(*args, **kwargs, random_state=random_state, n_jobs=-1)
    clf.fit(train_df[features], train_df[target])
    preds = clf.predict_proba(test_df[features])[:, 1]
    if save:
        np.savetxt(f'output/results/{file}', preds)
    return preds