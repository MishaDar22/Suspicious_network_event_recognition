import numpy as np
import pandas as pd
import typing as tp

from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn.metrics import roc_auc_score

from mlflow_tracking import run_experiment


###########################
# kNN
###########################
# Cross-Validation wrapper
def cv_knn(train_df: pd.DataFrame, target: str, features: list, n_folds: int = 5,
           random_state: tp.Union[int, None] = None, debug=False, mlflow_tracking: bool = False,
           exp_name: str = 'knn_exp', *args, **kwargs):
    """
    KNN Cross-Validation wrapper.
    :param train_df: a training dataset with selected and preprocessed features
    :param target: a target variable
    :param features: selected features for training model and making predictions
    :param n_folds: number of folds. Must be at least 2
    :param random_state: random state
    :param debug: display progress and results of each fold
    :param kwargs: parameters for kNN
    :param mlflow_tracking: to use mlflow for tracking parameters and metrics
    :param exp_name: an experiment name in mlflow tracking
    :return: train_results: score for a training dataset; valid_results: score for a valid dataset;
     predictions for each fold of validation dataset; record indexes for each fold of validation dataset;
     model of each iteration
    """
    # Prepare K-Folds cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # lists to store the results
    valid_results = []
    train_results = []
    predictions = []
    indexes = []

    # Model validation loop on successive folds
    for train, test in kf.split(train_df.index.values):
        # Estimator preparation
        clf = neighbors.KNeighborsClassifier(*args, **kwargs)
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

    # Using mlflow for tracking results
    if mlflow_tracking:
        metrics = {'Valid_AUC': np.mean(valid_results) * 100,
                   'Train_AUC': np.mean(train_results) * 100}
        run_experiment(exp_name, 'Random Forest', kwargs, metrics, features)
    return train_results, valid_results, predictions, indexes
