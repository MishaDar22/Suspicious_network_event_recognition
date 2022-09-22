import pandas as pd
import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import time
import xgboost as xgb

from operator import itemgetter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from mlflow_tracking import run_experiment


############################
# XGBoost
############################

def run_xgb(train: pd.DataFrame, validate: pd.DataFrame, features: list, target: str, test: pd.DataFrame = None,
            file: str = 'results',
            eta: float = 0.03, max_depth: int = 7, subsample: float = 0.7, colsample_bytree: float = 0.7,
            # hyperparameters
            colsample_bylevel: float = 1, lambdax: float = 1, alpha: float = 0, gamma: float = 0, min_child_weight=0,
            # hyperparameters
            rate_drop: float = 0.2, skip_drop: float = 0.5,  # hyperparameters/DART
            num_boost_round: int = 1000, early_stopping_rounds: int = 50,  # hyperparameters/configuration
            debug: bool = True, eval_metric: tp.List[str] = ["auc"], objective: str = "binary:logistic",
            # configuration
            seed: int = 2022, booster: str = "gbtree", tree_method: str = "exact",
            grow_policy: str = "depthwise"):  # configuration
    """
    XGB wrapper for gbtree and dart
    https://xgboost.readthedocs.io/en/stable/parameter.html

    Parameters
    ----------
    train, validate, features, target: wymagane zmienne bez domyślnych wartości
    train, validate: pd.DataFrames z kolumnami opisanymi w features i target
    test: pd.DataFrames z kolumnami opisanymi w features
    features: lista zmiennych do wykorzystania w trenowaniu
    target: nazwa zmiennej objasnianej

    --- Zmienne wspólne dla gbtree i dart
        --- Zmienne właściwe dla Ensemble/Boosting
        eta : "learning rate"
        max_depth=7: maksymalna głębokość drzew [0,∞]
        subsample: udział (0,1] obserwacji do treningu jednej iteracji
        colsample_bytree: udział (0,1] kolumn do treningu jednej iteracji
        colsample_bylevel: udział (0,1] kolumn na poziom do treningu jednej iteracji
        --- Zmienne regularyzacyjne
        lambdax=0: regularyzacja L2 [0,∞]
        alpha=0: regularyzacja L1 [0,∞]
        gamma=1: minimalna redukcja funkcji straty
        min_child_weight=0: minimalna suma wag poddrzewa

    --- Zmienne dla algorytmu dart
    rate_drop :
    skip_drop :

    --- Zmienne dla XGB, opis/agorytm/liczba drzew etc.
    num_boost_round: maksymalna liczba iteracji
    early_stopping_rounds: margines iteracji dla early stopping
    debug: Czy włączyć pełne opisy.
    eval_metric: Pełna lista dostępna https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
    objective: reg:linear, reg:logistic, binary:logistic, multi:softmax lub inne Pełna lista dostępna https://xgboost.readthedocs.io/en/stable/parameter.html
    seed: random seed
    booster: silnik dla drzew gbtree (cart), dart (gbtree z dropoutem) lub gblinear
    tree_method: ‘auto’, ‘exact’, ‘approx’, ‘hist’, ‘gpu_exact’, ‘gpu_hist’: http://xgboost.readthedocs.io/en/latest/parameter.html
    grow_policy: depthwise, lossguide
    """

    start_time = time.time()
    param_list = ['eta', 'max_depth',
                  'subsample', 'colsample_bytree', 'colsample_bylevel',
                  'lambdax', 'alpha', 'gamma', 'min_child_weight',
                  'num_boost_round', 'early_stopping_rounds',
                  'rate_drop', 'skip_drop',
                  'eval_metric', 'objective',
                  'seed', 'booster', 'tree_method', 'grow_policy']

    # Creating a dictionary to submit to XGB
    params = dict()
    for param in param_list:
        params[param] = eval(param)

    if debug:
        for param in param_list:
            print(param, eval(param), end=", ")
        print('\nLength train:', len(train.index))
        print('Length valid:', len(validate.index))

    # Automatic transfer of the number of classes for multiple levels of classification
    if params["objective"] == "multi:softmax" or params["objective"] == "multi:softprob":
        params["num_class"] = train[target].nunique()
    params["silent"] = 1

    # XGB requires the lambda keyword in the parameter dictionary
    params["lambda"] = lambdax

    # Convert sets to DMatrix structure
    # The DMatrix data structure allows to efficiently create trees
    dtrain = xgb.DMatrix(train[features].values, train[target].values, feature_names=train[features].columns.values)

    dvalid = xgb.DMatrix(validate[features].values, validate[target].values,
                         feature_names=validate[features].columns.values)
    if test is not None:
        dtest = xgb.DMatrix(test[features].values, feature_names=test[features].columns.values)

    # Create a list of sets for evaluation
    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    # Create a dictionary variable to save the model fit history
    train_result_history = dict()

    # Print evaluated metric every 10 stage if debug true
    verbose_eval = 10 if debug else False

    # Run the training algorithm
    gbm = xgb.train(params, dtrain,
                    num_boost_round, early_stopping_rounds=early_stopping_rounds,
                    evals=evals, evals_result=train_result_history, verbose_eval=verbose_eval)

    # Calculating statistics and additional values
    score = gbm.best_score

    # Convert training history to Pandas DataFrame
    train_history = dict()
    for key in train_result_history.keys():
        for metric in train_result_history[key].keys():
            train_history[key + metric.upper()] = train_result_history[key][metric]
    train_result_history = pd.DataFrame(train_history)

    # Save predicted values for the validation dataset for the best iteration
    train_pred = gbm.predict(dtrain, ntree_limit=gbm.best_iteration)
    valid_pred = gbm.predict(dvalid, ntree_limit=gbm.best_iteration)
    if test is not None:
        test_pred = gbm.predict(dtest, ntree_limit=gbm.best_iteration)
        np.savetxt(f'output/results/{file}', test_pred)

    # Prepare a sorted list of variable importance, instead of a dictionary
    important_variables = gbm.get_fscore()
    important_variables = sorted(important_variables.items(), key=itemgetter(1), reverse=True)

    imp_fig = None
    if debug:
        print(f'Training time: {round((time.time() - start_time) / 60, 2)} minutes')
        imp_fig, ax = plt.subplots(figsize=(20, 20))
        xgb.plot_importance(gbm, ax=ax, height=0.2)
    return score, train_pred, valid_pred, train_result_history, imp_fig, important_variables


def cv_xgb(train_df: pd.DataFrame, target: str, features: list, n_folds: int = 5, random_state: int = 2022,
           debug: bool = False, mlflow_tracking: bool = False, exp_name: str = 'Experiment_XGBoost', *args, **kwargs):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # lists for storing results:
    test_results = []
    train_results = []
    predictions = []
    indexes = []

    hists = []
    fold = 1

    # Loop for validation model on following folds
    for train, test in skf.split(train_df[features], train_df[target]):
        # Estimator preparation
        score, preds_train, preds, train_history, imp_fig, imp = run_xgb(train_df.iloc[train], train_df.iloc[test],
                                                                         features, target, debug=debug, *args, **kwargs)

        # Save prediction information for current iteration
        predictions.append(preds.tolist().copy())

        # and indexes in the original data frame
        indexes.append(train_df.iloc[test].index.tolist().copy())

        # Counting the fit using the ROC-AUC metric
        train_score = roc_auc_score(train_df[target].iloc[train], preds_train)
        test_score = roc_auc_score(train_df[target].iloc[test], preds)

        # Saving the results to the lists
        train_results.append(train_score)
        test_results.append(test_score)

        hists.append(train_history.add_suffix('_' + str(fold)))
        fold += 1

        # Optionally displaying results about each iteration
        if debug:
            print("Train AUC:", train_score,
                  "Valid AUC:", test_score)

    # Using mlflow for tracking results
    if mlflow_tracking:
        metrics = {'Valid_AUC': np.mean(test_results) * 100,
                   'Train_AUC': np.mean(train_results) * 100}
        run_experiment(exp_name, 'XGB', kwargs, metrics, features)
    return train_results, test_results, predictions, indexes, pd.concat(hists, axis=1)


# Function to make prediction on training dataset with target
def make_prediction_xgb(train_df: pd.DataFrame, features: list, target: str, test_df: pd.DataFrame = None,
                        random_state: tp.Union[int, None] = None, file: str = 'pred_target', valid_size: float = 0.2, *args, **kwargs):
    """
    Function to make prediction on training dataset with target and on test dataset without a target variable and
    to save test dataset results to file
    :param train_df: a training dataset with selected and preprocessed features
    :param features: selected features for training model and making predictions
    :param target: a target variable
    :param test_df: a test dataset with selected and preprocessed features without target variable
    :param random_state: random state
    :param file: file name for saving the forecast results for competition
    :param valid_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in
     the valid split. If int, represents the absolute number of test samples. If None, it will be set to 0.25.
    :param args:
    :param kwargs:
    :return: Plotting training and validation score. Saving test forecast target values
    """

    x_train, x_valid = train_test_split(train_df, test_size=valid_size, random_state=None,
                                        stratify=train_df[target].values)
    score, preds_train, preds, train_history, imp_fig, imp = run_xgb(x_train, x_valid, features, target, test=test_df,
                                                                     file=file, seed=random_state, *args, **kwargs)
    train_score = roc_auc_score(x_train[target], preds_train)
    test_score = roc_auc_score(x_valid[target], preds)
    print(f'Train score:{train_score}, Valid score: {test_score}')
