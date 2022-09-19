import pandas as pd
import numpy as np
import typing as tp
import pickle
import matplotlib.pyplot as plt
import time
import xgboost as xgb

from operator import itemgetter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

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
        metrics = {'Train_AUC': np.mean(valid_results) * 100,
                   'Valid_AUC': np.mean(train_results) * 100}
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
        unselected_features = set(train_df.columns.tolist()) - set(features)
        run_experiment(exp_name, 'XGB', kwargs, metrics, features, list(unselected_features))

    return train_results, test_results, predictions, indexes, pd.concat(hists, axis=1)


# Function to make prediction on training dataset with target
def make_prediction_xgb(train_df: pd.DataFrame, features: list, target: str, test_df: pd.DataFrame = None,
                        random_state: int = 2022, file: str = 'pred_target', valid_size: float = 0.2, *args, **kwargs):
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


############################
# Common tools
############################
# Writing results to file
def save_model_and_results(file_name: str, model_dict: dict) -> None:
    with open(f"output/models/{file_name}.p", "wb") as fp:
        pickle.dump(model_dict, fp)


# Function for plotting the AUC-ROC curve
def plot_roc_auc(results: tp.List[tp.Tuple[pd.Series, pd.Series, str]]) -> None:
    """
    Function to draw a range of ROC curve results for individual experiments.

    :param results: a list of tuples, where each tuple contain true target values, predicted target values and
    experiment name.
    """
    # Setting the drawing size
    fig, ax = plt.subplots(figsize=(10, 9))

    # Thickness of the curve
    lw = 2

    roc_score_list = []
    for true, pred, label in results:
        # Calculation of the points needed to draw the ROC curve
        # the roc_curve function returns three data series: fpr, tpr and thresholds
        fpr, tpr, thresholds = roc_curve(true, pred)
        # Calculation of the area under the curve
        roc_score = round(roc_auc_score(true, pred), 4)
        roc_score_list.append(roc_score)
        # Drawing the ROC curve
        ax.plot(fpr, tpr, lw=lw, label=f'{label}: {roc_score}')
    # Drawing a 45 degree curve as a reference point
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver operating characteristic - {max(roc_score_list)}')
    ax.legend(loc="lower right")
    plt.show()
