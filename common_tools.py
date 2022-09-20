import pandas as pd
import typing as tp
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


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