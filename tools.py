import pandas as pd
import numpy as np
import typing as tp
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_au(results):

    # ''' Funkcja, do rysowania szeregu wyników krzywych ROC dla poszczególnych eksperymentów
    # results - lista wyników jako 3 elementowe tuple (true, pred, label)
    # '''

    # Ustalanie wielkości rysunku
    # fig, ax = plt.subplots(figsize=(10, 9))
    #
    # # Grubość krzywej
    # lw = 2
    # roc_score_list = []
    # for true, pred, label in results:
    #     # Obliczenie punktów potrzebnych do narysowania krzywej ROC
    #     # funkcja roc_curve zwarca trzy serie danych, fpr, tpr oraz poziomy progów odcięcia
    #     fpr, tpr, thresholds = roc_curve(true, pred)
    #     # Obliczamy pole powierzchni pod krzywą
    #     roc_score = round(roc_auc_score(true, pred), 3)
    #     roc_score_list.append(roc_score)
    #     # Rysujemy krzywą ROC
    #     ax.plot(fpr, tpr, lw=lw, label=f'{label}: {roc_score}')
    # # Rysujemy krzywą 45 stopni jako punkt odniesienia
    # ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # ax.set_xlim([-0.01, 1.0])
    # ax.set_ylim([0.0, 1.01])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title(f'Receiver operating characteristic - {max(roc_score_list)}')
    # ax.legend(loc="lower right")
    # plt.show()
    pass
