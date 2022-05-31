import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
     average_precision_score,
     precision_recall_curve,
     roc_auc_score,
     roc_curve
)


def get_pred_true(pred, thresh):
    return pred > thresh


def get_true_pos(y, pred, thresh=0.5):
    return np.sum(get_pred_true(pred, thresh) & (y == 1))


def get_true_neg(y, pred, thresh=0.5):
    return np.sum(not get_pred_true(pred, thresh) & (y == 0))


def get_false_neg(y, pred, thresh=0.5):
    return np.sum(not get_pred_true(pred, thresh) & (y == 1))


def get_false_pos(y, pred, thresh):
    return np.sum(not get_pred_true(pred, thresh) & (y == 0))


def get_performance_metrics(y, pred, lbls,
                            tp=get_true_pos, tn=get_false_neg,
                            fp=get_false_pos, fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None,
                            f1=None, thresholds=None):
    if len(thresholds) != len(lbls):
        thresholds = [.5] * len(lbls)

    columns = ['Class', 'TP', 'TN', 'FP', 'FN', 'Accuracy',
               'Prevalence', 'Sensitivity', 'Specificity',
               'PPV', 'NPV', 'AUC', 'F1', 'Thresholds']

    df = pd.DataFrame(columns=columns)

    for idx in range(len(lbls)):
        df = df.append({
            'Class': lbls[idx],
            'TP': round(tp(y[:, idx], pred[:, idx]), 3) if tp != None else 'Not Defined',
            'TN': round(tn(y[:, idx], pred[:, idx]), 3) if tn != None else 'Not Defined',
            'FP': round(fp(y[:, idx], pred[:, idx]), 3) if fp != None else 'Not Defined',
            'FN': round(fn(y[:, idx], pred[:, idx]), 3) if fn != None else 'Not Defined',
            'Accuracy': round(acc(y[:, idx], pred[:, idx], thresholds[idx]), 3) if acc != None else "Not Defined",
            'Prevalence': round(prevalence(y[:, idx]), 3) if prevalence != None else "Not Defined",
            'Sensitivity': round(sens(y[:, idx], pred[:, idx], thresholds[idx]), 3) if sens != None else "Not Defined",
            'Specificity': round(spec(y[:, idx], pred[:, idx], thresholds[idx]), 3) if spec != None else "Not Defined",
            'PPV': round(ppv(y[:, idx], pred[:, idx], thresholds[idx]), 3) if ppv != None else "Not Defined",
            'NPV': round(npv(y[:, idx], pred[:, idx], thresholds[idx]), 3) if npv != None else "Not Defined",
            'AUC': round(auc(y[:, idx], pred[:, idx]), 3) if auc != None else "Not Defined",
            'F1': round(f1(y[:, idx], pred[:, idx] > thresholds[idx]), 3) if f1 != None else "Not Defined",
            'Thresholds': round(thresholds[idx], 3)
        }, ignore_index=True)

    return df.set_index('Class', inplace=True)





