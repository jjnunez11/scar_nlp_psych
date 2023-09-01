import os
import pandas as pd
import numpy as np
from copy import deepcopy


def tuning_analysis(r_dir, r_f, tuning_table):
    df = pd.read_csv(r_f, index_col="Run Name")
    df = df[df['Table'] == tuning_table]

    if tuning_table in ["lf_tuning", "bert_lf_tuning"]:
        df = df[df['Class Imbalance Fix'] == 'undersampling']
        df = df[df['Patience'] == 10]
    else:
        df = df[df['Class Imbalance Fix'] == "loss_weight"]

    if tuning_table == "cnn_tuning":
        to_group = ["CNN Weight Decay", "Learning Rate", "Dropout"]
    elif tuning_table == "lstm_tuning":
        to_group = ["Dropout", "LSTM Wt Dropout", "LSTM Embed Dropout", "Learning Rate"]
    elif tuning_table in ["bert_tuning", "lf_tuning", "bert_lf_tuning"]:
        to_group = ["CNN Weight Decay", "Learning Rate"]
    elif tuning_table == "bow_tuning":
        to_group = ['BoW Classifier', 'BoW LR C', 'BoW RF Estimators', 'Max Tokens']
    else:
        raise ValueError(f"Unrecognized tuning table input: {tuning_table}")

    grouped = df.groupby(by=to_group, axis=0)
    grouped = grouped[["Balanced Accuracy", "AUC"]].agg([np.mean, np.std, 'count'])
    grouped = grouped.drop(('Balanced Accuracy', 'count'), axis=1)
    grouped = grouped.sort_values([('AUC', 'mean')], ascending=False)
    grouped.to_csv(os.path.join(r_dir, f"{table}_analysis.csv"))


if __name__ == "__main__":

    # table = "bert_tuning"
    # table = "bow_tuning"
    # table = "cnn_tuning"
    # table = "lstm_tuning"
    # table = "lf_tuning"
    table = "bert_lf_tuning"
    target = "psych"
    # target = "sw"

    if target == "psych":
        results_dir = r"C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\results\dspln_PSYCHIATRY_12"
        results_f = os.path.join(results_dir, "dspln_PSYCHIATRY_12_results.csv")
    elif target == "sw":
        results_dir = r"C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\results\dspln_SOCIALWORK_12"
        results_f = os.path.join(results_dir, "dspln_SOCIALWORK_12_results.csv")
    else:
        results_dir = None
        raise ValueError(f"Supports psych and sw targets right now, provided {target}")

    tuning_analysis(results_dir, results_f, table)

    print("Tuning analysis complete!")