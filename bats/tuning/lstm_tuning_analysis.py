import os
import pandas as pd
import numpy as np

results_dir = r"C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\results\dspln_PSYCHIATRY_60"
results_f = os.path.join(results_dir, "dspln_PSYCHIATRY_60_results.csv")


df = pd.read_csv(results_f, index_col="Run Name")

df = df[df['Table'] == "lstm_tuning"]
df = df[df['Class Imbalance Fix'] == "loss_weight"]

to_group =["Dropout", "LSTM Wt Dropout", "LSTM Embed Dropout", "Learning Rate"]

grouped = df.groupby(by=to_group, axis=0)

lengths = grouped["AUC"].count()

grouped = grouped[["Balanced Accuracy", "AUC"]].aggregate(np.mean)

## grouped = grouped.sort_values(['AUC'])
## lengths = lengths.sort_values(['AUC'])

grouped.to_csv(os.path.join(results_dir, "tuning_analysis.csv"))
lengths.to_csv(os.path.join(results_dir, "lengths.csv"))

merged = pd.merge(grouped, lengths, how="inner", left_on=to_group, right_on=to_group)

merged.to_csv(os.path.join(results_dir, "merged.csv"))
