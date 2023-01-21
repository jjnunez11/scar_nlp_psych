import os
import pandas as pd
import numpy as np

results_dir = r"C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\results\dspln_PSYCHIATRY_60"
results_f = os.path.join(results_dir, "dspln_PSYCHIATRY_60_results.csv")


df = pd.read_csv(results_f, index_col="Run Name")

df = df[df['Table'] == "cnn_tuning"]
df = df[df['Class Imbalance Fix'] == "loss_weight"]

to_group =["CNN Weight Decay", "Learning Rate", "Dropout"]

grouped = df.groupby(by=to_group, axis=0)

lengths = grouped["AUC"].count()

grouped = grouped[["Balanced Accuracy", "AUC"]].aggregate(np.mean)

merged = pd.merge(grouped, lengths, how="inner", left_on=to_group, right_on=to_group)
merged = merged.rename({"AUC_x": "AUC", "AUC_y": "Runs"}, axis='columns')
merged = merged.sort_values("AUC")
merged.to_csv(os.path.join(results_dir, "cnn_tuning_analysis.csv"))
