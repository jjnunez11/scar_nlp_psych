import pandas as pd
import numpy as np
from scipy import stats


def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std


def calculate_cohens_d(df, model_col='Model', value_col='Accuracy', n_rows_per_model=10):
    unique_models = df[model_col].unique()
    cohen_d_results = {}

    for i in range(len(unique_models) - 1):
        for j in range(i + 1, len(unique_models)):
            model1 = unique_models[i]
            model2 = unique_models[j]

            model1_data = df[df[model_col] == model1][value_col].values[:n_rows_per_model]
            model2_data = df[df[model_col] == model2][value_col].values[:n_rows_per_model]

            cohen_d_value = cohen_d(model1_data, model2_data)
            cohen_d_results[f"{model1}_vs_{model2}"] = cohen_d_value

    print(cohen_d_results)
    return cohen_d_results




# Read the CSV file into a pandas DataFrame
filename = r'C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\tables\result_tables\psych_raw_df.csv'
data = pd.read_csv(filename, delimiter=',')


# Call the function to calculate Cohen's d between models
cohen_d_results = calculate_cohens_d(data, model_col='Model', value_col='Balanced Accuracy', n_rows_per_model=10)

print(cohen_d_results)

# Convert the results to a DataFrame and export to CSV
cohen_d_df = pd.DataFrame.from_dict(cohen_d_results, orient='index', columns=["Cohen's d"])
cohen_d_df.index.name = 'Model Comparison'
cohen_d_df.to_csv('psych_cohen_d.csv')

print("Cohen's d results exported to psych_cohen_d.csv")