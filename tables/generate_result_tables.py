import os
import pandas as pd
import re
import numpy as np

pd.set_option("display.max_colwidth", 10000)


class ResultsGenerator(object):
    # Metrics to include in paper results tables
    paper_metrics = ["Accuracy", "Balanced Accuracy", "AUC", "F1", "Recall", "Specificity"]
    # Metrics to include in supplementary results tables
    supp_metrics = ["Accuracy", "Balanced Accuracy", "AUC", "F1", "Recall", "Precision", "Specificity",
                    "PPV", "NPV", "TP", "TN", "FP",	"FN"]
    # Columns needed to filter things correctly
    filter_cols = ["Table", "Model"]
    # Models
    models = ["BoW", "CNN", "LSTM", "BERT"]

    # CSV files to read raw results from
    psych_raw_results_csv = os.path.join(r"../", "results", "dspln_PSYCHIATRY_12", "dspln_PSYCHIATRY_12_results.csv")
    sw_raw_results_csv = os.path.join(r"../", "results", "dspln_SOCIALWORK_12", "dspln_SOCIALWORK_12_results.csv")

    # Folder to print result tables
    out_dir = os.path.join("./result_tables")

    def __init__(self):

        # Prepare raw results DataFrame
        self.psych_raw_df = self.extract_raw_results(self.psych_raw_results_csv)
        self.sw_raw_df = self.extract_raw_results(self.sw_raw_results_csv)

        self.psych_raw_df.to_csv(os.path.join(self.out_dir, "psych_raw_df.csv"))
        self.sw_raw_df.to_csv(os.path.join(self.out_dir, "sw_raw_df.csv"))

    def extract_raw_results(self, raw_csv):
        raw_df = pd.read_csv(raw_csv, index_col="Run Name")
        cols_to_keep = self.filter_cols + self.supp_metrics

        raw_df = raw_df[cols_to_keep]

        return raw_df

    def generate_result_table(self, target):

        if target == 'psych':
            df = self.psych_raw_df
        elif target == 'sw':
            df = self.sw_raw_df
        else:
            raise ValueError

        # Group and calculate standard deviation for each metrics
        df = df.groupby(by="Model", axis=0)
        df = df[self.paper_metrics].agg([np.mean, np.std])
        df = df.reindex(self.models)
        df = df.round(decimals=4)
        df.to_csv(os.path.join(self.out_dir, f"{target}_results_table.csv"))

        # Write-out a copy-paste version formatted for JAMA
        df = df.round(decimals=3)
        f = open(os.path.join(self.out_dir, f"{target}_results.table.txt"), "w")

        # Rename and write the metrics
        renamed_paper_metrics = ["BAC" if x == "Balanced Accuracy" else x for x in self.paper_metrics]
        renamed_paper_metrics = ["Sensitivity" if x == "Recall" else x for x in renamed_paper_metrics]
        f.write("\t".join(["Model"] + renamed_paper_metrics))

        horiz_sp = "\t"  # What to separate columns
        vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically
        f.write(vert_sp)

        # Write the results in JAMA format
        for model in self.models:
            f.write(model)
            for metric in self.paper_metrics:
                mean = str(df.loc[model, metric]['mean'])
                std = str(df.loc[model, metric]['std'])
                num_format = '{:<05}'
                mean = num_format.format(mean)
                std = num_format.format(std)
                # print(std)
                s = f'{horiz_sp}{mean} ({std})'
                f.write(s)
            f.write(vert_sp)

        f.close()

    def generate_result_tables(self):
        self.generate_result_table("psych")
        self.generate_result_table("sw")

def generate_result_table(table):
    """
    Scans the results files and extracts the results that are part of the given table,
    and then writes the LaTeX string to a text file for copy pasting

    """
    f = open(r"result_tables\\" + table + ".txt", "w")

    table_df = pd.DataFrame([])

    for root, dirs, files in os.walk("../results"):
        for file in files:
            if file[-11:] == 'results.csv' and not('old_results' in root):
                df = pd.read_csv(os.path.join(root, file))
                try:
                    df = df[df['Table'] == table]
                    df = df.drop_duplicates(['Model', 'Table Extra'], keep='last')
                    df = df[['Model', 'Table Extra', 'LaTeX String']]
                    table_df = table_df.append(df)
                    if len(df.index) > 0:
                        print(f'found results in {file}')
                except KeyError:
                    pass

    # Sort by Model, we'll have the order go BoW, CNN, LSTM, BERT. And then sort by the extra desc
    table_df['sort'] = 5
    table_df['sort'] = table_df['Model'].map({"BoW": 0, "CNN": 1, "LSTM": 2, "BERT": 3})
    table_df = table_df.sort_values(['sort', 'Table Extra'])

    latex_strings = table_df["LaTeX String"]

    horiz_sp = "\t"  # What to separate columns
    vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically

    if len(latex_strings) > 0:
        s = latex_strings.to_string(header=False, index=False)
        s = re.sub(r"^\s*", "", s)
        s = re.sub(f"\n\s*", "", s)

        # Replace latex horizontal cell separator with specified
        s = re.sub(r"\s\&\s", horiz_sp, s)

        # Replace latex vertical cell separator with specified
        s = re.sub(r"\s\\\\", vert_sp, s)

        f.write(s + "\n")

    f.close()


if __name__ == "__main__":

    # generate_result_table('survival_dif_lengths')
    # generate_result_table('see_psych')
    # generate_result_table('see_counselling')
    # generate_result_table('compare_sexes')
    # generate_result_table('need_emots_all_models')
    # generate_result_table('need_infos_all_models')
    # generate_result_table('need_emots_dif_n'
    # generate_result_table('survival_all_models')
    # generate_result_table('need_emots_dif_n')
    # generate_result_table('need_infos_dif_n')
    # generate_result_table('compare_n')
    # generate_result_table('compare_stages')
    # generate_result_table('survival_dif_lengths')

    # generate_result_table("imbalance_fix")
    # generate_result_table("compare_stages_emots")

    # if False:  # all tables for paper
    #    generate_result_table('survival_all_models')
    #    generate_result_table('survival_dif_lengths')

    # generate_result_table("compare_sex_controls")

    generator = ResultsGenerator()
    generator.generate_result_tables()

    print("Printed table LaTeX string to file!")


