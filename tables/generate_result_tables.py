import os
import pandas as pd
import re
import numpy as np
from scipy.stats import ttest_rel

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
    # Spacers for writing to copy-pastes for the tables
    horiz_sp = "\t"  # What to separate columns
    vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically
    # Formatting for numbers for tables
    num_format = '{:<05}'

    # CSV files to read raw results from
    psych_raw_results_csv = os.path.join(r"../", "results", "final_results", "dspln_PSYCHIATRY_12", "dspln_PSYCHIATRY_12_results.csv")
    sw_raw_results_csv = os.path.join(r"../", "results", "final_results", "dspln_SOCIALWORK_12", "dspln_SOCIALWORK_12_results.csv")

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

        f.write(self.vert_sp)

        # Write the results in JAMA format
        for model in self.models:
            f.write(model)
            for metric in self.paper_metrics:
                mean = str(df.loc[model, metric]['mean'])
                std = str(df.loc[model, metric]['std'])
                mean = self.num_format.format(mean)
                std = self.num_format.format(std)
                s = f'{self.horiz_sp}{mean} ({std})'
                f.write(s)
            f.write(self.vert_sp)

        f.close()

    def generate_result_tables(self):
        self.generate_result_table("psych")
        self.generate_result_table("sw")

    def generate_p_table(self, target):

        if target == 'psych':
            df = self.psych_raw_df
        elif target == 'sw':
            df = self.sw_raw_df
        else:
            raise ValueError

        f = open(os.path.join(self.out_dir, f"{target}_pvalue_table.txt"), "w")
        header = self.horiz_sp.join(['Model'] + self.models) + self.vert_sp

        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric + self.vert_sp)
            f.write(header)
            df_metric = df[["Model", metric]]
            for model_a in self.models:
                f.write(model_a)
                for model_b in self.models:
                    df_model_a = df_metric[df_metric["Model"] == model_a][metric]
                    df_model_b = df_metric[df_metric["Model"] == model_b][metric]
                    p_value = ttest_rel(df_model_a, df_model_b).pvalue
                    f.write(f"{self.horiz_sp}{p_value.round(6)}")
                f.write(self.vert_sp)

        # Also add in Cohen's d for effect size calculations
        # WE NEED TO DO THIS ST ILL
        
        f.close()

    def generate_p_table_both_dsplns(self):
        df_psych = self.psych_raw_df
        df_sw = self.sw_raw_df

        f = open(os.path.join(self.out_dir, f"both_dsplns_pvalue_table.txt"), "w")
        f.write(self.horiz_sp + "P-value" + self.vert_sp)

        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric+self.vert_sp)
            for model in self.models:
                df_psych_metric = df_psych[df_psych["Model"] == model][metric]
                df_sw_metric = df_sw[df_sw["Model"] == model][metric]
                p_value = ttest_rel(df_psych_metric, df_sw_metric).pvalue
                f.write(f"{model}{self.horiz_sp}{p_value.round(3)}{self.vert_sp}")
                # f.write(self.vert_sp)

        f.close()

    def generate_p_tables(self):
        self.generate_p_table('psych')
        self.generate_p_table('sw')
        self.generate_p_table_both_dsplns()


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
    generator.generate_p_tables()

    print("Printed table LaTeX string to file!")


