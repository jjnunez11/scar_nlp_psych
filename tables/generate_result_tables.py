import os
import pandas as pd
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_1samp
import statsmodels.api as sm

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
    models = ["BoW", "CNN", "LSTM", "BERT"]  # "Rule", will need to adjust to do one sided t-tests? Or just put rule in its own table
    max_tokens_models = ["Longformer_512", "Longformer_1024", "Longformer_2048", "Longformer_4096", "CNN_us", "BERT_512_us"]
    # Spacers for writing to copy-pastes for the tables
    horiz_sp = "\t"  # What to separate columns
    vert_sp = "\n"  # What to place at the end of the line to seperate rows vertically
    # Formatting for numbers for tables
    num_format = '{:<05}'

    # CSV files to read raw results from
    psych_raw_results_csv = os.path.join(r"../", "results", "final_results", "dspln_PSYCHIATRY_12", "dspln_PSYCHIATRY_12_results.csv")
    sw_raw_results_csv = os.path.join(r"../", "results", "final_results", "dspln_SOCIALWORK_12", "dspln_SOCIALWORK_12_results.csv")
    max_tokens_results_csv = os.path.join(r"../", "tables", "result_tables",  "max_tokens_raw_df.csv")
    # Folder to print result tables
    out_dir = os.path.join("./result_tables")

    def __init__(self):

        # Prepare raw results DataFrame
        self.psych_raw_df = self.extract_raw_results(self.psych_raw_results_csv)
        self.sw_raw_df = self.extract_raw_results(self.sw_raw_results_csv)

        self.psych_raw_df.to_csv(os.path.join(self.out_dir, "psych_raw_df.csv"))
        self.sw_raw_df.to_csv(os.path.join(self.out_dir, "sw_raw_df.csv"))

        # For the max token results, I manually moved it to a new df in the table folder so I could rename the model
        # to be Longformer_512, Longformer_1024 just to need less manual adjustment of these resulting tables
        self.max_tokens_raw_df = self.extract_raw_results(self.max_tokens_results_csv)

    def extract_raw_results(self, raw_csv):
        raw_df = pd.read_csv(raw_csv, index_col="Run Name")
        cols_to_keep = self.filter_cols + self.supp_metrics

        raw_df = raw_df[cols_to_keep]

        return raw_df

    def generate_result_table(self, target):

        if target == 'psych':
            df = self.psych_raw_df
            models = self.models
        elif target == 'sw':
            df = self.sw_raw_df
            models = self.models
        elif target == 'max_tokens':
            df = self.max_tokens_raw_df
            models = self.max_tokens_models
        else:
            raise ValueError

        # Group and calculate standard deviation for each metrics
        df = df.groupby(by="Model", axis=0)
        df = df[self.paper_metrics].agg([np.mean, np.std])
        df = df.reindex(models)
        df = df.round(decimals=4)
        df.to_csv(os.path.join(self.out_dir, f"{target}_results_table.csv"))

        # Write-out a copy-paste version formatted
        df = df.round(decimals=3)
        f = open(os.path.join(self.out_dir, f"{target}_results.table.txt"), "w")

        # Rename and write the metrics
        renamed_paper_metrics = ["BAC" if x == "Balanced Accuracy" else x for x in self.paper_metrics]
        renamed_paper_metrics = ["Sensitivity" if x == "Recall" else x for x in renamed_paper_metrics]
        f.write("\t".join(["Model"] + renamed_paper_metrics))

        f.write(self.vert_sp)

        # Write the results
        for model in models:
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
        self.generate_result_table("max_tokens")

    def generate_p_table(self, target):

        if target == 'psych':
            df = self.psych_raw_df
            models = self.models
        elif target == 'sw':
            df = self.sw_raw_df
            models = self.models
        elif target == 'max_tokens':
            df = self.max_tokens_raw_df
            models = self.max_tokens_models
        else:
            raise ValueError

        f = open(os.path.join(self.out_dir, f"{target}_pvalue_cohend_table.txt"), "w")
        header = self.horiz_sp.join(['Model'] + models) + self.vert_sp

        f.write("Two-tailed t-test p-value")
        f.write(self.vert_sp)
        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric + self.vert_sp)
            f.write(header)
            df_metric = df[["Model", metric]]
            for model_a in models:
                f.write(model_a)
                for model_b in models:
                    df_model_a = df_metric[df_metric["Model"] == model_a][metric]
                    df_model_b = df_metric[df_metric["Model"] == model_b][metric]
                    if model_a == model_b:
                        p_value = "-"
                    else:
                        p_value = ttest_rel(df_model_a, df_model_b).pvalue
                        p_value = p_value.round(6)

                    f.write(f"{self.horiz_sp}{p_value}")
                f.write(self.vert_sp)

        f.write(self.vert_sp)
        f.write(self.horiz_sp + "Cohen's d Effect Size")
        f.write(self.vert_sp)

        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric + self.vert_sp)
            f.write(header)
            df_metric = df[["Model", metric]]
            for model_a in models:
                f.write(model_a)
                for model_b in models:
                    df_model_a = df_metric[df_metric["Model"] == model_a][metric]
                    df_model_b = df_metric[df_metric["Model"] == model_b][metric]
                    if model_a == model_b:
                        cohen_d_value = "-"
                    else:
                        cohen_d_value = self.cohen_d(df_model_a, df_model_b)
                        cohen_d_value = cohen_d_value.round(2)

                    f.write(f"{self.horiz_sp}{cohen_d_value}")
                f.write(self.vert_sp)

        f.close()

    def generate_rule_p_values_cohen(self, target):

        if target == 'psych':
            df = self.psych_raw_df
            models = self.models
            bac = 0.5416
            auc = 0.5416
        elif target == 'sw':
            df = self.sw_raw_df
            models = self.models
            bac = 0.5532
            auc = 0.5532
        else:
            raise ValueError

        f = open(os.path.join(self.out_dir, f"{target}_pvalue_cohen_rule.txt"), "w")
        header = self.horiz_sp.join(['Model'] + models) + self.vert_sp

        f.write("One-tailed t-test p-value")
        f.write(self.vert_sp)
        for metric in ["Balanced Accuracy", "AUC"]:
            if metric == "Balanced Accuracy":
                rule_based_metric = bac
            elif metric == "AUC":
                rule_based_metric = auc
            else:
                raise ValueError("Must be either Balanced Accuracy or AUC")
            f.write(metric + self.vert_sp)
            f.write(header)
            df_metric = df[["Model", metric]]
            for model_a in models:
                df_model = df_metric[df_metric["Model"] == model_a][metric]
                f.write(model_a)
                p_value = ttest_1samp(a=df_model, popmean=rule_based_metric).pvalue
                p_value = p_value.round(6)

                f.write(f"{self.horiz_sp}{p_value}")
                f.write(self.vert_sp)

        f.write(self.vert_sp)
        f.write(self.horiz_sp + "Cohen's d Effect Size")
        f.write(self.vert_sp)

        for metric in ["Balanced Accuracy", "AUC"]:
            if metric == "Balanced Accuracy":
                rule_based_metric = bac
            elif metric == "AUC":
                rule_based_metric = auc
            else:
                raise ValueError("Must be either Balanced Accuracy or AUC")
            f.write(metric + self.vert_sp)
            f.write(header)
            df_metric = df[["Model", metric]]
            for model_a in models:
                df_model = df_metric[df_metric["Model"] == model_a][metric]
                f.write(model_a)
                cohen_d_value = self.one_sided_cohen_d(rule_based_metric, df_model).round(2)

                f.write(f"{self.horiz_sp}{cohen_d_value}")
                f.write(self.vert_sp)

        f.close()

    def generate_p_table_both_dsplns(self):
        df_psych = self.psych_raw_df
        df_sw = self.sw_raw_df

        f = open(os.path.join(self.out_dir, f"both_dsplns_pvalue_cohend_table.txt"), "w")
        f.write(self.horiz_sp + "P-value" + self.vert_sp)

        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric+self.vert_sp)
            for model in self.models:
                df_psych_metric = df_psych[df_psych["Model"] == model][metric]
                df_sw_metric = df_sw[df_sw["Model"] == model][metric]
                p_value = ttest_rel(df_psych_metric, df_sw_metric).pvalue
                f.write(f"{model}{self.horiz_sp}{p_value.round(3)}{self.vert_sp}")

        f.write(self.vert_sp)
        f.write(self.horiz_sp + "Cohen's d Effect Size")
        f.write(self.vert_sp)

        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric+self.vert_sp)
            for model in self.models:
                df_psych_metric = df_psych[df_psych["Model"] == model][metric]
                df_sw_metric = df_sw[df_sw["Model"] == model][metric]
                if model == "BoW":
                    cohend_value = "inf"
                else:
                    cohend_value = self.cohen_d(df_psych_metric, df_sw_metric).round(2)
                f.write(f"{model}{self.horiz_sp}{cohend_value}{self.vert_sp}")

        f.close()

    def generate_regression(self):

        df = pd.read_csv(self.max_tokens_results_csv, index_col="Run Name")

        # Only longformers
        lf_df = df[df['Model'].str.contains("Longformer")]
        X = lf_df['Max Tokens']
        X = sm.add_constant(X)

        f = open(os.path.join(self.out_dir, f"longformer_regression_max_tokens.txt"), "w")
        header = "Simple Regression Values with independent variable max tokens, dependent Balanced Accuracy and AUC"\
                 + self.vert_sp

        f.write("Simple regression p-value")
        f.write(self.vert_sp)
        for metric in ["Balanced Accuracy", "AUC"]:
            f.write(metric + self.vert_sp)
            f.write(header)

            y = lf_df[metric]

            model = sm.OLS(y, X).fit()
            f.write(str(model.summary()))
            f.write(self.vert_sp)
            f.write(f'P-value is: {model.pvalues["Max Tokens"]}')
            f.write(self.vert_sp)

        f.close()

    def generate_p_tables(self):
        self.generate_p_table('psych')
        self.generate_p_table('sw')
        self.generate_p_table('max_tokens')
        self.generate_p_table_both_dsplns()
        self.generate_rule_p_values_cohen("psych")
        self.generate_rule_p_values_cohen("sw")

    @staticmethod
    def cohen_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std

    @staticmethod
    def one_sided_cohen_d(popmean, group):
        mean = np.mean(group)
        std = np.std(group, ddof=1)
        return (mean - popmean) / std


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

    generator = ResultsGenerator()
    generator.generate_result_tables()
    generator.generate_p_tables()
    generator.generate_regression()

    print("Printed table LaTeX string to file!")


