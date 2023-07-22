import os
import pandas as pd
from tables.table_globals import RESULT_TABLES_DIR


def generate_dictator_counts(doc_for_pt_path, document_details_path, out_path):
    # Read the datasets
    doc_for_pt = pd.read_csv(doc_for_pt_path)
    document_details = pd.read_csv(document_details_path)

    # Merge datasets based on document_id
    merged_df = doc_for_pt.merge(
        document_details[['document_id', 'author_dictator', 'fasr_reporter', 'fasr_discipline']],
        on='document_id', how='left')

    # Count unique values in fasr_reporter for each fasr_discipline
    unique_reporter_by_discipline = merged_df.groupby('fasr_discipline')['fasr_reporter'].nunique()

    # Count unique values in author_dictator for each fasr_discipline
    unique_author_by_discipline = merged_df.groupby('fasr_discipline')['author_dictator'].nunique()

    # Count rows and calculate percentage for each fasr_discipline
    rows_count_by_discipline = merged_df['fasr_discipline'].value_counts()
    rows_percentage_by_discipline = (rows_count_by_discipline / len(merged_df)) * 100

    # Write results to output file
    with open(output_file, 'w') as file:
        file.write("Unique values in 'fasr_reporter' for each 'fasr_discipline':\n")
        file.write(str(unique_reporter_by_discipline) + '\n\n')
        file.write("Unique values in 'author_dictator' for each 'fasr_discipline':\n")
        file.write(str(unique_author_by_discipline) + '\n\n')
        file.write("Rows count and percentage for each 'fasr_discipline' in doc_for_pt:\n")
        file.write("Discipline\tRows Count\tPercentage\n")
        for discipline, count, percentage in zip(rows_count_by_discipline.index, rows_count_by_discipline,
                                                 rows_percentage_by_discipline):
            file.write(f"{discipline}\t{count}\t{percentage:.2f}%\n")

    print("Results written to", output_file)


if __name__ == '__main__':
    # Define input file paths
    doc_for_pt_file = r'C:\Users\jjnunez\PycharmProjects\scar_nlp_data\interm_data\doc_for_pt_after_dx_180_days.csv'
    document_details_file = r'C:\Users\jjnunez\SCAR_data\q18-0193_document_details.csv'
    output_file = os.path.join(RESULT_TABLES_DIR, "dictator_counts.txt")

    # Call the function
    generate_dictator_counts(doc_for_pt_file, document_details_file, output_file)
