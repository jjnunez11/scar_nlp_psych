"""
Script to determine the number of 

"""
import torch

from tables.table_globals import RESULT_TABLES_DIR
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.scar_bow import StemTokenizer


def count_bert_tokens(model_name, config, dataset):
    train_data = dataset.train_dataset
    dev_data = dataset.dev_dataset
    test_data = dataset.test_dataset
    n_consults = len(train_data) + len(dev_data) + len(test_data)

    token_counts = []

    def count_bert_dataset(a_dataset, token_counts_so_far):
        for i in range(len(a_dataset)):
            doc_tensor = a_dataset[i]
            token_tensor = doc_tensor['input_ids']
            token_array = token_tensor.numpy()
            token_array = np.trim_zeros(token_array, 'b')
            n_tokens = len(token_array)
            token_counts_so_far.append(n_tokens)

        return token_counts_so_far

    token_counts = count_bert_dataset(train_data, token_counts)
    token_counts = count_bert_dataset(dev_data, token_counts)
    token_counts = count_bert_dataset(test_data, token_counts)

    token_counts = np.array(token_counts)
    generate_count_files(model_name, config, token_counts, n_consults)


def trim_padding_from_np(arr, pad_number):
    reversed_arr = arr[::-1]  # Reverse the array to simplify the process
    index = 0

    # Find the index of the first non-three element
    for i in range(len(reversed_arr)):
        if reversed_arr[i] != pad_number:
            index = i
            break

    # Remove the consecutive threes from the end
    trimmed_arr = arr[:-index] if index > 0 else arr

    return trimmed_arr


def count_neural_tokens(model_name, config, train_loader, dev_loader, test_loader):
    train_loader = list(train_loader)
    dev_loader = list(dev_loader)
    test_loader = list(test_loader)
    n_consults = 0
    token_counts = []

    def count_neural_dataloader(dataloader, token_counts_so_far, n_consults_so_far):
        for batch, _ in tqdm(dataloader, leave=True, desc=f'Counting tokens for {model_name}'):
            batch = torch.transpose(batch, 0, 1)
            batch = batch.detach().cpu().numpy()
            for consult in batch:
                # Must remove the padding, which is token number 3
                trimmed_consult = trim_padding_from_np(consult, 3)
                token_counts.append(len(trimmed_consult))
                n_consults_so_far = n_consults_so_far + 1

        return token_counts_so_far, n_consults_so_far

    token_counts, n_consults = count_neural_dataloader(train_loader, token_counts, n_consults)
    token_counts, n_consults = count_neural_dataloader(dev_loader, token_counts, n_consults)
    token_counts, n_consults = count_neural_dataloader(test_loader, token_counts, n_consults)

    token_counts = np.array(token_counts)
    generate_count_files(model_name, config, token_counts, n_consults)


def count_bow_tokens(model_name, config, scar_bow):
    train_data = scar_bow.get_train_data()
    train_text = train_data["text"]
    dev_data = scar_bow.get_dev_data()
    dev_text = dev_data["text"]
    test_data = scar_bow.get_test_data()
    test_text = test_data["text"]

    stemmer = StemTokenizer()
    n_consults = len(train_text) + len(dev_text) + len(test_text)
    token_counts = []

    def count_bow_textset(textset, token_counts_so_far):
        for i in tqdm(range(len(textset)), desc=f'Counting tokens for {model_name}'):
            stemmed_consult = stemmer(textset[i])
            token_counts_so_far.append(len(stemmed_consult))

        return token_counts_so_far

    token_counts = count_bow_textset(train_text, token_counts)
    token_counts = count_bow_textset(dev_text, token_counts)
    token_counts = count_bow_textset(test_text, token_counts)

    token_counts = np.array(token_counts)
    generate_count_files(model_name, config, token_counts, n_consults)


def generate_count_files(model_name, config, token_counts, n_consults):
    f_stem = f'{model_name}_{config.target}_'
    f_fig = os.path.join(RESULT_TABLES_DIR, f_stem + "n_token.png")
    f_stats = os.path.join(RESULT_TABLES_DIR, f_stem + "n_token.txt")

    # Write some summary statistics
    f_stats = open(f_stats, 'w')
    f_stats.write(f"Summary Statistics for the Token Count when Training "
                  f"Documents are Tokenized using {model_name} Tokenizer \n")
    f_stats.write(f'Based on counting tokens from {n_consults} documents\n')
    f_stats.write(f'Mean is {token_counts.mean()}\n')
    f_stats.write(f'Median is {np.median(token_counts)}\n')
    f_stats.write(f'SD is {np.std(token_counts)}\n')
    n_eql_512 = (token_counts <= 512).sum()
    perc_eql_512 = n_eql_512/n_consults
    f_stats.write(f'Number <= 512 {n_eql_512}\n')
    f_stats.write(f'Percentage <= 512 {perc_eql_512}\n')

    # Make Histogram
    _ = plt.hist(token_counts, bins=30)  # arguments are passed to np.histogram
    plt.axvline(512, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('Tokens in Document')
    plt.ylabel('Number of Documents')
    # plt.title("Histogram with 'auto' bins")
    plt.savefig(f_fig)
    plt.close()
    f_stats.close()
