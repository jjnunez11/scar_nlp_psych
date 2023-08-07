import os
import sys
import captum
import torch
import torchtext
import torchtext.data
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vocab
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from torchtext.data.utils import get_tokenizer
from models.cnn.model import CNN
from datasets.scar import SCAR
import pandas as pd
## import BERTopic

sys.path.insert(0, os.path.abspath('../'))


class MultiLIGTopic:

    def __init__(self, config):
        self.config = config
        self.tokenizer = get_tokenizer('basic_english')
        self.text_to_viz = []
        self.device = config.device

        # Criteria for sentence selection to feed to BERTTopic
        self.criteria = config.criteria
        self.cutoff = config.cutoff
        self.min_len = 1500

        print(f'Criteria used: {self.criteria}, cutoff used: {self.cutoff}')

        # Setup the model that will be used
        checkpoint = torch.load(config.model_path)
        model = CNN(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(config.device)
        # print(model)

        # Setup the vocabulary to be used
        model_config = checkpoint['config']
        scar = SCAR(model_config.batch_size, model_config.data_dir, model_config.target, eval_only=False)
        self.vocab = scar.vocab
        self.itos = self.vocab.get_itos()
        PAD_IND = self.vocab['<PAD>']
        self.token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

        # Get the Layered Intergraded Gradients for the model
        self.lig = LayerIntegratedGradients(model, model.embed)

    def call_bertopic(self, docs):

        ## opic_model = BERTopic()
        ## topics, probs = topic_model.fit_transform(docs)

        ## print(topic_model.get_topic_info())

        print('goodbye')

    def extract_impt_sens_from_doc(self, doc_text, doc_label):

        # Interpret a document, finding a LIG importance score for all words in this context
        doc_viz_record = self.interpret_doc(doc_text, doc_label)

        print('done interpreting document')
        print(f'Here is the record type: {type(doc_viz_record)}')
        print(doc_viz_record.__dir__())

        # Separate the document into sentences, and score per our criteria
        doc_sentence_df = self.score_sentences(doc_viz_record)

        # Filter out the sentences that meet our criteria
        doc_filtered_sentences = self.filter_sentences(doc_sentence_df)

        return doc_filtered_sentences

    def interpret_doc(self, doc_text, doc_label):
        text = [tok for tok in self.tokenizer(doc_text.lower())]
        if len(text) < self.min_len:
            text += ['<PAD>'] * (self.min_len - len(text))
        indexed = [self.vocab[t] for t in text]

        self.model.zero_grad()

        input_indices = torch.tensor(indexed, device=self.device)
        input_indices = input_indices.unsqueeze(0)

        seq_length = self.min_len

        # predict
        pred = self.forward_with_sigmoid(input_indices).item()
        pred_ind = round(pred)

        # generate reference indices for each sample
        reference_indices = self.token_reference.generate_reference(seq_length, device=self.device).unsqueeze(0)

        # compute attributions and approximation delta using layer integrated gradients
        attributions_ig, delta = self.lig.attribute(input_indices, reference_indices, \
                                                    n_steps=500, return_convergence_delta=True)

        attributions_ig = attributions_ig.sum(dim=2).squeeze(0)
        attributions_ig = attributions_ig / torch.norm(attributions_ig)
        attributions_ig = attributions_ig.cpu().detach().numpy()

        viz_data_record = visualization.VisualizationDataRecord(attributions_ig,
                                                                pred,
                                                                self.itos[pred_ind],
                                                                self.itos[doc_label],
                                                                self.itos[1],
                                                                attributions_ig.sum(),
                                                                text,
                                                                delta)
        return viz_data_record

    @staticmethod
    def score_sentences(doc_viz_record):
        print(doc_viz_record.__doc__)
        raw_words = doc_viz_record.raw_input_ids
        importances = doc_viz_record.word_attributions

        # Use pandas dataframe to keep track of words, their importances, and what sentances they are in
        # Using pandas will allow us to easily change criteria for included sentences.
        doc_df = pd.DataFrame()

        doc_df['Words'] = raw_words
        doc_df['Importance'] = importances
        doc_df['Sentence'] = 'not_assigned'

        sentence_i = 0
        for i in range(len(raw_words)):
            doc_df.loc[i, 'Sentence'] = sentence_i
            # do not change sentence if we are at the start or end
            if 0 < i < len(raw_words) - 1:
                if raw_words[i] == ".":
                    is_title = raw_words[i - 1] in ["mr", "mrs", 'ms', 'dr']
                    is_number = raw_words[i - 1].isdigit() and raw_words[i + 1].isdigit()
                    if not is_title and not is_number:
                        sentence_i += 1

        # Group the DataFrame by "Sentence" and calculate the required values for each group
        grouped_df = doc_df.groupby("Sentence").agg(
            Max_Positive=("Importance", "max"),  # Maximum positive importance for each sentence group
            Max_Negative=("Importance", "min"),  # Minimum negative importance for each sentence group
            Mean_Importance=("Importance", "mean")  # Mean importance for each sentence group
        )

        # Rename the columns in the grouped DataFrame
        grouped_df.rename(columns={
            "Max_Positive": "Max Positive",
            "Max_Negative": "Max Negative",
            "Mean_Importance": "Mean Importance"
        }, inplace=True)

        # Merge the grouped DataFrame back into the original DataFrame based on the "Sentence" column
        doc_df = pd.merge(doc_df, grouped_df, on="Sentence", suffixes=("", "_grouped"))

        doc_df.to_csv('doc_pd.csv')

        return doc_df

    def filter_sentences(self, doc_df):
        criterion = self.criteria
        cutoff_value = self.cutoff

        if criterion == "max_pos":
            filtered_df = doc_df[doc_df["Max Positive"] > cutoff_value]
        elif criterion == "max_neg":
            filtered_df = doc_df[doc_df["Max Negative"] < cutoff_value]
        elif criterion == "mean_above":
            filtered_df = doc_df[doc_df["Mean Importance"] > cutoff_value]
        elif criterion == "mean_below":
            filtered_df = doc_df[doc_df["Mean Importance"] < cutoff_value]
        else:
            raise ValueError(
                "Invalid criterion. Valid criteria are 'max_pos', 'max_neg', 'mean_above', and 'mean_below'.")

        filtered_sentences = filtered_df.groupby("Sentence")["Words"].apply(' '.join).tolist()
        return filtered_sentences


    def forward_with_sigmoid(self, input):
        return torch.sigmoid(self.model(input))
