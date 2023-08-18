import copy
import os
import sys
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

from multiligtopic.openai_api_key import OPENAI_API_KEY
from models.cnn.model import CNN
from datasets.scar import SCAR
import pandas as pd
from bertopic import BERTopic
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer


sys.path.insert(0, os.path.abspath('../'))


class MultiLIGTopic:

    def __init__(self, config):
        self.config = config
        self.tokenizer = get_tokenizer('basic_english')
        self.text_to_viz = []
        self.device = config.device
        self.target = config.target
        self.filt_sents_f = None
        self.topic_model = None
        self.top_topics = None

        # Either extract the sentences fresh, or load from file
        if config.load_sents:
            # Read the file and store each line (sentence) in a list
            with open(config.load_file, 'r') as file:
                sentences = file.readlines()
            # Remove newline characters and any leading/trailing whitespace from each sentence
            self.sents = [sentence.strip() for sentence in sentences]
            self.filt_sents_f = config.load_file  # this is the filename for the filtered sentenc
        else:
            self.sents = []

            if config.device == 'gpu':
                self.device = torch.device('cuda:0')
                print("Using a CUDA GPU, woot!")
            elif config.device == 'cpu':
                self.device = 'cpu'
                print("Using a CPU, sad!")
            else:
                raise ValueError('Device argument must be cpu or gpu')

            # Criteria for sentence selection to feed to BERTTopic
            self.criteria = config.criteria
            self.cutoff = config.cutoff
            self.gpu_vec_len_limit = 1500  # limit for current gpu
            print(f'Criteria used: {self.criteria}, cutoff used: {self.cutoff}')

            # Data to interpret
            self.data_dir = os.path.join(config.data_dir, self.target)
            self.f_in = os.path.join(self.data_dir, 'test.tsv')

            # Dir to output
            self.results_dir = os.path.join(config.results_dir, self.target, "MultiLIGTopic")
            if not os.path.exists(self.results_dir):
                os.mkdir(self.results_dir)
            f_out_name = f'impt_sents_{self.target}_{self.criteria}_{self.cutoff}'
            self.filt_sents_f = os.path.join(self.results_dir, f_out_name)

            # Setup the model that will be used
            checkpoint = torch.load(config.model_path)
            model = CNN(config=checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.model = model.to(self.device)
            if config.device == "gpu":
                # Due to gpu limits, we will have a backup copy of the model on cpu
                # to use when a doc is too big for our gpu
                model_for_cpu = copy.deepcopy(model)
                self.model_cpu = model_for_cpu.to("cpu")

            # Setup the vocabulary to be used
            model_config = checkpoint['config']
            scar = SCAR(model_config.batch_size, model_config.data_dir, model_config.target, eval_only=False)
            self.vocab = scar.vocab
            self.itos = self.vocab.get_itos()
            PAD_IND = self.vocab['<PAD>']
            self.token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

            # Get the Layered Integrated Gradients for the model
            self.lig = LayerIntegratedGradients(model, model.embed)

            self.extract_sents_from_docs(self.f_in)

    def fit_topic_model(self, docs):

        # Prepare UMAP
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        # Prepare HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)

        # Prepare Vectorizer
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

        # Prepare representations
        # KeyBERT
        keybert_model = KeyBERTInspired()
        # MMR
        # mmr_model = MaximalMarginalRelevance(diversity=0.3)
        # GPT-3.5
        openai.api_key = OPENAI_API_KEY  # Loaded from a file that I wont upload to git

        prompt = """
        I have a topic that contains the following documents:
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make 
        sure it is in the following format: topic: <topic label> """
        openai_model = OpenAI(model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

        representation_model = {
            "KeyBERT": keybert_model,
            "OpenAI": openai_model,
            # "MMR": mmr_model, this is the default so no need to also show it
        }
        print('Prepared BERTopic Components')

        # topics, probs = topic_model.fit_transform(docs)
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            nr_topics=21,  # The first one is the outlier topic, so need 21
            # n_gram_range=(1, 2),
            verbose=True
        )
        print('About to fit')
        topic_model.fit(docs)
        # topics, probs = topic_model.fit_transform(docs)

        print('Fitted the topic model!')

        self.topic_model = topic_model

        # Write out the topic_model to a new file
        topic_f = self.filt_sents_f
        # Extract the directory, filename, and extension
        directory, full_filename = os.path.split(topic_f)
        filename_without_extension, extension = os.path.splitext(full_filename)
        # Replace "impt_sents_" with "topic_df_"
        new_filename_without_prefix = filename_without_extension.replace("impt_sents_", "raw_topic_df_")
        # Create the new filename with the .csv extension
        topic_df_filename = os.path.join(directory, new_filename_without_prefix + ".csv")

        topic_model.get_topic_info().to_csv(topic_df_filename, index=False)

    def extract_top_topics(self, n_top_topics):

        def extract_first_from_tuple_list(tuple_list):
            return [item[0] for item in tuple_list]

        topic_df = self.topic_model.get_topic_info().head(n_top_topics + 1)
        topic_df = topic_df.iloc[1:]
        topic_df = topic_df.set_index('Topic')
        topic_df['Words'] = topic_df.index.map(self.topic_model.get_topic)
        topic_df['Words'] = topic_df['Words'].apply(extract_first_from_tuple_list)

        self.top_topics = topic_df
        print(topic_df)

        # Write out the DataFrame to a new file
        topic_f = self.filt_sents_f
        # Extract the directory, filename, and extension
        directory, full_filename = os.path.split(topic_f)
        filename_without_extension, extension = os.path.splitext(full_filename)
        # Replace "impt_sents_" with "topic_df_"
        new_filename_without_prefix = filename_without_extension.replace("impt_sents_", "cleaned_topic_df_")
        # Create the new filename with the .csv extension
        topic_df_filename = os.path.join(directory, new_filename_without_prefix + ".csv")

        topic_df.to_csv(topic_df_filename)

        print(f'Printed topic_df to: {topic_df_filename}')

    def extract_sents_from_docs(self, f):
        assert len(self.sents) == 0
        i = 0

        file = open(f, "r")

        for line in tqdm(file):
            values = line.split("\t")
            assert len(values) == 2, f"Reading a file, we found a line with {len(values)} values: \n{line}\n"
            raw_label, raw_text = values[0], values[1]

            if raw_label == '10':
                label = 0
            elif raw_label == '01':
                label = 1
            else:
                raise ValueError("Invalid label text parsed")

            sents_from_doc = self.extract_sents_from_doc(raw_text, label)
            self.sents = self.sents + sents_from_doc
            i += 1

            #if i > 1:  # TODO REMOVE FOR FULL RUN
            #    break

        file.close()

        n_sents = len(self.sents)

        f_out = open(self.filt_sents_f + f'_{n_sents}.txt', 'x')
        for sent in self.sents:
            f_out.write(sent + "\n")
        f_out.close()

    def extract_sents_from_doc(self, doc_text, doc_label):

        # Interpret a document, finding a LIG importance score for all words in this context
        doc_viz_record = self.interpret_doc(doc_text, doc_label)

        # Separate the document into sentences, and score per our criteria
        doc_sentence_df = self.score_sentences(doc_viz_record)

        # Filter out the sentences that meet our criteria
        doc_filtered_sentences = self.filter_sentences(doc_sentence_df)

        # print(f'This document resulted in {len(doc_filtered_sentences)} new important sentences!')

        return doc_filtered_sentences

    def interpret_doc(self, doc_text, doc_label):
        text = [tok for tok in self.tokenizer(doc_text.lower())]
        if len(text) < self.gpu_vec_len_limit:
            text += ['<PAD>'] * (self.gpu_vec_len_limit - len(text))
        else:
            text = text[0:self.gpu_vec_len_limit]  # TODO remove for final run, this is to save time
        indexed = [self.vocab[t] for t in text]

        # Current gpu has issues with vram, so if doc is too big, move to cpu
        move_to_cpu = len(text) > self.gpu_vec_len_limit and not self.device == "cpu"
        move_to_cpu = False  # TODO remove for final run, this is to save time
        if move_to_cpu:
            old_device = self.device
            self.device = "cpu"
            self.model.to("cpu")
        else:
            old_device = None

        self.model.zero_grad()

        input_indices = torch.tensor(indexed, device=self.device)
        input_indices = input_indices.unsqueeze(0)

        seq_length = len(text)

        # predict
        pred = self.forward_with_sigmoid(input_indices).item()
        pred_ind = round(pred)

        # generate reference indices for each sample
        reference_indices = self.token_reference.generate_reference(seq_length, device=self.device).unsqueeze(0)

        # compute attributions and approximation delta using layer integrated gradients
        attributions_ig, delta = self.lig.attribute(input_indices, reference_indices,
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

        # If doc was too big for gpu, move back to gpu for next
        if move_to_cpu:
            self.device = old_device
            self.model.to(self.device)

        return viz_data_record

    @staticmethod
    def score_sentences(doc_viz_record):
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

    def forward_with_sigmoid(self, nn_input):
        return torch.sigmoid(self.model(nn_input))
