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

    def extract_impt_sens_from_doc(self, doc_text, doc_label):
        doc_viz_record = self.interpret_doc(doc_text, doc_label)

        print(f'finished interpreting doc, here is the type of the record: {type(doc_viz_record)}')

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
        # Replace Label with Text below
        # self.add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, doc_label, delta, vis_data_records_ig)

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

    def forward_with_sigmoid(self, input):
        return torch.sigmoid(self.model(input))
