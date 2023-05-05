from transformers import BertTokenizer
from datasets.scar_transformer import SCARTransformer


class SCARBERT(SCARTransformer):
    def __init__(self, config, eval_only, undersample=False):
        super().__init__(config, BertTokenizer, eval_only, undersample)
