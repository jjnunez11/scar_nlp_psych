from transformers import LongformerTokenizer
from datasets.scar_transformer import SCARTransformer


class SCARLongformer(SCARTransformer):
    def __init__(self, config, eval_only, undersample=False):
        super().__init__(config, LongformerTokenizer, eval_only, undersample)