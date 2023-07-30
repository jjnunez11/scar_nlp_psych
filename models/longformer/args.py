import models.args
import os

def get_args():
    parser = models.args.get_args()
    # Model arguments
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help="Maximum number of tokens to processes in, though Longformer can handle 4096")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--attention-window', type=int, default=512,
                        help="Size of attention window around each token.")

    # Directory to import in pretrained BERT model
    parser.add_argument('--pretrained_dir',
                        default=os.path.join(r"C:\Users\jjnunez\PycharmProjects", 'shared_data', 'models'))
    parser.add_argument('--pretrained-file', default='allenai-longformer-base-4096',
                        choices=['allenai-longformer-base-4096',
                                 'allenai/longformer-large-4096'])
    parser.add_argument('--hparams-file',
                        default=None,
                        help="Absolute path to a PyTorch lightning hparams.yaml file to load hparams from")

    args = parser.parse_args()
    return args
