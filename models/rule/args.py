import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--max-tokens', type=int, default=5000, help='Number of features to use in our BoW Model')
    parser.add_argument('--use-idf', type=bool, default=True,
                        help='True, uses term frequency and inverse document freq to adjust vectors, False just tf')

    # Extra arguments for saving models etc, can implement later
    # TODO_ parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'reg_lstm'))
    # TODO_ parser.add_argument('--resume-snapshot', type=str)
    # TODO_ parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
