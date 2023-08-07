import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Visualization method for neural networks using BERTTOPIC to visual topics "
                                        "important across multiple documents based on what sentences had high "
                                        "importance according to layered integraded gradients ")

    parser.add_argument('--criteria', default='max_pos', choices=['max_pos', 'max_neg', 'mean_above', 'mean_below'],
                        help="Criteria to determine if a sentence is important")

    parser.add_argument('--cutoff', default=0.08, type=float, help='Value a sentence must have for given criteria')

    ckpt_name = "CNN_20230222-2048"
    ckpt_path = os.path.join(
        r"C:\Users\jjnunez\PycharmProjects\scar_nlp_psych\results\final_results\dspln_PSYCHIATRY_12\CNN",
        ckpt_name + ".pt")
    parser.add_argument('--model-path', default=ckpt_path)

    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])

    args = parser.parse_args()

    return args
