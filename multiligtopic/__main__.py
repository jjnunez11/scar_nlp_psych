from multiligtopic.args import get_args
from multiligtopic.multi_lig_topic import MultiLIGTopic

if __name__ == '__main__':
    args = get_args()

    print('Starting multiple document interpretation of neural model using Layered Integrated Gradients and BERT Topic')

    multi = MultiLIGTopic(args)

    multi.fit_topic_model(multi.sents)

    multi.extract_top_topics(args.n_top_topics)


