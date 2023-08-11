from multiligtopic.args import get_args
from multiligtopic.multi_lig_topic import MultiLIGTopic

if __name__ == '__main__':
    args = get_args()

    print('Starting multiple document interpretation of neural model using Layered Integrated Gradients and BERT Topic')

    multi = MultiLIGTopic(args)

    # file = open('anon.txt', mode='r')  # TODO delete this
    # text = file.read()
    # file.close()

    # sentences = multi.extract_sents_from_doc(text, 1)

    multi.call_bertopic(multi.sents)


