""" TextCNN model configuration """

class TextCNNConfig(object):

    def __init__(
        self,
        vocab_size=30000,
        embedding_dim=300,
        num_filters=200,
        filter_sizes=[1,2,3,4,5,6,7,8,9],
        output_dim=1,
        dropout=0.1,
        padding_idx=0,
        unk_idx=None,
        pretrained_embeddings=None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.pretrained_embeddings = pretrained_embeddings

