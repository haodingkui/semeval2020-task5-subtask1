""" TextRNN model configuration """

class TextRNNConfig(object):

    def __init__(
        self,
        vocab_size=30000,
        pretrained_embedding=None,
        embedding_matrix=None,
        embedding_dim=300,
        embedding_dropout=0.3,
        lstm_hidden_size=128,
        output_dim=1,
        **kwargs
    ):
        self.pretrained_embedding = pretrained_embedding
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.lstm_hidden_size = lstm_hidden_size
        self.output_dim = output_dim
