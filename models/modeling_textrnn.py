""" RNN model. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.embedding_dim,
            padding_idx = config.padding_idx
        )
        self.embedding_dropout = SpatialDropout(config.embedding_dropout)

        self.lstm1 = nn.LSTM(
            input_size=config.embedding_dim, 
            hidden_size=config.lstm_hidden_size, 
            bidirectional=True, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=config.lstm_hidden_size * 2, 
            hidden_size=config.lstm_hidden_size, 
            bidirectional=True, batch_first=True
        )

        self.linear1 = nn.Linear(config.lstm_hidden_size * 4, config.lstm_hidden_size * 4)
        self.linear2 = nn.Linear(config.lstm_hidden_size * 4, config.lstm_hidden_size * 4)

        self.linear_out = nn.Linear(
            config.lstm_hidden_size * 4, config.output_dim
        )
        
        self.init_weights(config)

    def init_weights(self, config):
        self.embedding.weight.data.copy_(config.pretrained_embeddings)
        self.embedding.weight.data[config.unk_idx] = torch.zeros(config.embedding_dim)
        self.embedding.weight.data[config.padding_idx] = torch.zeros(config.embedding_dim)
    
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), dim=1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        logits = self.linear_out(hidden)
        
        return logits
