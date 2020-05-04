""" TextCNN model. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx = config.padding_idx
        )
        self.dropout = nn.Dropout(config.dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels = config.embedding_dim, 
                out_channels = config.num_filters, 
                kernel_size = fs)
            for fs in config.filter_sizes
        ])
        self.classifier = nn.Linear(
            len(config.filter_sizes) * config.num_filters, config.output_dim
        )
        self.init_weights(config)
    
    def init_weights(self, config):
        self.embedding.weight.data.copy_(config.pretrained_embeddings)
        self.embedding.weight.data[config.unk_idx] = torch.zeros(config.embedding_dim)
        self.embedding.weight.data[config.padding_idx] = torch.zeros(config.embedding_dim)

    def forward(self, sentences, labels=None):
        
        #sentences = [batch size, sent len]
        embedded = self.embedding(sentences)      
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, num_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, num_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, num_filters * len(filter_sizes)]
        logits = self.classifier(cat)
        
        return logits