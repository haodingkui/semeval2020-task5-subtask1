import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    BertPreTrainedModel, 
    RobertaModel, 
    RobertaForSequenceClassification)


class CustomRobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.high_dropout = nn.Dropout(config.high_hidden_dropout_prob)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        # self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output = outputs[0]
        hidden_layers = outputs[2]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers],
            dim=2
        )
        cls_output = (
            torch.softmax(self.layer_weights, dim=0) * cls_outputs
        ).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(cls_output))
            for _ in range(5)
        ], dim=0), dim=0)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_layers), (attentions)
