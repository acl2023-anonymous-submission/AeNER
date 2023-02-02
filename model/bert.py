import torch.nn as nn
from dataset.dataloader import DataLoader
from tools.config import Config


class Bert(nn.Module):
    def __init__(self, bert, config: Config) -> None:
        super(Bert, self).__init__()
        self.bert = bert
        self.config = config

        if config.bert_hidden_size != config.hidden_size:
            print(
                "changing hidden: {}->{}".format(
                    config.bert_hidden_size, config.hidden_size
                )
            )
            self.linear = nn.Linear(config.bert_hidden_size, config.hidden_size)

    def forward(self, input_ids, input_mask, token_type_ids):
        outputs = self.bert(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        if self.config.encoder in ["roberta", "bert"]:
            sequence_output_list = [item for item in outputs[2][-4:]]
        elif self.config.encoder in ["deberta-v2", "deberta-v3"]:
            sequence_output_list = [item for item in outputs[1][-4:]]
        else:
            raise Exception("Bad config: {}".format(self.config.encoder))

        if self.config.bert_hidden_size != self.config.hidden_size:
            sequence_output = self.linear(sequence_output)
            for i in range(len(sequence_output_list)):
                sequence_output_list[i] = self.linear(sequence_output_list[i])

        return sequence_output, sequence_output_list
