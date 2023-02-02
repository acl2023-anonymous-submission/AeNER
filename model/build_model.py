from model.aener import AeNER
from tools.config import Config, model_full_name

from transformers import RobertaModel, DebertaV2Model, BertModel


def build_model(config: Config):
    bert_model = None
    if config.encoder == "roberta":
        bert_model = RobertaModel.from_pretrained(config.roberta_model_path)
    else:
        model_name = model_full_name[config.encoder]
        if config.encoder in ["deberta-v2", "deberta-v3"]:
            bert_model = DebertaV2Model.from_pretrained(model_name)
        elif config.encoder == "bert":
            bert_model = BertModel.from_pretrained(model_name)
        else:
            raise Exception("Bad config: {}".format(config.encoder))
        bert_model.config.output_hidden_states = True

    return AeNER(config=config, bert=bert_model)
