import os, torch, json

from transformers import RobertaConfig, DebertaV2Config, BertConfig

model_full_name = {
    "roberta": None,
    "deberta-v2": "microsoft/deberta-v2-xxlarge",
    "deberta-v3": "microsoft/deberta-v3-large",
    "bert": "bert-base-uncased",
}


class Config:
    def __init__(
        self,
        name="default",
        wandb_project=None,
        wandb_entity="aener",
        data_path="data",
        device="cuda",
        dataset="drop",  # "drop", "tatqa", "race", "squad"
        wandb_mode=None,
        add_table=False,
        seps_in_table=True,
        embedding_lr_mult=3,
        table_y_embedding_len=31,
        table_x_embedding_len=11,
        predict_scale=False,
        answer_depended_scale_predictor=True,
        reasoning_option="gcn",  # None, 'empty' 'gcn', 'transfromer'
        reasoning_steps=3,
        reasoner_transformer_lr=5e-4,
        number_embeddins_range=(-6, 15),
        gcn_diff_opt=None,
        eval_batch_size=32,
        accumulated_train_batch_size=16,
        gradient_accumulation_steps=4,
        seed=42,
        min_epoch=0,
        max_epoch=15,
        early_stop=0,
        dropout=0.1,
        learning_rate=5e-4,
        bert_learning_rate=1.5e-5,
        weight_decay=5e-5,
        bert_weight_decay=0.01,
        log_per_updates=100,
        eps=1e-6,
        warmup=0.06,
        warmup_schedule="warmup_linear",
        grad_clipping=1.0,
        total_length_limit=None,
        question_length_limit=46,
        do_save_model=True,
        postnet_transformer_layers=0,
        postnet_transformer_learning_rate=1.5e-5,
        postnet_gru_layers=2,
        add_postnet_layernorm=True,
        add_postnet_table_embeddings=False,
        encoder="roberta",
        hidden_size=None,
        no_single_span=False,
        use_heavy_postnet=True,
        use_no_postnet=False,
        number_value_embeddings=None,  # None | float | "cnn" | "prototype_based" | "periodic" | "ranking" | "ensemble"
        trainable_number_value_embeddings_grid=False,
        number_value_embeddings_loss=None,  # None, logarithm, relative
        number_value_embeddings_weight=1.0,
        number_value_embeddings_loss_after_bert=False,
        full_logging=False,
        span_2d=False,
        forced_reasoning=True,
        sep_digits=False,
        counting_as_span=False,
    ) -> None:
        self.name = name
        self.wandb_project = wandb_project
        self.data_path = data_path

        self.device = device
        self.dataset = dataset
        self.wandb_mode = wandb_mode
        self.wandb_entity = wandb_entity
        self.add_table = add_table
        self.seps_in_table = seps_in_table
        self.embedding_lr_mult = embedding_lr_mult
        self.table_y_embedding_len = table_y_embedding_len
        self.table_x_embedding_len = table_x_embedding_len
        self.predict_scale = predict_scale
        self.answer_depended_scale_predictor = answer_depended_scale_predictor
        self.reasoning_option = reasoning_option
        self.reasoning_steps = reasoning_steps
        self.reasoner_transformer_lr = reasoner_transformer_lr
        self.number_embeddins_range = number_embeddins_range
        self.gcn_diff_opt = gcn_diff_opt
        self.eval_batch_size = eval_batch_size
        self.accumulated_train_batch_size = accumulated_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bert_learning_rate = bert_learning_rate
        self.weight_decay = weight_decay
        self.bert_weight_decay = bert_weight_decay
        self.log_per_updates = log_per_updates
        self.eps = eps
        self.warmup = warmup
        self.warmup_schedule = warmup_schedule
        self.grad_clipping = grad_clipping
        self.total_length_limit = total_length_limit
        self.question_length_limit = question_length_limit
        self.do_save_model = do_save_model
        self.postnet_transformer_layers = postnet_transformer_layers
        self.postnet_gru_layers = postnet_gru_layers
        self.postnet_transformer_learning_rate = postnet_transformer_learning_rate
        self.add_postnet_layernorm = add_postnet_layernorm
        self.add_postnet_table_embeddings = add_postnet_table_embeddings
        self.hidden_size = hidden_size
        self.no_single_span = no_single_span
        self.use_heavy_postnet = use_heavy_postnet
        self.use_no_postnet = use_no_postnet
        self.number_value_embeddings = number_value_embeddings
        self.trainable_number_value_embeddings_grid = (
            trainable_number_value_embeddings_grid
        )
        self.number_value_embeddings_loss = number_value_embeddings_loss
        self.number_value_embeddings_weight = number_value_embeddings_weight
        self.number_value_embeddings_loss_after_bert = (
            number_value_embeddings_loss_after_bert
        )
        self.full_logging = full_logging
        self.span_2d = span_2d
        self.forced_reasoning = forced_reasoning
        self.encoder = encoder
        self.sep_digits = sep_digits
        self.counting_as_span = counting_as_span

        # Saving config.

        model_folder = os.path.join(data_path, "models", name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        self.to_json(os.path.join(model_folder, "config.json"))

        if encoder not in ["roberta", "deberta-v2", "deberta-v3", "bert"]:
            raise Exception("Check encoder config: {}".format(encoder))

        # Processing parameters.
        self.train_batch_size = (
            accumulated_train_batch_size // gradient_accumulation_steps
        )
        self.use_reasoning = reasoning_option is not None
        if encoder == "roberta":
            self.sep_token_txt = "</s>"
        else:
            self.sep_token_txt = "[SEP]"

        # Processing directories.
        self.roberta_model_path = os.path.join(data_path, "roberta.large")
        self.dataset_path = os.path.join(data_path, self.dataset + "_dataset")
        self.train_dataset_path = os.path.join(
            self.dataset_path, self.dataset + "_dataset_train.json"
        )
        self.dev_dataset_path = os.path.join(
            self.dataset_path, self.dataset + "_dataset_dev.json"
        )
        self.test_dataset_path = os.path.join(
            self.dataset_path, self.dataset + "_dataset_test.json"
        )
        self.model_folder = model_folder
        self.model_path = os.path.join(self.model_folder, "checkpoint_best.pt")
        self.dump_path = os.path.join(self.model_folder, "prediction.json")
        self.sample_dev = os.path.join(self.model_folder, "dev_sample_prediction.json")
        self.sample_test = os.path.join(
            self.model_folder, "test_sample_prediction.json"
        )
        self.log_path = os.path.join(self.model_folder, "train.log")

        # Encoder parameters.
        bert_config = prepare_bert_config(self)
        self.bert_attention_probs_dropout_prob = (
            bert_config.attention_probs_dropout_prob
        )
        self.bert_hidden_dropout_prob = bert_config.hidden_dropout_prob
        self.bert_hidden_act = bert_config.hidden_act
        self.bert_hidden_size = bert_config.hidden_size
        self.bert_intermediate_size = bert_config.intermediate_size
        self.bert_layer_norm_eps = bert_config.layer_norm_eps
        self.bert_max_position_embeddings = bert_config.max_position_embeddings
        self.bert_num_attention_heads = bert_config.num_attention_heads

        # Correction of parameters.
        if wandb_project is None:
            self.wandb_project = self.dataset
        if hidden_size is None:
            print(
                "No hidden_size. Will use encoder hidden: {}".format(
                    self.bert_hidden_size
                )
            )
            self.hidden_size = self.bert_hidden_size
        if add_table == True and dataset in ["drop", "race", "squad"]:
            print(
                "add_table is True but DROP/SQUAD/RACE has no tables. Will use add_table=False"
            )
            self.add_table = False
        if predict_scale == True and dataset in ["drop", "race", "squad"]:
            print(
                "predict_scale is True but DROP/SQUAD/RACE has no scales. Will use predict_scale=False"
            )
            self.predict_scale = False
        if self.total_length_limit is None:
            print(
                "No total_length_limit size. Will use encoder max_position_embeddings: {}".format(
                    self.bert_max_position_embeddings
                )
            )
            self.total_length_limit = self.bert_max_position_embeddings
        if self.encoder == "roberta":
            print("Encoder is roberta. Will use forced_reasoning=False")
            self.forced_reasoning = False
        if self.total_length_limit > self.bert_max_position_embeddings:
            print(
                "total_length_limit={} is bigger then bert_max_position_embeddings={}!".format(
                    self.total_length_limit, self.bert_max_position_embeddings
                )
            )
        if use_no_postnet:
            print("Use no postnet:")
            print("\tuse_heavy_postnet {}->{}".format(self.use_heavy_postnet, False))
            self.use_heavy_postnet = False
            print(
                "\tadd_postnet_layernorm {}->{}".format(
                    self.add_postnet_layernorm, False
                )
            )
            self.add_postnet_layernorm = False
            print(
                "\tpostnet_transformer_layers {}->{}".format(
                    self.postnet_transformer_layers, 0
                )
            )
            self.postnet_transformer_layers = 0
            print("\tpostnet_gru_layers {}->{}".format(self.postnet_gru_layers, 0))
            self.postnet_gru_layers = 0
            print("\treasoning_option {}->{}".format(self.reasoning_option, None))
            self.reasoning_option = None
            self.use_reasoning = False

        if (
            self.trainable_number_value_embeddings_grid
            and number_value_embeddings is None
        ):
            print("No number_value_embeddings:")
            print(
                "\ttrainable_number_value_embeddings_grid {}->{}".format(
                    self.trainable_number_value_embeddings_grid, False
                )
            )
            self.trainable_number_value_embeddings_grid = False
            print(
                "\tnumber_value_embeddings_loss {}->{}".format(
                    self.add_number_value_embeddings_loss, None
                )
            )
            self.number_value_embeddings_loss = None

        self.device = torch.device(device)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self, f, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def load_from_file(path_to_load_from, data_path="data"):
        with open(path_to_load_from, "r") as f:
            config_dict = json.load(f)
        config_dict["data_path"] = data_path
        return Config(**config_dict)


def prepare_bert_config(config: Config):
    if config.encoder == "roberta":
        return RobertaConfig.from_pretrained(config.roberta_model_path)
    else:
        model_name = model_full_name[config.encoder]
        if config.encoder in ["deberta-v2", "deberta-v3"]:
            return DebertaV2Config.from_pretrained(model_name)
        elif config.encoder == "bert":
            return BertConfig.from_pretrained(model_name)
        else:
            raise Exception("Unknown model name: {}".format(config.encoder))
