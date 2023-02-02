import torch
from tqdm import tqdm
from .optimizer import BertAdam as Adam
from .utils import AverageMeter

from tools.config import Config
from model.aener import AeNER


class AeNerTrainer:
    def __init__(
        self, args: Config, network: AeNER, state_dict=None, num_train_updates=-1
    ):
        self.args = args
        self.train_loss = AverageMeter()
        self.train_num_emb_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        postnet_transformer_name = "postnet.transformer."
        reasoner_transformer_name = "reasoner.reasoner.transformer."

        optimizer_parameters = [
            # encoder
            {
                "params": [
                    p
                    for n, p in self.network.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.bert_weight_decay,
                "lr": args.bert_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.network.bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": args.bert_learning_rate,
            },
            # reasoner
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not (
                        n.startswith("bert.")
                        or n.startswith(postnet_transformer_name)
                        or n.startswith(reasoner_transformer_name)
                    )
                ],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
            # postnet_transformer
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and (n.startswith(postnet_transformer_name))
                ],
                "weight_decay": args.bert_weight_decay,
                "lr": args.postnet_transformer_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and (n.startswith(postnet_transformer_name))
                ],
                "weight_decay": 0.0,
                "lr": args.postnet_transformer_learning_rate,
            },
            # reasoner_transformer
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and (n.startswith(reasoner_transformer_name))
                ],
                "weight_decay": args.weight_decay,
                "lr": args.reasoner_transformer_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and (n.startswith(reasoner_transformer_name))
                ],
                "weight_decay": 0.0,
                "lr": args.reasoner_transformer_lr,
            },
        ]
        self.optimizer = Adam(
            optimizer_parameters,
            lr=args.learning_rate,
            warmup=args.warmup,
            t_total=num_train_updates,
            max_grad_norm=args.grad_clipping,
            schedule=args.warmup_schedule,
        )
        self.network.to(args.device)

    def avg_reset(self):
        self.train_loss.reset()
        self.train_num_emb_loss.reset()

    def update(self, tasks):
        self.network.train()
        output_dict = self.network(**tasks)
        loss = output_dict["loss"]
        self.train_loss.update(loss.item(), 1)
        num_emb_loss = output_dict["num_emb_loss"]
        self.train_num_emb_loss.update(num_emb_loss.item(), 1)
        loss = (
            loss + num_emb_loss * self.args.number_value_embeddings_weight
        ) / self.args.gradient_accumulation_steps
        loss.backward()
        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_data_list):
        dev_data_list.reset()
        self.network.eval()
        loss_sum = 0
        num_emb_loss_sum = 0
        total_batch = 0
        total_num = 0
        for batch in tqdm(dev_data_list):
            total_num += batch["input_ids"].size(0)
            output_dict = self.network(**batch)
            loss_sum += output_dict["loss"].item()
            num_emb_loss_sum += output_dict["num_emb_loss"].item()
            total_batch += 1
        details = self.network.eval_metrics.get_raw()
        metrics = self.network.get_eval_metrics(True)

        return (
            total_num,
            loss_sum / total_batch,
            num_emb_loss_sum / total_batch,
            metrics,
            details,
        )

    def save(self, prefix, epoch):
        network_state = dict(
            [(k, v.cpu()) for k, v in self.network.state_dict().items()]
        )
        other_params = {
            "optimizer": self.optimizer.state_dict(),
            "config": self.args,
            "epoch": epoch,
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print("model saved to {}".format(prefix))
