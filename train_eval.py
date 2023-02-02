import os
import wandb
import json
import torch
from datetime import datetime

from dataset.dataloader import DataLoader
from dataset.utils import prepare_tokenizer
from tools.config import Config
from tools.trainer import AeNerTrainer
from tools.utils import create_logger, set_seed
from model.build_model import build_model
import argparse


def main(args: Config, is_train: bool = True):
    logger = create_logger("Bert Drop Pretraining", log_file=args.log_path)

    config_folder = os.path.join(args.model_folder, "config.json")
    with open(config_folder, "r") as f:
        config_dict = json.load(f)
    logger.info(config_dict)

    if is_train:
        wandb.init(
            project=args.wandb_project,
            dir=os.path.join(args.model_folder),
            name=args.name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=config_dict,
        )

    tokenizer = prepare_tokenizer(args)
    set_seed(args.seed)
    logger.info("Build AeNER...")
    model = build_model(config=args)
    if not is_train:
        weights_folder = os.path.join(args.model_folder, "checkpoint_best.pt")
        model.load_state_dict(torch.load(weights_folder))
        model.eval()

    logger.info("Loading data...")
    dev_itr = DataLoader(args, dataset="dev", tokenizer=tokenizer)
    if is_train:
        best_result = float("-inf")
        train_itr = DataLoader(args, dataset="train", tokenizer=tokenizer)
        num_train_updates = int(
            args.max_epoch * len(train_itr) / args.gradient_accumulation_steps
        )
        logger.info("Num update updates {}!".format(num_train_updates))
    else:
        try:
            test_itr = DataLoader(args, dataset="test", tokenizer=tokenizer)
        except:
            print("No test data!")
        num_train_updates = -1

    logger.info("Build optimizer etc...")
    trainer = AeNerTrainer(args, model, num_train_updates=num_train_updates)

    if is_train:
        train_start = datetime.now()
        first = True

        epoch = 0
        early_stop = 0
        while (
            epoch < args.min_epoch
            or (
                args.early_stop != 0
                and early_stop < args.early_stop
                or args.early_stop == 0
            )
            and (args.max_epoch != 0 and epoch < args.max_epoch or args.max_epoch == 0)
        ):
            epoch += 1
            early_stop += 1

            trainer.avg_reset()
            if not first:
                train_itr.reset()
            first = False
            logger.info("At epoch {}".format(epoch))
            for batch in train_itr:
                trainer.update(batch)
                if (
                    trainer.step
                    % (args.log_per_updates * args.gradient_accumulation_steps)
                    == 0
                    or trainer.step == 1
                ):
                    metrics = trainer.network.get_train_metrics(reset=True)
                    logger.info(
                        "Updates[{0:6}] train loss[{1:.5f}] num_emb loss [{2:.5f}] em[{3:.5f}] f1[{4:.5f}] scale[{5:.5f}] remaining[{6}]".format(
                            trainer.updates,
                            trainer.train_loss.avg,
                            trainer.train_num_emb_loss.avg,
                            metrics["em"],
                            metrics["f1"],
                            metrics["scale"] if "scale" in metrics else -1,
                            str(
                                (datetime.now() - train_start)
                                / trainer.step
                                * (
                                    num_train_updates * args.gradient_accumulation_steps
                                    - trainer.step
                                )
                            ).split(".")[0],
                        )
                    )
                    for metric in metrics:
                        wandb.log(
                            {"train/loss": trainer.train_loss.avg}, step=trainer.updates
                        )
                        wandb.log(
                            {"train/num_emb_loss": trainer.train_num_emb_loss.avg},
                            step=trainer.updates,
                        )
                        if metrics[metric] is not None:
                            wandb.log(
                                {"train/" + metric: metrics[metric]},
                                step=trainer.updates,
                            )
                    if args.gcn_diff_opt == "tanh":
                        wandb.log(
                            {
                                "gcn_tanh_T": trainer.network.gcn.gcn.diff_process.T.item()
                            },
                            step=trainer.updates,
                        )
                        logger.info(
                            "gcn_tanh_T: {}".format(
                                trainer.network.gcn.gcn.diff_process.T.item()
                            )
                        )
                    if args.trainable_number_value_embeddings_grid:
                        wandb.log(
                            {
                                "number_value_embedding_power": trainer.network.reasoner.number_value_embeddings_power.item()
                            },
                            step=trainer.updates,
                        )
                        logger.info(
                            "number_value_embedding_power: {}".format(
                                trainer.network.reasoner.number_value_embeddings_power.item()
                            )
                        )
                    trainer.avg_reset()
            (
                total_num,
                eval_loss,
                eval_num_emb_loss,
                metrics,
                details,
            ) = trainer.evaluate(dev_itr)
            logger.info(
                "Eval {} examples, result in epoch {} eval loss {} num_emb loss {} em {} f1 {} scale {}.".format(
                    total_num,
                    epoch,
                    eval_loss,
                    eval_num_emb_loss,
                    metrics["em"],
                    metrics["f1"],
                    metrics["scale"] if "scale" in metrics else -1,
                )
            )
            wandb.log({"eval/num_emb_loss": eval_num_emb_loss}, step=trainer.updates)
            for metric in metrics:
                if metrics[metric] is not None:
                    wandb.log({"eval/" + metric: metrics[metric]}, step=trainer.updates)

            if metrics["f1"] > best_result:
                with open(args.dump_path, "w") as f:
                    json.dump(details, f, indent=4)
                if args.do_save_model:
                    save_prefix = os.path.join(args.model_folder, "checkpoint_best")
                    trainer.save(save_prefix, epoch)
                best_result = metrics["f1"]
                logger.info("Best eval F1 {} at epoch {}".format(best_result, epoch))
                early_stop = 0

        logger.info(
            "done training in {} seconds!".format(
                (datetime.now() - train_start).seconds
            )
        )
        wandb.finish()

    if not is_train:
        total_num, eval_loss, num_emb_loss, metrics, details = trainer.evaluate(dev_itr)
        print("Number embeddings loss:", num_emb_loss)
        print("Eval metrics:", metrics, sep="\n")
        sample_prediction = dict()
        for detail in details:
            if args.dataset == "tatqa":
                sample_prediction[detail["question_id"]] = [
                    detail["predicted_answer"],
                    detail["predicted_scale"],
                ]
            else:
                answer = detail["predicted_answer"]
                if not isinstance(answer, list):
                    answer = [answer]
                sample_prediction[detail["question_id"]] = answer
        with open(args.sample_dev, "w") as f:
            json.dump(sample_prediction, f, indent=4)

        total_num, eval_loss, num_emb_loss, metrics, details = trainer.evaluate(
            test_itr
        )
        sample_prediction = dict()
        with open(args.dump_path, "w") as f:
            json.dump(details, f, indent=4)
        for detail in details:
            if args.dataset == "tatqa":
                sample_prediction[detail["question_id"]] = [
                    detail["predicted_answer"],
                    detail["predicted_scale"],
                ]
            else:
                answer = detail["predicted_answer"]
                if not isinstance(answer, list):
                    answer = [answer]
                sample_prediction[detail["question_id"]] = answer
        with open(args.sample_test, "w") as f:
            json.dump(sample_prediction, f, indent=4)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description="train/eval")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        default=None,
        help="location of the configuration file",
    )
    parser.add_argument(
        "--is_eval",
        type=bool,
        required=False,
        default=False,
        help="if True, then evaluate, else train",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config_json = json.load(fp)

    main(Config(**config_json), not args.is_eval)
