import pickle
from tools.config import Config
from dataset.readers.drop import DropReader
from dataset.readers.tatqa import TatQAReader
from dataset.utils import dataset_full_path, prepare_tokenizer
import json
import argparse


def main(config: Config):
    tokenizer = prepare_tokenizer(config=config)

    if config.dataset in ["drop", "race", "squad"]:
        Reader = DropReader
    elif config.dataset == "tatqa":
        Reader = TatQAReader
    else:
        raise "Unknown dataset"

    dev_reader = Reader(tokenizer, config=config)

    train_reader = Reader(
        tokenizer,
        config=config,
        skip_when_all_empty=[
            "passage_span",
            "question_span",
            "addition_subtraction",
            "average",
            "change_ratio",
            "division",
            "counting",
            "multi_span",
        ]
        if config.dataset == "tatqa"
        else [
            "passage_span",
            "question_span",
            "addition_subtraction",
            "counting",
            "multi_span",
        ]
        if config.dataset in ["drop", "race"]
        else ["passage_span"],
    )

    data_mode = ["dev", "train"]
    paths = {
        "dev": config.dev_dataset_path,
        "test": config.test_dataset_path,
        "train": config.train_dataset_path,
    }

    for dm in data_mode:
        data = (
            dev_reader._read(paths[dm])
            if (dm in ["dev", "test"])
            else train_reader._read(paths[dm])
        )
        path_to_save = dataset_full_path(config, dm)
        print("Save data to {}.".format(path_to_save))
        with open(path_to_save, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description="prepare the dataset")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        default=None,
        help="location of the configuration file",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config_json = json.load(fp)

    main(Config(**config_json))
