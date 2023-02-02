import sys
import random
import torch
import numpy
import logging
from time import gmtime, strftime
from tools.config import Config


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = (
            log_file
            if log_file is not None
            else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_number_to_str(number, config: Config, is_div: bool = False):
    if isinstance(number, int):
        return str(number)

    if config.dataset in ["drop", "race", "squad"]:
        # we leave at most 3 decimal places
        num_str = "%.3f" % number
    elif config.dataset == "tatqa":
        if not is_div:
            num_str = "%.2f" % number
        else:
            num_str = "%.4f" % number
    else:
        raise Exception("Unknown dataset: {}".format(config.dataset))

    while len(num_str) > 2 and num_str[-1] == "0":
        num_str = num_str[:-1]

    if num_str[-1] == ".":
        num_str = num_str[:-1]

    return num_str
