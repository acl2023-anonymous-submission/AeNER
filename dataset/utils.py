from typing import List, Tuple
import re, string
from word2number.w2n import word_to_num
import numpy as np
import itertools
from tools.config import Config, model_full_name
from tools.utils import convert_number_to_str
import os
from transformers import RobertaTokenizer, DebertaV2Tokenizer, BertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


def prepare_tokenizer(config: Config) -> PreTrainedTokenizer:
    if config.encoder == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(config.roberta_model_path)
    else:
        model_name = model_full_name[config.encoder]
        if config.encoder in ["deberta-v2", "deberta-v3"]:
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        elif config.encoder in ["bert"]:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            raise Exception("Unknown config: {}".format(config.encoder))
    return tokenizer


def dataset_full_path(config: Config, mode):
    sep_str = "sep_" if config.seps_in_table else "nosep_"
    return os.path.join(
        config.dataset_path,
        "{}_{}{}{}{}{}_{}.pkl".format(
            config.encoder,
            ("tab_" + sep_str) if config.add_table else "",
            "mspan_" if config.no_single_span else "",
            "",
            "sepdigits_" if config.sep_digits else "",
            config.total_length_limit,
            mode,
        ),
    )


WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
ANSWER_SCALE = {"": 0, "percent": 1, "thousand": 2, "million": 3, "billion": 4}
PERCENT_SCALE = "percent"
ID_TO_ANSWER_SCALE = {i: answer_scale for answer_scale, i in ANSWER_SCALE.items()}


def get_number_from_word(word, improve_number_extraction=True):
    original = word

    lpunctuation = string.punctuation.replace("-", "").replace(".", "")
    rpunctuation = string.punctuation

    word = word.lstrip(lpunctuation).rstrip(rpunctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match("^\d*1st$", word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match("^\d*2nd$", word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match("^\d*3rd$", word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match("^\d+th$", word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == "0" and re.match("^\d+s$", word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match("^\d+(\.?\d+)?/km[²2]$", word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if "." in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match("^\d+(\.?\d+)?/month$", word):
                        # per month, e.g "1050.95/month"
                        if "." in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    except:
        print("Unexpected error in get_number_from_word in string:", original, sep="\n")
        return None
    return number


def is_whitespace(c):
    return ord(c) == 0x202F or c in [
        " ",
        " ",
        "\t",
        "\n",
        "\ufffd" "\u2009",
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u202f",
        "\u3000",
        "\ufeff",
    ]


def get_number_order(numbers):
    if len(numbers) < 1:
        return None
    ordered_idx_list = np.argsort(np.array(numbers)).tolist()

    rank = 0
    number_rank = []
    for i, idx in enumerate(ordered_idx_list):
        if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
            rank += 1
        number_rank.append(rank)

    ordered_idx_rank = zip(ordered_idx_list, number_rank)

    final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
    final_rank = [item[1] for item in final_rank]

    return final_rank


def create_bio_labels(spans: List[Tuple[int, int]], n_labels: int):
    # initialize all labels to O
    labels = [0] * n_labels

    for span in spans:
        start = span[0]
        end = span[1]
        # create B labels
        labels[start] = 1
        # create I labels
        labels[start + 1 : end + 1] = [2] * (end - start)

    return labels


def find_valid_add_sub_expressions(
    numbers: List,
    targets: List,
    config: Config,
    max_number_of_numbers_to_consider: int = 3,
    avg: bool = False,
) -> List[List[int]]:
    valid_signs_for_add_sub_expressions = []
    # TODO: Try smaller numbers.
    for number_of_numbers_to_consider in range(
        2, max_number_of_numbers_to_consider + 1
    ):
        possible_signs = list(
            itertools.product((-1, 1), repeat=number_of_numbers_to_consider)
        )
        for number_combination in itertools.combinations(
            enumerate(numbers), number_of_numbers_to_consider
        ):
            indices = [it[0] for it in number_combination]
            values = [it[1] for it in number_combination]
            for signs in possible_signs:
                eval_value = sum(sign * value for sign, value in zip(signs, values))
                if avg:
                    eval_value /= number_of_numbers_to_consider

                eval_value_str = convert_number_to_str(eval_value, config)
                if eval_value_str in targets:
                    labels_for_numbers = [0] * len(
                        numbers
                    )  # 0 represents ``not included''.
                    for index, sign in zip(indices, signs):
                        labels_for_numbers[index] = (
                            1 if sign == 1 else 2
                        )  # 1 for positive, 2 for negative
                    valid_signs_for_add_sub_expressions.append(labels_for_numbers)
    return valid_signs_for_add_sub_expressions


def find_valid_change_ratio_expressins(numbers: List, targets: List, config: Config):
    valid_abs_for_change_ratio_expressions = []
    for i_a, a in enumerate(numbers):
        if a == 0:
            continue
        for i_b, b in enumerate(numbers):
            if i_a == i_b:
                continue
            eval_value = (b - a) / a * 100
            eval_value_str = convert_number_to_str(eval_value, config)
            if eval_value_str in targets:
                labels_for_numbers = [0] * len(numbers)
                labels_for_numbers[i_a] = 1
                labels_for_numbers[i_b] = 2
                valid_abs_for_change_ratio_expressions.append(labels_for_numbers)
    return valid_abs_for_change_ratio_expressions


def find_valid_div_expressions(
    numbers: List,
    targets: List,
    config: Config,
    scale: str,
    max_number_of_numbers_to_consider: int = 1,
):
    valid_signs_for_div_expressions = []
    for upper_num in range(len(numbers)):
        if numbers[upper_num] == 0:
            continue
        for number_of_numbers_to_consider in range(
            1, max_number_of_numbers_to_consider + 1
        ):
            numbers_local = numbers[:upper_num] + numbers[upper_num + 1 :]
            possible_signs = list(
                itertools.product((-1, 1), repeat=number_of_numbers_to_consider)
            )
            for number_combination in itertools.combinations(
                enumerate(numbers_local), number_of_numbers_to_consider
            ):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    eval_value /= numbers[upper_num]
                    if scale == PERCENT_SCALE:
                        eval_value *= 100

                    eval_value_str = convert_number_to_str(eval_value, config)
                    if eval_value_str in targets:
                        labels_for_numbers = [0] * len(
                            numbers_local
                        )  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = (
                                1 if sign == 1 else 2
                            )  # 1 for positive, 2 for negative
                        labels_for_numbers = (
                            labels_for_numbers[:upper_num]
                            + [3]
                            + labels_for_numbers[upper_num:]
                        )
                        valid_signs_for_div_expressions.append(labels_for_numbers)
    return valid_signs_for_div_expressions


def restore_expressions_from_filtered(expressions, filter_correspondence, numbers_len):
    expressions_restored = []
    for expression in expressions:
        expressions_restored.append([])
        i_filter = 0
        for i in range(numbers_len):
            if (
                i_filter == len(filter_correspondence)
                or filter_correspondence[i_filter] > i
            ):
                expressions_restored[-1].append(0)
            else:
                expressions_restored[-1].append(expression[i_filter])
                i_filter += 1
    return expressions_restored
