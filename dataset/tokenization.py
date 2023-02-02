from dataset.utils import is_whitespace, get_number_from_word
from copy import deepcopy

from transformers import DebertaV2Tokenizer
from typing import Tuple

from tools.config import Config


class RobertaTokens:
    def __init__(
        self,
        config: Config,
        text: str,
        tokenizer,
        table_idx=(1, 1),
        segment_id: int = 0,
        column_id: int = 0,
        row_id: int = 0,
    ) -> None:
        self.text = deepcopy(text)
        self.tokens = []
        self.offset = []
        self.numbers = []
        self.number_indices = []
        self.wordpiece_mask = []
        self.table_position = []
        self.text_segments = []
        self.segment_ids = []
        self.column_ids = []
        self.row_ids = []

        word_to_char_offset = []
        next_token = True
        tokens = []
        tokens_white = []
        for i, c in enumerate(text):
            prev_c_white = True
            if i > 0:
                prev_c_white = is_whitespace(text[i - 1])
            next_c_num = False
            if i + 1 < len(text):
                next_c_num = text[i + 1].isnumeric()

            if is_whitespace(c):
                next_token = True
            elif (
                c
                in [
                    "+",
                    "~",
                    "$",
                    "€",
                    "%",
                    "(",
                    ")",
                    "?",
                    "[",
                    "]",
                    "*",
                    "/",
                    ":",
                    ";",
                    "-",
                    "–",
                    "!",
                    "{",
                    "}",
                    "”",
                    "“",
                    "‘",
                    "’",
                    "—",
                    "|",
                ]
                or (c in [",", "."] and not next_c_num)
            ):
                tokens.append(c)
                tokens_white.append(prev_c_white)
                word_to_char_offset.append(i)
                next_token = True
            else:
                if text[i : i + 3] == "'s ":
                    next_token = True
                if next_token:
                    tokens.append(c)
                    tokens_white.append(prev_c_white)
                    word_to_char_offset.append(i)
                else:
                    tokens[-1] += c
                next_token = False

        for i in range(len(tokens)):
            token_number = get_number_from_word(tokens[i])
            if token_number is not None:
                self.numbers.append(token_number)
                self.number_indices.append(len(self.tokens))

            allowed_for_number = [str(i) for i in range(10)] + [",", "."]
            if (
                token_number is not None
                and all([char in allowed_for_number for char in tokens[i]])
                and config.sep_digits
            ):
                local_tokens = list(tokens[i])
            else:
                local_tokens = [tokens[i]]

            sub_tokens = []
            for j in range(len(local_tokens)):
                if tokens_white[i] and j == 0:
                    local_sub_tokens = tokenizer._tokenize(" " + local_tokens[j])
                else:
                    if isinstance(tokenizer, DebertaV2Tokenizer):
                        local_sub_tokens = tokenizer._tokenize("Ġ" + local_tokens[j])[
                            2:
                        ]
                    else:
                        local_sub_tokens = tokenizer._tokenize(local_tokens[j])
                    if local_sub_tokens[0] == "▁":
                        local_sub_tokens = local_sub_tokens[1:]
                sub_tokens += local_sub_tokens

            if not sub_tokens:
                raise Exception("Bad tokens: {}".format(tokens))

            for sub_token in sub_tokens:
                self.tokens.append(sub_token)
                self.offset.append(
                    (word_to_char_offset[i], word_to_char_offset[i] + len(tokens[i]))
                )
                self.table_position.append(table_idx)

            self.wordpiece_mask += [1]
            if len(sub_tokens) > 1:
                self.wordpiece_mask += [0] * (len(sub_tokens) - 1)

        self.segment_ids = [segment_id] * len(self.tokens)
        self.column_ids = [column_id] * len(self.tokens)
        self.row_ids = [row_id] * len(self.tokens)
        if len(self.tokens):
            self.text_segments = [(0, len(self.tokens))]

        assert len(self.tokens) == len(self.offset)

    def add_sep(
        self,
        sep_token,
        table_idx=(1, 1),
        segment_id: int = 0,
        column_id: int = 0,
        row_id: int = 0,
    ):
        if len(self.text) != 0:
            self.text += " "
        self.text += "SEP"
        self.tokens += [sep_token]
        self.offset += [(len(self.text) - 3, len(self.text))]
        self.wordpiece_mask += [1]
        self.table_position.append(table_idx)
        self.segment_ids += [segment_id]
        self.column_ids += [column_id]
        self.row_ids += [row_id]

    def clip(self, max_len: int):
        self.tokens = self.tokens[:max_len]
        self.offset = self.offset[:max_len]
        self.wordpiece_mask = self.wordpiece_mask[:max_len]
        self.table_position = self.table_position[:max_len]
        self.segment_ids = self.segment_ids[:max_len]
        self.column_ids = self.column_ids[:max_len]
        self.row_ids = self.row_ids[:max_len]

        if len(self.offset) > 0:
            self.text = self.text[: self.offset[-1][1]]
        else:
            self.text = ""

        l = -1
        r = len(self.text_segments)
        while l + 1 < r:
            mid = (l + r) // 2
            if self.text_segments[mid][0] < len(self.tokens):
                l = mid
            else:
                r = mid
        self.text_segments = self.text_segments[:r]
        if len(self.text_segments):
            self.text_segments[-1] = (
                self.text_segments[-1][0],
                min(self.text_segments[-1][1], len(self.tokens)),
            )

        if len(self.number_indices) < 1 or self.number_indices[-1] < max_len:
            return

        lo = 0
        hi = len(self.number_indices) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.number_indices[mid] < max_len:
                lo = mid + 1
            else:
                hi = mid
        self.numbers = self.numbers[:lo]
        self.number_indices = self.number_indices[:lo]

    def add(self, to_add):
        if len(to_add.text) == 0:
            return
        if len(self.text) == 0:
            self.text = deepcopy(to_add.text)
            self.tokens = deepcopy(to_add.tokens)
            self.offset = deepcopy(to_add.offset)
            self.numbers = deepcopy(to_add.numbers)
            self.number_indices = deepcopy(to_add.number_indices)
            self.text_segments = deepcopy(to_add.text_segments)
            self.wordpiece_mask = deepcopy(to_add.wordpiece_mask)
            self.table_position = deepcopy(to_add.table_position)
            self.segment_ids = deepcopy(to_add.segment_ids)
            self.column_ids = deepcopy(to_add.column_ids)
            self.row_ids = deepcopy(to_add.row_ids)
            return

        add_to_token = len(self.tokens)
        add_to_offset = len(self.text) + 1

        self.text += " " + to_add.text
        self.tokens += to_add.tokens
        for i in range(len(to_add.offset)):
            to_add.offset[i] = (
                to_add.offset[i][0] + add_to_offset,
                to_add.offset[i][1] + add_to_offset,
            )
        self.offset += to_add.offset
        self.numbers += to_add.numbers
        for i in range(len(to_add.number_indices)):
            to_add.number_indices[i] += add_to_token
        self.number_indices += to_add.number_indices
        for i in range(len(to_add.text_segments)):
            to_add.text_segments[i] = (
                to_add.text_segments[i][0] + add_to_token,
                to_add.text_segments[i][1] + add_to_token,
            )
        self.text_segments += to_add.text_segments
        self.wordpiece_mask += to_add.wordpiece_mask
        self.table_position += to_add.table_position
        self.segment_ids += to_add.segment_ids
        self.column_ids += to_add.column_ids
        self.row_ids += to_add.row_ids


def roberta_extract_numbers(config: Config, text: str, tokenizer):
    return RobertaTokens(config=config, text=text, tokenizer=tokenizer).numbers


def get_number_from_text(text: str):
    texts = text.split()
    if len(texts) == 1:
        return get_number_from_word(texts[0])
    return None


def tokenize_table(
    config: Config, table, tokenizer, sep_token: str = None
) -> Tuple[RobertaTokens, bool]:
    output = RobertaTokens(config=config, text="", tokenizer=tokenizer)
    for i, line in enumerate(table):
        for j, cell_text in enumerate(line):
            cell = RobertaTokens(
                config=config,
                text=cell_text,
                tokenizer=tokenizer,
                table_idx=(i + 2, j + 2),
                segment_id=1,
                column_id=i + 1,
                row_id=j + 1,
            )
            if len(cell.tokens) == 0:
                continue
            if sep_token is not None:
                cell.add_sep(
                    sep_token=sep_token,
                    table_idx=(i + 2, j + 2),
                    segment_id=1,
                    column_id=i + 1,
                    row_id=j + 1,
                )
            output.add(cell)
    return output


def prepare_tokens(question_text, passage_texts, table, tokenizer, config: Config):
    # Preparing the question.
    question_length_limit = config.question_length_limit
    total_length_limit = config.total_length_limit
    question = RobertaTokens(config=config, text=question_text, tokenizer=tokenizer)
    if len(question.tokens) > question_length_limit:
        raise Exception(
            "Long question: " + str(len(question.tokens)) + " - " + str(question.tokens)
        )

    # Preparing the paragraph.
    passage_length_limit = total_length_limit - len(question.tokens) - 3
    passage = RobertaTokens(config=config, text="", tokenizer=tokenizer)
    for passage_text in passage_texts:
        passage.add(
            RobertaTokens(
                config=config, text=passage_text, tokenizer=tokenizer, segment_id=1
            )
        )

    # Preparing the table.
    if config.add_table:
        table = tokenize_table(
            config=config,
            table=table,
            tokenizer=tokenizer,
            sep_token=config.sep_token_txt if config.seps_in_table else None,
        )
        if not config.seps_in_table:
            table.add_sep(sep_token=config.sep_token_txt, segment_id=1)
        table.add(passage)
        passage = deepcopy(table)

    is_cut = False
    if len(passage.tokens) > passage_length_limit:
        is_cut = True
        passage.clip(passage_length_limit)

    # Used only for mutli-span.
    qp_tokens = ["<s>"] + question.tokens + ["</s>"] + passage.tokens + ["</s>"]
    qp_wordpiece_mask = (
        [1] + question.wordpiece_mask + [1] + passage.wordpiece_mask + [1]
    )
    qp_table_position = (
        [(1, 1)]
        + question.table_position
        + [(1, 1)]
        + passage.table_position
        + [(1, 1)]
    )

    question_text_segments = deepcopy(question.text_segments)
    for i in range(len(question_text_segments)):
        question_text_segments[i] = (
            question_text_segments[i][0] + 1,
            question_text_segments[i][1] + 1,
        )
    add_to_token = len(question.tokens) + 2
    passage_text_segments = deepcopy(passage.text_segments)
    for i in range(len(passage_text_segments)):
        passage_text_segments[i] = (
            passage_text_segments[i][0] + add_to_token,
            passage_text_segments[i][1] + add_to_token,
        )
    qp_text_segments = question_text_segments + passage_text_segments

    return (
        passage.text,
        passage.tokens,
        passage.offset,
        passage.numbers,
        passage.number_indices,
        passage.text_segments,
        question.text,
        question.tokens,
        question.offset,
        question.numbers,
        question.number_indices,
        question.text_segments,
        qp_tokens,
        qp_wordpiece_mask,
        qp_table_position,
        qp_text_segments,
        is_cut,
    )
