from tools.config import Config
from typing import List, Tuple


class Reader:
    def __init__(
        self,
        tokenizer,
        config: Config,
        normalize_answer_function,
        skip_when_all_empty: List[str] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.flexibility_threshold = 1000
        self.normalize_answer_function = normalize_answer_function
        self.skip_when_all_empty = (
            skip_when_all_empty if skip_when_all_empty is not None else []
        )

    @staticmethod
    def find_new_i(offsets, i):
        while offsets[i + 1] == offsets[i]:
            i += 1
        i += 1
        return i

    @staticmethod
    def find_new_j(offsets, j):
        while offsets[j - 1] == offsets[j]:
            j -= 1
        j -= 1
        return j

    def find_valid_spans(
        self,
        text: List[str],
        offsets: List[str],
        answer_texts: List[str],
        text_segments: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:

        spans = []
        for answer_text in answer_texts:
            gold = self.normalize_answer_function(answer_text)
            buggy = ("(" in answer_text and ")" in answer_text) or (
                "€" in answer_text
                or "£" in answer_text
                or "/" in answer_text
                or ":" in answer_text
                or "–" in answer_text
                or "%" in answer_text
            )
            if buggy:
                gold_words = set(gold.split())

            next_segment = 0
            for i in range(len(offsets)):
                end_of_segments = False
                while text_segments[next_segment][1] <= i:
                    next_segment += 1
                    if next_segment == len(text_segments):
                        end_of_segments = True
                        break
                if end_of_segments:
                    break
                if text_segments[next_segment][0] > i:
                    continue
                if i != 0 and offsets[i] == offsets[i - 1]:
                    continue

                for j in range(i, text_segments[next_segment][1]):
                    if j + 1 < len(offsets) and offsets[j] == offsets[j + 1]:
                        continue

                    predicted = self.normalize_answer_function(
                        text[offsets[i][0] : offsets[j][1]]
                    )

                    if (
                        len(predicted) > len(gold)
                        or gold[: len(predicted)] != predicted
                    ) and (not buggy or len(predicted) > len(gold) + 10):
                        break

                    if buggy:
                        for word in predicted.split():
                            if word.isalpha() and word not in gold_words:
                                break

                    if predicted == gold:
                        i_new, j_new = i, j

                        while offsets[i_new] != offsets[j_new]:
                            i_new_new = self.find_new_i(offsets, i_new)
                            if (
                                self.normalize_answer_function(
                                    text[offsets[i_new_new][0] : offsets[j_new][1]]
                                )
                                == gold
                            ):
                                i_new = i_new_new
                            else:
                                break

                        while offsets[i_new] != offsets[j_new]:
                            j_new_new = self.find_new_j(offsets, j_new)
                            if (
                                self.normalize_answer_function(
                                    text[offsets[i_new][0] : offsets[j_new_new][1]]
                                )
                                == gold
                            ):
                                j_new = j_new_new
                            else:
                                break

                        if (i_new, j_new) not in spans:
                            spans.append((i_new, j_new))
                        break
        return spans
