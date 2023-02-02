import torch
import random
import tqdm
import pickle

from dataset.token import Token
from dataset.utils import dataset_full_path
from tools.config import Config

from allennlp.data import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.fields.text_field import TextField
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data import Instance, Batch


class DataLoader(object):
    def __init__(self, args: Config, dataset, tokenizer, is_train: bool = True):
        self.args = args
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.is_train = is_train

        self.number_vocab = Vocabulary()
        self.number_vocab.add_tokens_to_namespace(
            [str(i) for i in range(10)] + [",", "."], namespace="character_vocab"
        )
        self.number_tokenizer = CharacterTokenizer()
        self.number_token_indexer = TokenCharactersIndexer(
            namespace="character_vocab", min_padding_length=5
        )

        if isinstance(dataset, str):
            dataset_str = dataset
            dpath = dataset_full_path(args, dataset)
            with open(dpath, "rb") as f:
                print("Load data from {}.".format(dpath))
                dataset = pickle.load(f)
            print("Successfully loaded.")

        all_data = []

        t = tqdm.tqdm(dataset, desc="Preparing data")
        for item in t:
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])

            question_passage_tokens = [
                Token(text=item[0], idx=item[1][0], edx=item[1][1])
                for item in zip(
                    item["question_passage_tokens"],
                    [(0, 0)]
                    + item["question_token_offsets"]
                    + [(0, 0)]
                    + item["passage_token_offsets"]
                    + [(0, 0)],
                )
            ]
            item["question_passage_tokens"] = question_passage_tokens
            if (
                not self.args.counting_as_span
                and item["counts"]
                and self.args.dataset == "tatqa"
            ):
                item["multi_span"] = []

            all_data.append((question_tokens, passage_tokens, item))

        print("Load data size {}.".format(len(all_data)))

        self.data = DataLoader.make_baches(
            all_data,
            args.train_batch_size if self.is_train else args.eval_batch_size,
            self.is_train,
        )
        self.offset = 0

    @staticmethod
    def make_baches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i : i + batch_size]
                if i + batch_size < len(data)
                else data[i:] + data[: i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)
            ]
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    def make_number_str_batch(self, all_number_strs):
        batch_tokens = self.number_tokenizer.batch_tokenize(all_number_strs)
        instances = []
        for tokens in batch_tokens:
            text_field = TextField(
                tokens, {"token_characters": self.number_token_indexer}
            )
            instance = Instance({"numbers": text_field})
            instances.append(instance)
        batch = Batch(instances)
        for instance in instances:
            instance.index_fields(self.number_vocab)
        batch_tensor = batch.as_tensor_dict(batch.get_padding_lengths())
        batch_tensor["numbers"]["token_characters"]["token_characters"] = batch_tensor[
            "numbers"
        ]["token_characters"]["token_characters"].to(self.args.device)
        return batch_tensor

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    @torch.no_grad()
    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            q_tokens, p_tokens, metas = zip(*batch)
            bsz = len(batch)
            max_seq_len = max([len(q) + len(p) for q, p in zip(q_tokens, p_tokens)]) + 3

            max_pnum_len = max(
                [1] + [len(item["passage_number_indices"]) for item in metas]
            )
            max_qnum_len = max(
                [1] + [len(item["question_number_indices"]) for item in metas]
            )

            max_pans_choice = max(
                [1] + [len(item["answer_passage_spans"]) for item in metas]
            )
            max_qans_choice = max(
                [1] + [len(item["answer_question_spans"]) for item in metas]
            )
            max_sign_add_sub_choice = max(
                [1] + [len(item["signs_for_add_sub_expressions"]) for item in metas]
            )
            max_sign_avg_choice = (
                max([1] + [len(item["signs_for_avg_expressions"]) for item in metas])
                if "signs_for_avg_expressions" in metas[0]
                else None
            )
            max_sign_change_ratio_choice = (
                max(
                    [1]
                    + [
                        len(item["signs_for_change_ratio_expressions"])
                        for item in metas
                    ]
                )
                if "signs_for_change_ratio_expressions" in metas[0]
                else None
            )
            max_sign_div_choice = (
                max([1] + [len(item["signs_for_div_expressions"]) for item in metas])
                if "signs_for_change_ratio_expressions" in metas[0]
                else None
            )

            input_ids = torch.LongTensor(bsz, max_seq_len).fill_(self.pad_idx)
            input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            table_position_ids = torch.LongTensor(bsz, 2, max_seq_len).fill_(0)
            token_type_ids = torch.LongTensor(bsz, max_seq_len).fill_(0)

            passage_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            question_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            span_mask = torch.LongTensor(bsz, max_seq_len, max_seq_len).fill_(0)

            passage_number_indices = torch.LongTensor(bsz, max_pnum_len).fill_(
                torch.iinfo(torch.long).min
            )
            question_number_indices = torch.LongTensor(bsz, max_qnum_len).fill_(
                torch.iinfo(torch.long).min
            )
            passage_number_order = torch.LongTensor(bsz, max_pnum_len).fill_(-1)
            question_number_order = torch.LongTensor(bsz, max_qnum_len).fill_(-1)
            passage_numbers = torch.FloatTensor(bsz, max_pnum_len).fill_(0.0)
            question_numbers = torch.FloatTensor(bsz, max_qnum_len).fill_(0.0)
            all_number_strs = [
                ["" for _ in range(max_qnum_len + max_pnum_len)] for _ in range(bsz)
            ]

            answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(
                -1
            )
            answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(
                -1
            )
            answer_as_add_sub_expressions = torch.LongTensor(
                bsz, max_sign_add_sub_choice, max_pnum_len
            ).fill_(0)
            answer_as_avg_expressions = (
                torch.LongTensor(bsz, max_sign_avg_choice, max_pnum_len).fill_(0)
                if (max_sign_avg_choice is not None)
                else None
            )
            answer_as_change_ratio_expressions = (
                torch.LongTensor(bsz, max_sign_change_ratio_choice, max_pnum_len).fill_(
                    0
                )
                if (max_sign_change_ratio_choice is not None)
                else None
            )
            answer_as_div_expressions = (
                torch.LongTensor(bsz, max_sign_div_choice, max_pnum_len).fill_(0)
                if (max_sign_div_choice is not None)
                else None
            )
            answer_as_counts = torch.LongTensor(bsz).fill_(-1)
            answer_scales = (
                torch.LongTensor(bsz).fill_(-1) if "scales" in metas[0] else None
            )

            # Multiple span label.
            max_text_answers = max(
                [1]
                + [
                    0
                    if len(metas[i]["multi_span"]) < 1
                    else len(metas[i]["multi_span"][1])
                    for i in range(bsz)
                ]
            )
            max_answer_spans = max(
                [1]
                + [
                    0
                    if len(metas[i]["multi_span"]) < 1
                    else max([len(item) for item in metas[i]["multi_span"][1]])
                    for i in range(bsz)
                ]
            )
            max_correct_sequences = max(
                [1]
                + [
                    0
                    if len(metas[i]["multi_span"]) < 1
                    else len(metas[i]["multi_span"][2])
                    for i in range(bsz)
                ]
            )
            is_counts_multi_span = torch.LongTensor(bsz).fill_(0)
            is_bio_mask = torch.LongTensor(bsz).fill_(0)
            bio_wordpiece_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
            answer_as_text_to_disjoint_bios = torch.LongTensor(
                bsz, max_text_answers, max_answer_spans, max_seq_len
            ).fill_(0)
            answer_as_list_of_bios = torch.LongTensor(
                bsz, max_correct_sequences, max_seq_len
            ).fill_(0)
            span_bio_labels = torch.LongTensor(bsz, max_seq_len).fill_(0)

            for i in range(bsz):
                q_len = len(q_tokens[i])
                p_len = len(p_tokens[i])

                # Inputs and their masks.
                input_ids[i, : 3 + q_len + p_len] = torch.LongTensor(
                    [self.cls_idx]
                    + q_tokens[i]
                    + [self.sep_idx]
                    + p_tokens[i]
                    + [self.sep_idx]
                )
                if self.args.encoder in ["deberta-v2", "deberta-v3", "bert"]:
                    token_type_ids[i, 2 + q_len : 3 + q_len + p_len] = torch.LongTensor(
                        p_len + 1
                    ).fill_(1)

                input_mask[i, : 3 + q_len + p_len] = 1
                question_mask[i, 1 : 1 + q_len] = 1
                passage_mask[i, 2 + q_len : 2 + q_len + p_len] = 1
                for start, end in metas[i]["text_segments"]:
                    span_mask[i, start:end, start:end] = (
                        torch.ones(end - start, end - start).type(torch.long).triu()
                    )
                for j, table_position in enumerate(metas[i]["table_position"]):
                    table_position_ids[i][0][j] = table_position[0]
                    table_position_ids[i][1][j] = table_position[1]

                passage_start = q_len + 2
                question_start = 1

                pn_len = len(metas[i]["passage_number_indices"])
                if pn_len > 0:
                    passage_number_indices[
                        i, :pn_len
                    ] = passage_start + torch.LongTensor(
                        metas[i]["passage_number_indices"][:pn_len]
                    )
                    # We need to add a fictitious number with the value 100 to answer percentage questions.
                    if metas[i]["passage_number_indices"][pn_len - 1] == -1:
                        passage_number_indices[i, pn_len - 1] = -1
                    passage_number_order[i, :pn_len] = torch.LongTensor(
                        metas[i]["passage_number_order"][:pn_len]
                    )
                    passage_numbers[i, :pn_len] = torch.tensor(
                        metas[i]["passage_numbers"][:pn_len]
                    )
                    for j in range(pn_len):
                        num_str = "{:,}".format(metas[i]["passage_numbers"][j])
                        all_number_strs[i][max_qnum_len + j] = num_str
                qn_len = len(metas[i]["question_number_indices"])
                if qn_len > 0:
                    question_number_indices[
                        i, :qn_len
                    ] = question_start + torch.LongTensor(
                        metas[i]["question_number_indices"][:qn_len]
                    )
                    question_number_order[i, :qn_len] = torch.LongTensor(
                        metas[i]["question_number_order"][:qn_len]
                    )
                    question_numbers[i, :qn_len] = torch.tensor(
                        metas[i]["question_numbers"][:qn_len]
                    )
                    for j in range(qn_len):
                        num_str = "{:,}".format(metas[i]["question_numbers"][j])
                        all_number_strs[i][j] = num_str

                pans_len = min(len(metas[i]["answer_passage_spans"]), max_pans_choice)
                for j in range(pans_len):
                    answer_as_passage_spans[i, j, 0] = (
                        metas[i]["answer_passage_spans"][j][0] + passage_start
                    )
                    answer_as_passage_spans[i, j, 1] = (
                        metas[i]["answer_passage_spans"][j][1] + passage_start
                    )

                qans_len = min(len(metas[i]["answer_question_spans"]), max_qans_choice)
                for j in range(qans_len):
                    answer_as_question_spans[i, j, 0] = (
                        metas[i]["answer_question_spans"][j][0] + question_start
                    )
                    answer_as_question_spans[i, j, 1] = (
                        metas[i]["answer_question_spans"][j][1] + question_start
                    )

                # Answer sign information.
                sign_add_sub_len = min(
                    len(metas[i]["signs_for_add_sub_expressions"]),
                    max_sign_add_sub_choice,
                )
                for j in range(sign_add_sub_len):
                    answer_as_add_sub_expressions[i, j, :pn_len] = torch.LongTensor(
                        metas[i]["signs_for_add_sub_expressions"][j][:pn_len]
                    )
                if answer_as_avg_expressions is not None:
                    sign_avg_len = min(
                        len(metas[i]["signs_for_avg_expressions"]), max_sign_avg_choice
                    )
                    for j in range(sign_avg_len):
                        answer_as_avg_expressions[i, j, :pn_len] = torch.LongTensor(
                            metas[i]["signs_for_avg_expressions"][j][:pn_len]
                        )
                if answer_as_change_ratio_expressions is not None:
                    sign_change_ratio_len = min(
                        len(metas[i]["signs_for_change_ratio_expressions"]),
                        max_sign_change_ratio_choice,
                    )
                    for j in range(sign_change_ratio_len):
                        answer_as_change_ratio_expressions[
                            i, j, :pn_len
                        ] = torch.LongTensor(
                            metas[i]["signs_for_change_ratio_expressions"][j][:pn_len]
                        )
                if answer_as_div_expressions is not None:
                    sign_div_len = min(
                        len(metas[i]["signs_for_div_expressions"]), max_sign_div_choice
                    )
                    for j in range(sign_div_len):
                        answer_as_div_expressions[i, j, :pn_len] = torch.LongTensor(
                            metas[i]["signs_for_div_expressions"][j][:pn_len]
                        )

                if len(metas[i]["counts"]) > 0:
                    answer_as_counts[i] = metas[i]["counts"][0]

                if "scales" in metas[i] and len(metas[i]["scales"]) > 0:
                    answer_scales[i] = metas[i]["scales"][0]

                # Add multi-span prediction.
                cur_seq_len = q_len + p_len + 3
                bio_wordpiece_mask[i, :cur_seq_len] = torch.LongTensor(
                    metas[i]["wordpiece_mask"][:cur_seq_len]
                )
                is_counts_multi_span[i] = metas[i]["is_counts_multi_span"]
                if len(metas[i]["multi_span"]) > 0:
                    is_bio_mask[i] = metas[i]["multi_span"][0]
                    span_bio_labels[i, :cur_seq_len] = torch.LongTensor(
                        metas[i]["multi_span"][-1][:cur_seq_len]
                    )
                    for j in range(len(metas[i]["multi_span"][1])):
                        for k in range(len(metas[i]["multi_span"][1][j])):
                            answer_as_text_to_disjoint_bios[
                                i, j, k, :cur_seq_len
                            ] = torch.LongTensor(
                                metas[i]["multi_span"][1][j][k][:cur_seq_len]
                            )
                    for j in range(len(metas[i]["multi_span"][2])):
                        answer_as_list_of_bios[i, j, :cur_seq_len] = torch.LongTensor(
                            metas[i]["multi_span"][2][j][:cur_seq_len]
                        )

            all_number_strs = self.make_number_str_batch(all_number_strs)

            out_batch = {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "token_type_ids": token_type_ids,
                "table_position_ids": table_position_ids,
                "passage_mask": passage_mask,
                "question_mask": question_mask,
                "span_mask": span_mask,
                "passage_number_indices": passage_number_indices,
                "passage_number_order": passage_number_order,
                "passage_numbers": passage_numbers,
                "question_number_order": question_number_order,
                "question_number_indices": question_number_indices,
                "question_numbers": question_numbers,
                "all_number_strs": all_number_strs,
                "answer_as_passage_spans": answer_as_passage_spans,
                "answer_as_question_spans": answer_as_question_spans,
                "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
                "answer_as_avg_expressions": answer_as_avg_expressions,
                "answer_as_change_ratio_expressions": answer_as_change_ratio_expressions,
                "answer_as_div_expressions": answer_as_div_expressions,
                "answer_as_counts": answer_as_counts.unsqueeze(1),
                "answer_scales": answer_scales.unsqueeze(1)
                if answer_scales is not None
                else None,
                "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
                "answer_as_list_of_bios": answer_as_list_of_bios,
                "span_bio_labels": span_bio_labels,
                "is_bio_mask": is_bio_mask,
                "bio_wordpiece_mask": bio_wordpiece_mask,
                "is_counts_multi_span": is_counts_multi_span,
                "metadata": metas,
            }
            for k in out_batch.keys():
                if isinstance(out_batch[k], torch.Tensor):
                    out_batch[k] = out_batch[k].to(self.args.device)

            yield out_batch
