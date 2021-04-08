import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class NerDataset(Dataset):
    def __init__(
        self,
        path,
        batch_size=32,
        num_workers=4,
        architecture="allegro/herbert-base-cased",
        max_sentence_length=25,
    ):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.architecture = architecture
        self.max_sentence_length = max_sentence_length
        self._dataset = None

        self.load_or_generate_data()

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data/preprocessed"

    def load_or_generate_data(self):
        """Generate preprocessed data from a file"""
        try:
            self._dataset = _load_data(self.path)
        except:
            self._dataset = _generate_data(
                self.path, self.architecture, self.max_sentence_length
            )
            _save_data(
                self._dataset, self.path, self.architecture, self.max_sentence_length
            )

    def __getitem__(self, index):
        """Get item"""
        example = self._dataset[index]
        item = {}
        item["input_ids"] = torch.tensor(example[0], dtype=torch.long)
        item["attention_mask"] = torch.tensor(example[1], dtype=torch.long)
        item["target"] = torch.tensor(example[2], dtype=torch.long)

        return item

    def __len__(self):
        return len(self._dataset)


def _generate_data(path, architecture, max_sentence_length):

    logging.info("Generating")

    ner_vocab = NerVocab(path)
    path_to_data = NerDataset.data_dirname() / path
    tokenizer = AutoTokenizer.from_pretrained(architecture, use_fast=True)
    max_sentence_length = min(max_sentence_length, tokenizer.max_len_single_sentence)

    logging.info("Reading file")
    with open(path_to_data, "r") as f:
        examples = []
        example = []
        for line in tqdm(f.readlines()):
            line = line.rstrip("\n")
            if len(line) == 1:
                examples.append(example)
                example = []
                continue
            example.append(line)

    logging.info("Tokenizing")
    dataset = []
    for example in tqdm(examples):
        tokens_count = 0
        example_len = len(example)
        input_ids = [tokenizer.bos_token_id]
        attention_mask = [1]
        targets = [ner_vocab.get_id("O")]
        for current_el, el in enumerate(example, 1):
            try:
                word, tag = el.split(" ")
            except:
                logging.exception(f"Exception ocurred on: {example}")
            else:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                attentions = [1] * len(tokens)
                ner_tags = [ner_vocab.get_id(tag)] * len(tokens)
                tokens_count += len(tokens)
                if tokens_count <= max_sentence_length:
                    # add elements
                    input_ids.extend(tokens)
                    attention_mask.extend(attentions)
                    targets.extend(ner_tags)
                    if current_el < example_len:
                        continue
                    else:
                        # end sequence
                        input_ids.extend([tokenizer.sep_token_id])
                        attention_mask.extend([1])
                        targets.extend([ner_vocab.get_id("O")])
                        # pad
                        input_ids.extend(
                            [tokenizer.pad_token_id]
                            * (max_sentence_length + 2 - len(input_ids))
                        )
                        attention_mask.extend(
                            [0] * (max_sentence_length + 2 - len(attention_mask))
                        )
                        targets.extend(
                            [ner_vocab.get_id("O")]
                            * (max_sentence_length + 2 - len(targets))
                        )
                        # add example to dataset
                        dataset.append((input_ids, attention_mask, targets))
                        break
                else:
                    # end sequence
                    input_ids.extend([tokenizer.sep_token_id])
                    attention_mask.extend([1])
                    targets.extend([ner_vocab.get_id("O")])
                    # add example to dataset
                    dataset.append((input_ids, attention_mask, targets))
                    break

    return dataset


def _save_data(dataset, path, architecture, max_sentence_length):
    path_to_data = NerDataset.data_dirname() / path
    filename = f"{path_to_data.stem}_{architecture.replace('/', '-')}_{max_sentence_length}.pkl"
    path = path_to_data.parent / filename
    torch.save(dataset, path)
    logging.info("Saving")


def _load_data(path):
    logging.info("Trying to load")
    path_to_data = NerDataset.data_dirname() / path
    return torch.load(path_to_data)


class NerVocab:
    def __init__(self, path):
        self.path = path
        self.tag2id = {}
        self.id2tag = {}

        try:
            self._load_vocab()
        except:
            self._make_vocab()
            self._save_vocab()

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data/preprocessed"

    def _make_vocab(self):
        path_to_data = NerVocab.data_dirname() / self.path
        with open(path_to_data, "r") as f:
            ner_tags = set()
            for line in f.readlines():
                line = line.rstrip("\n")
                if len(line) == 1:
                    continue
                _, tag = line.split(" ")
                ner_tags.add(tag)

        sorted(ner_tags)
        for i, tag in enumerate(ner_tags):
            self.tag2id[tag] = i
            self.id2tag[i] = tag

    def _save_vocab(self):
        path_to_data = NerVocab.data_dirname() / "ner_vocab.json"
        vocab = {}
        vocab["tag2id"] = self.tag2id
        vocab["id2tag"] = self.id2tag
        with open(path_to_data, "w") as f:
            json.dump(vocab, f)

    def _load_vocab(self):
        path_to_data = NerVocab.data_dirname() / "ner_vocab.json"
        with open(path_to_data, "w") as f:
            vocab = json.load(f)
        self.tag2id = vocab["tag2id"]
        self.id2tag = vocab["id2tag"]

    def get_id(self, tag):
        return self.tag2id[tag]

    def get_tag(self, id):
        return self.id2tag[id]
