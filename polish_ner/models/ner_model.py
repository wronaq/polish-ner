from polish_ner.models.base import Model
from transformers import AutoTokenizer
from polish_ner.datasets.ner_dataset import NerVocab


class NerModel(Model):
    def __init__(self, network, device, **kwargs):
        super().__init__(network, device, **kwargs)
        self.ner_vocab = NerVocab()

    def predict_on_text(self, input_text):
        tokenizer = AutoTokenizer.from_pretrained(
            self.network.architecture, use_fast=True
        )
        input_text = tokenizer(
            input_text,
            return_token_type_ids=False,
            is_split_into_words=False,
            truncation=True,
            return_tensors="pt",
        )
        self.network.eval()
        predictions = (
            self.network(input_text["input_ids"], input_text["attention_mask"])
            .squeeze()
            .argmax(dim=1)
        )

        return self._output_pairs(
            input_text["input_ids"].squeeze()[1:-1], predictions[1:-1], tokenizer
        )

    def _output_pairs(self, tokens, prediction, tokenizer):
        words = []
        tags = []
        for i, (token, pred) in enumerate(zip(tokens, prediction)):
            if i == 0:
                tag = self.ner_vocab.get_tag(str(pred.item()))
                word = tokenizer.decode(token)
            elif tag == self.ner_vocab.get_tag(str(pred.item())):
                word += tokenizer.decode(token)
            else:
                words.append(word)
                tags.append(tag)
                tag = self.ner_vocab.get_tag(str(pred.item()))
                word = tokenizer.decode(token)
        # last pair
        words.append(word)
        tags.append(tag)

        return [(w, t) for w, t in zip(words, tags)]
