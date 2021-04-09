from importlib import import_module
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from polish_ner.datasets.ner_dataset import NerVocab


class AutoTransformer(nn.Module):
    def __init__(
        self,
        architecture="allegro/herbert-base-cased",
        freeze=True,
        activation="torch.nn.LogSoftmax",
    ):
        super().__init__()

        self.architecture = architecture
        self.config = AutoConfig.from_pretrained(self.architecture, use_fast=True)

        self.config.output_hidden_states = False
        self.config.output_attentions = False
        self.transformer_model = AutoModel.from_pretrained(
            self.architecture, config=self.config
        )

        last_module = list(self.transformer_model.children())[-1]
        last_layer = list(last_module.children())[0]
        self.transformer_output_features = last_layer.out_features

        module_path, class_name = activation.rsplit(".", 1)
        module = import_module(module_path)
        self._activation = getattr(module, class_name)(dim=1)

        self._ner_vocab = NerVocab()
        self.classification_head = nn.Sequential(
            nn.Linear(
                in_features=self.transformer_output_features,
                out_features=len(self._ner_vocab),
            ),
            self._activation,
        )

        if freeze:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        trans = self.transformer_model(input_ids, attention_mask)
        head = self.classification_head(trans.last_hidden_state)

        return head
