from core.model import IntentDetector, Encoder
from core.utils import utils
import os
import torch
from torch import jit, Tensor, nn
from typing import Dict, Union, Tuple, List
from pathlib import Path
import json
from shutil import make_archive
from datetime import datetime

Str2Int = Dict[str, int]


class ScriptableIntentClassifier(nn.Module):
    def __init__(self, word2idx: Str2Int, label2idx: Str2Int, encoder: Encoder):
        super().__init__()
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.encoder = encoder

    @classmethod
    def from_intent_detector(cls, idm: IntentDetector) -> "ScriptableIntentClassifier":
        return cls(idm.vocab, idm.label2index, idm.encoder)

    def forward(self, sentence_tensor: Tensor, topk: int = 1):
        """
        """
        label2prob = []

        output = self.encoder(sentence_tensor)
        output = torch.softmax(output, dim=1)

        if topk == -1:
            k = output.shape[1]
        else:
            k = topk

        topk_probs, topk_preds = torch.topk(output, k=k)
        topk_probs = topk_probs.squeeze(0)
        topk_preds = topk_preds.squeeze(0)
        return topk_probs, topk_preds

    def save_torch_script(self, path: Union[Path, str], language=None):
        if type(path) == str:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_content = path / "model_content"
        model_content.mkdir(parents=True, exist_ok=True)
        model_date = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
        model_file_name = ""
        if not language:
            model_file_name = model_date
        else:
            model_file_name = f"{model_date}_{language}"
        model_file = path / f"model_{model_file_name}"

        self.eval()
        jit_classifier = jit.script(self)
        jit_classifier.save(str(model_content / "model.pt"))
        with open(model_content / "vocab.json", "w") as file:
            json.dump(self.word2idx, file)
        with open(model_content / "label.json", "w") as file:
            json.dump(self.label2idx, file)

        output_file = make_archive(
            str(model_file), 'zip', root_dir=str(model_content))
        return str(model_file) + '.zip', jit_classifier
