"""Load an ATLOP checkpoint and predict document relations."""
import json

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model import DocREModel
from .prepro import read_docred, docred_rel2id
from .utils import collate_fn


class ATLOP:
    def __init__(self, module, checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run make train first.")
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(str(module / "checkpoints/bert-base-cased"))
        config = AutoConfig.from_pretrained(str(module / "checkpoints/bert-base-cased"), num_labels=97)
        config.cls_token_id = self.tokenizer.cls_token_id
        config.sep_token_id = self.tokenizer.sep_token_id
        config.transformer_type = "bert"
        self.model = DocREModel(config, AutoModel.from_config(config), num_labels=4)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.relation_ids = {value: key for key, value in docred_rel2id.items()}
        self.relation_names = json.loads((module / "data/relations.json").read_text())

    def __call__(self, document):
        features = read_docred([document], self.tokenizer, max_seq_length=1024)
        feature = features[0]
        if any(end + 1 > len(feature["input_ids"]) - 1
               for mentions in feature["entity_pos"] for start, end in mentions):
            raise ValueError("An entity falls beyond the model's token limit. Shorten the document.")
        batch = collate_fn(features)
        with torch.no_grad():
            predictions, = self.model(input_ids=batch[0], attention_mask=batch[1],
                                      entity_pos=batch[3], hts=batch[4])
        relations = []
        for (head, tail), row in zip(feature["hts"], predictions):
            for index in row.nonzero(as_tuple=False).flatten().tolist():
                if index == 0:
                    continue
                relation = self.relation_ids[index]
                relations.append({"head": document["vertexSet"][head][0]["name"],
                                  "relation": self.relation_names[relation],
                                  "relation_id": relation,
                                  "tail": document["vertexSet"][tail][0]["name"]})
        return relations
