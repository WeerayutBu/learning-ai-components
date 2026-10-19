"""ATLOP training, evaluation, and checkpoint execution."""
import os
import math
from pathlib import Path

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from .model import DocREModel
from .utils import set_seed, collate_fn
from .prepro import read_docred
from .evaluation import to_official, official_evaluate


def train(args, model, train_features, dev_features, test_features):
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not train_features:
        raise ValueError("Training data is empty")
    Path("artifacts").mkdir(exist_ok=True)
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        train_iterator = range(int(num_epoch))
        total_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * int(num_epoch)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                window_start = (step // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                window_size = min(args.gradient_accumulation_steps, len(train_dataloader) - window_start)
                loss = outputs[0] / window_size
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                updated = (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader)
                if updated:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # Evaluate after the final update, including one-batch epochs.
                if (step + 1) == len(train_dataloader) or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and updated):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("artifacts/result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    ans = report(args, model, features)
    best_f1 = best_f1_ign = 0.0
    if ans:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, args.dev_file)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def report(args, model, features):

    if not features:
        return []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


class ATLOPTraining:
    """Own Torch objects, dataset access, and checkpoints."""

    def __init__(self, args):
        self.args = args

    def prepare(self):
        args = self.args

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        set_seed(args)

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_class,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        test_file = os.path.join(args.data_dir, args.test_file)
        train_features = read_docred(train_file, tokenizer, max_seq_length=args.max_seq_length)
        dev_features = read_docred(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        test_features = read_docred(test_file, tokenizer, max_seq_length=args.max_seq_length)

        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.transformer_type = args.transformer_type

        set_seed(args)
        model = DocREModel(config, model, num_labels=args.num_labels)
        model.to(args.device)

        self.model = model
        self.features = (train_features, dev_features, test_features)

    def fit(self):
        train(self.args, self.model, *self.features)

    def restore(self, checkpoint):
        self.model = amp.initialize(self.model, opt_level="O1", verbosity=0)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.args.device, weights_only=True))

    def evaluate(self):
        _, metrics = evaluate(self.args, self.model, self.features[1], tag="dev")
        return metrics

    def export_predictions(self):
        predictions = report(self.args, self.model, self.features[2])
        with open("artifacts/result.json", "w") as output:
            json.dump(predictions, output)
