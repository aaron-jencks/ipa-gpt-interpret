import os
import pathlib
import random
from argparse import ArgumentParser
from typing import List, Tuple
import logging

import torch
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import GPT2TokenizerFast

from model import GPT, GPTConfig
from tokenizer import eod_token


logger = logging.getLogger(__name__)


def get_tokenizer_paths(cfg: dict, model_type: str) -> Tuple[pathlib.Path, pathlib.Path]:
    tokenizer_settings = cfg["tokenizers"]
    tokenizer_name = tokenizer_settings['name'][model_type]
    logger.info(f"Loading tokenizer '{tokenizer_name}' from '{tokenizer_settings['prefix']}'")
    return (
        pathlib.Path(tokenizer_settings["prefix"]) / f'{tokenizer_name}-vocab.json',
        pathlib.Path(tokenizer_settings["prefix"]) / f'{tokenizer_name}-merges.txt',
    )


def load_tokenizer(vocab: pathlib.Path, merges: pathlib.Path) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast(
        vocab_file=str(vocab),
        merges_file=str(merges),
        add_prefix_space=True,
    )

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return tokenizer


def get_checkpoint_path(cfg: dict, model_type: str) -> pathlib.Path:
    checkpoint_settings = cfg['checkpoints']
    return pathlib.Path(checkpoint_settings['prefix']) / f'{checkpoint_settings["name"][model_type]}.pt'


def load_pretrained_model(cfg: dict, model_type: str, device: str = 'cuda') -> GPT:
    path = get_checkpoint_path(cfg, model_type)
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    filtered = {k: v for k, v in state_dict.items()
                if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict({**model.state_dict(), **filtered})
    return model.to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.from_numpy(logits).argmax(dim=-1)
    labels = torch.from_numpy(labels)

    correct = (preds == labels).sum().item()
    total = len(labels)
    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0)
    }


def setup_default_args(ap: ArgumentParser) -> ArgumentParser:
    ap.add_argument('config', type=pathlib.Path, nargs='+', help='paths to config files')
    ap.add_argument('--default-config', type=pathlib.Path, default=pathlib.Path('config/default.json'),
                    help='path to the default config file')
    ap.add_argument('--cpus', type=int, default=os.cpu_count(), help='number of cpus')
    ap.add_argument('--debug', action='store_true', help='enable debug mode')
    return ap
