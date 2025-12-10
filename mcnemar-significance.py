import argparse
import json
import logging
import pathlib
from typing import Dict, List, Tuple, Set

from datasets import Dataset
import torch
import torch.nn as nn
from tqdm import tqdm

import config
import utils
from probing_exp_routines import do_eval_epoch
from probing_exp_utils import LinearProbe, find_latest_checkpoint, load_probe_checkpoint_no_optimizers
from probing_new import load_and_preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_excluded_rows(normal_eval_ds: Dataset, ipa_eval_ds: Dataset) -> Set[int]:
    result = set()
    for row_idx in tqdm(range(len(normal_eval_ds)), desc='Finding excluded rows'):
        normal_row = normal_eval_ds[row_idx]
        ipa_row = ipa_eval_ds[row_idx]

        if normal_row['start_positions'] < 0 or normal_row['end_positions'] < 0 or ipa_row['start_positions'] < 0 or ipa_row['end_positions'] < 0:
            result.add(row_idx)

    return result


def compute_disagreement_matrix(
        layer: int, phoneme: int,
        a_preds: List[Dict[int, List[bool]]], b_preds: List[Dict[int, List[bool]]],
        labels: Dataset, excluded: Set[int],
) -> Tuple[int, int, int, int]:
    ignored = set()
    boc, ac, bc, nc = 0, 0, 0, 0
    a_layer_preds = a_preds[layer]
    b_layer_preds = b_preds[layer]
    for idx in range(len(labels)):
        if idx in excluded:
            logger.info(f'ignoring excluded row: {idx}')
            continue

        row = labels[idx]
        start_position = row['start_positions']
        end_position = row['end_positions']

        if start_position < 0 or end_position < 0:
            logger.warning(f'skipping row with no answer: {idx}')
            raise ValueError(f'row {idx} has no answer')

        label_list = row['features'][0]
        a_preds = a_layer_preds[idx]
        b_preds = b_layer_preds[idx]
        if label_list[phoneme]:
            if a_preds[phoneme]:
                if b_preds[phoneme]:
                    boc += 1
                else:
                    ac += 1
            elif b_preds[phoneme]:
                bc += 1
            else:
                nc += 1
        else:
            if a_preds[phoneme]:
                if b_preds[phoneme]:
                    nc += 1
                else:
                    bc += 1
            elif b_preds[phoneme]:
                ac += 1
            else:
                boc += 1
    return boc, ac, bc, nc


def determine_mcnemar_significance(
        eval_ds: Dataset, phoneme_count: int,
        normal_preds: List[Dict[int, List[bool]]], ipa_preds: List[Dict[int, List[bool]]],
        excluded: Set[int],
) -> List[List[bool]]:
    result = []
    for layer_idx in range(12):
        layer_result = []
        for phoneme_idx in range(phoneme_count):
            boc, ac, bc, nc = compute_disagreement_matrix(layer_idx, phoneme_idx, normal_preds, ipa_preds, eval_ds, excluded)
            chi_2 = ((ac - bc) * (ac - bc)) / (bc + ac)
            layer_result.append(chi_2 >= 3.84)
        result.append(layer_result)
    return result


def get_eval_dataset(cfg, mt: str, cpus: int) -> Dataset:
    logger.info('loading tokenizer')
    vocab, merges = utils.get_tokenizer_paths(cfg, mt)
    tokenizer = utils.load_tokenizer(vocab, merges)

    logger.info('loading datasets')
    return load_and_preprocess(cfg, 'validation', tokenizer, mt, cpus)


def preprocess_datasets(cfg, cpus: int) -> Tuple[Dataset, Dataset, Set[int]]:
    normal_eval_ds = get_eval_dataset(cfg, 'normal', cpus)
    ipa_eval_ds = get_eval_dataset(cfg, 'ipa', cpus)
    excluded = find_excluded_rows(normal_eval_ds, ipa_eval_ds)
    return normal_eval_ds, ipa_eval_ds, excluded


def get_predictions(cfg, mt: str, eval_ds: Dataset, phoneme_count: int) -> Tuple[float, List[dict], List[Dict[int, List[bool]]]]:
    logger.info(f'initializing 12 linear probes')
    probes = nn.ModuleList([
        LinearProbe(768, phoneme_count).to(DEVICE)
        for _ in range(12)
    ])

    hidden_states_dir = pathlib.Path(cfg['hidden_states'])
    logger.info(f'Using hidden states directory: {hidden_states_dir}')

    checkpoint_path = find_latest_checkpoint(checkpoint_dir, mt)
    if checkpoint_path is None:
        raise FileNotFoundError(f'No checkpoint found at {checkpoint_dir}')
    load_probe_checkpoint_no_optimizers(checkpoint_path, probes)

    return do_eval_epoch(
        probes,
        eval_ds, phoneme_count, mt, 'validation',
        hidden_states_dir, num_layers=12, average_span=args.average_span
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--average-span', action='store_true',
                    help='Average all tokens in the span instead of using last token')
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.default_config)

    logger.info(f'training probes on {DEVICE}')

    logger.info('loading phoneme mapping file')
    mapping_path = pathlib.Path(cfg['mappings'])
    with open(mapping_path, 'r') as fp:
        phoneme_mappings = json.load(fp)
    phoneme_count = len(phoneme_mappings['features'])

    checkpoint_dir = pathlib.Path(cfg['checkpoints']['probe_prefix'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    normal_ds, eval_ds, excluded_rows = preprocess_datasets(cfg, args.cpus)

    norm_loss, norm_met, norm_pred = get_predictions(cfg, 'normal', normal_ds, phoneme_count)
    ipa_loss, ipa_met, ipa_pred = get_predictions(cfg, 'ipa', eval_ds, phoneme_count)

    significance = determine_mcnemar_significance(eval_ds, phoneme_count, norm_pred, ipa_pred, excluded_rows)


