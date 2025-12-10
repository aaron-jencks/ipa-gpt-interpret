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

        if -phoneme in label_list:
            boc += 1
            continue

        if phoneme in label_list:
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
    pbar = tqdm(total=phoneme_count * 12, desc='computing disagreement matrices')
    for layer_idx in range(12):
        layer_result = []
        for phoneme_idx in range(phoneme_count):
            boc, ac, bc, nc = compute_disagreement_matrix(layer_idx, phoneme_idx, normal_preds, ipa_preds, eval_ds, excluded)
            chi_2 = (((ac - bc) * (ac - bc)) / (bc + ac)) if (bc + ac) > 0 else 0
            layer_result.append(chi_2 >= 3.84)
            pbar.update(1)
        result.append(layer_result)
    return result


def mp_determine_mcnemar_significance(
        eval_ds: Dataset, phoneme_count: int,
        normal_preds: List[Dict[int, List[bool]]], ipa_preds: List[Dict[int, List[bool]]],
        excluded: Set[int],
        cpus: int,
) -> List[List[bool]]:
    def process_chunk(rows: Dict[str, list], idxs: List[int]) -> Dict[str, list]:
        result = {}
        for idx in range(len(idxs)):
            if idxs[idx] in excluded:
                logger.info(f'ignoring excluded row: {idxs[idx]}')
                for layer_idx in range(12):
                    for phoneme_idx in range(phoneme_count):
                        result_name = f'layer_{layer_idx}_phone_{phoneme_idx}'
                        if result_name not in result:
                            result[result_name] = []
                        result[result_name].append([1, 0, 0, 0])  # boc
                continue
            labels = rows['features'][idx][0]
            for layer_idx in range(12):
                normal_layer_preds = normal_preds[layer_idx]
                ipa_layer_preds = ipa_preds[layer_idx]
                for phoneme_idx in range(phoneme_count):
                    result_name = f'layer_{layer_idx}_phone_{phoneme_idx}'
                    if result_name not in result:
                        result[result_name] = []
                    row_matrix = [0]*4
                    if -phoneme_idx in labels:
                        row_matrix[0] = 1  # boc
                        result[result_name].append(row_matrix)
                        continue
                    row_index = 0
                    if phoneme_idx in labels:
                        if normal_layer_preds[phoneme_idx]:
                            if ipa_layer_preds[phoneme_idx]:
                                pass
                            else:
                                row_index = 1
                        elif ipa_layer_preds[phoneme_idx]:
                            row_index = 2
                        else:
                            row_index = 3
                    else:
                        if normal_layer_preds[phoneme_idx]:
                            if ipa_layer_preds[phoneme_idx]:
                                row_index = 3
                            else:
                                row_index = 2
                        elif ipa_layer_preds[phoneme_idx]:
                            row_index = 1
                    row_matrix[row_index] = 1
                    result[result_name].append(row_matrix)
        return result

    pds = eval_ds.map(process_chunk, batched=True, with_indices=True, num_proc=cpus)

    result_matrices = []
    for layer_idx in range(12):
        result_matrices.append([])
        for phoneme_idx in range(phoneme_count):
            result_matrices[-1].append([0]*4)

    for row in tqdm(pds, desc='condensing matrices'):
        for layer_idx in range(12):
            for phoneme_idx in range(phoneme_count):
                result_name = f'layer_{layer_idx}_phone_{phoneme_idx}'
                arr = row[result_name]
                result_matrices[layer_idx][phoneme_idx][0] += arr[0]
                result_matrices[layer_idx][phoneme_idx][1] += arr[1]
                result_matrices[layer_idx][phoneme_idx][2] += arr[2]
                result_matrices[layer_idx][phoneme_idx][3] += arr[3]

    result = []
    for layer_idx in range(12):
        result_matrices.append([])
        for phoneme_idx in range(phoneme_count):
            boc, ac, bc, nc = result_matrices[layer_idx][phoneme_idx]
            chi_2 = (((ac - bc) * (ac - bc)) / (bc + ac)) if (bc + ac) > 0 else 0
            result[-1].append(chi_2 >= 3.84)

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


def log_layer_feature_metrics(
        normal_layer_metrics: List[dict],
        ipa_layer_metrics: List[dict],
        significances: List[List[bool]],
        phoneme_mapping: dict, fname: pathlib.Path
):
    feature_names = sorted(phoneme_mapping['features'].keys(), key=lambda k: phoneme_mapping['features'][k])
    lines = [
        [
            'layer', 'feature',
            'normal_true_positive', 'normal_false_positive', 'normal_false_negative', 'normal_true_negative',
            'normal_accuracy', 'normal_precision', 'normal_recall', 'normal_f1',
            'ipa_true_positive', 'ipa_false_positive', 'ipa_false_negative', 'ipa_true_negative',
            'ipa_accuracy', 'ipa_precision', 'ipa_recall', 'ipa_f1',
            'is_significant']
    ]
    for li in range(12):
        normal_metric = normal_layer_metrics[li]
        ipa_metric = ipa_layer_metrics[li]
        for fi in range(phoneme_count):
            fi += 1
            if 'matrix' not in normal_metric[fi]:
                continue
            entry_line = [
                li, feature_names[fi - 1],
                normal_metric[fi]['matrix']['tp'],
                normal_metric[fi]['matrix']['fp'],
                normal_metric[fi]['matrix']['fn'],
                normal_metric[fi]['matrix']['tn'],
                normal_metric[fi].get('accuracy', 0),
                normal_metric[fi].get('precision', 0),
                normal_metric[fi].get('recall', 0),
                normal_metric[fi].get('f1', 0),
                ipa_metric[fi]['matrix']['tp'],
                ipa_metric[fi]['matrix']['fp'],
                ipa_metric[fi]['matrix']['fn'],
                ipa_metric[fi]['matrix']['tn'],
                ipa_metric[fi].get('accuracy', 0),
                ipa_metric[fi].get('precision', 0),
                ipa_metric[fi].get('recall', 0),
                ipa_metric[fi].get('f1', 0),
                significances[li][fi - 1],
            ]
            lines.append(list(map(str, entry_line)))

    s_lines = ['\t'.join(l) for l in lines]
    with open(fname, 'w+') as fp:
        fp.write('\n'.join(s_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--average-span', action='store_true',
                    help='Average all tokens in the span instead of using last token')
    ap.add_argument('--log-output-dir', type=pathlib.Path, default=pathlib.Path('data'))
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

    significance = mp_determine_mcnemar_significance(eval_ds, phoneme_count, norm_pred, ipa_pred, excluded_rows, args.cpus)

    log_layer_feature_metrics(
        norm_met, ipa_met, significance,
        phoneme_mappings, args.log_output_dir / 'significance_test.tsv'
    )



