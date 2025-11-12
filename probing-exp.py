import argparse
import json
import logging
import os
import pathlib
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
import wandb

import config
import utils
from probing import load_and_preprocess, ProbedGPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def label_to_hot_vector_w_mask(features: List[int], phoneme_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    binary_vector = torch.zeros(phoneme_count, device=DEVICE)
    mask = torch.ones(phoneme_count, device=DEVICE)
    for f in features:
        if f > 0:
            binary_vector[f-1] = 1
        else:
            mask[f-1] = 0
    return binary_vector.float(), mask.float()


def do_eval_epoch(model: ProbedGPT, eval_ds: Dataset, phoneme_count: int) -> Tuple[float, List[dict]]:
    model.eval()
    dataset_order = list(range(len(eval_ds)))
    random.shuffle(dataset_order)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    layer_metrics = []
    epoch_loss = 0
    errors = 0
    for idx in tqdm(dataset_order, desc=f'eval'):
        row = eval_ds[idx]
        input_ids = torch.LongTensor(row['input_ids'], device=DEVICE)
        start_position = row['start_positions']
        end_position = row['end_positions']
        if start_position < 0 or end_position < 0:
            logger.warning(f'skipping row with no answer: {idx}')
            errors += 1
            continue
        labels = set(row['features'][0])
        label_vector, label_mask = label_to_hot_vector_w_mask(row['features'][0], phoneme_count)

        probe_outputs = model(input_ids, start_position, end_position)

        layer_losses = []
        for layer_probe in probe_outputs:
            pooled_probe = layer_probe.mean(dim=0)
            loss_vec = criterion(pooled_probe, label_vector)
            masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)
            layer_losses.append(masked_loss)
        loss = torch.stack(layer_losses).mean()
        epoch_loss += loss.item()

        if len(layer_metrics) == 0:
            for _ in range(len(probe_outputs)):
                layer_metrics.append({})
        for li, layer_probe in enumerate(probe_outputs):
            metric = layer_metrics[li]
            pooled_probe = layer_probe.mean(dim=0)
            pred = (pooled_probe > 0.5).bool()
            pred_indices = set(torch.nonzero(pred, as_tuple=False).squeeze(1).tolist())
            for fi in range(1, phoneme_count + 1):
                if fi not in metric:
                    metric[fi] = {
                        'matrix': {
                            'tp': 0,
                            'fp': 0,
                            'tn': 0,
                            'fn': 0,
                        },
                    }
                if -fi in labels:
                    metric[fi]['ignored'] = True
                    continue
                confusion_matrix = metric[fi]['matrix']
                if fi in labels:
                    if fi in pred_indices:
                        confusion_matrix['tp'] += 1
                    else:
                        confusion_matrix['fn'] += 1
                elif fi in pred_indices:
                    confusion_matrix['fp'] += 1
                else:
                    confusion_matrix['tn'] += 1

    for layer, layer_metric in enumerate(layer_metrics):
        for fi in range(1, phoneme_count + 1):
            if fi not in layer_metric:
                logger.warning(f'layer {layer} never saw feature {fi}')
                continue
            matrix = layer_metric[fi]['matrix']
            layer_metric[fi]['accuracy'] = (matrix['tp'] + matrix['tn']) / (matrix['tp'] + matrix['tn'] + matrix['fp'] + matrix['fn'])
            prec = matrix['tp'] / max((matrix['tp'] + matrix['fp']), 1)
            layer_metric[fi]['precision'] = prec
            rec = matrix['tp'] / max((matrix['tp'] + matrix['fn']), 1)
            layer_metric[fi]['recall'] = rec
            layer_metric[fi]['f1'] = 2 * prec * rec / max((prec + rec), 1)

    return epoch_loss / len(eval_ds) - errors, layer_metrics


def compute_macro_metrics(layer_metrics: List[dict], layer: int) -> Tuple[float, float, float, float]:
    macro_matrix = [0]*4  # confusion matrix
    metric = layer_metrics[layer]
    for feat in metric.keys():
        macro_matrix[0] += metric[feat]['matrix']['tp']
        macro_matrix[1] += metric[feat]['matrix']['fp']
        macro_matrix[2] += metric[feat]['matrix']['tn']
        macro_matrix[3] += metric[feat]['matrix']['fn']
    acc = (macro_matrix[0] + macro_matrix[2]) / max(sum(macro_matrix), 1)
    prec = macro_matrix[0] / max((macro_matrix[0] + macro_matrix[1]), 1)
    rec = macro_matrix[0] / max((macro_matrix[0] + macro_matrix[3]), 1)
    f1 = 2 * prec * rec / max((prec + rec), 1)
    return acc, prec, rec, f1


def compute_layer_feature_heatmap(layer_metrics: List[dict], phoneme_mapping: dict, metric: str = 'f1'):
    fig, ax = plt.subplots()
    ax.set_xlabel('Layer')
    ax.set_ylabel('Feature')
    ax.set_title(f'{metric.upper()} by layer x feature')

    # create heatmap
    phoneme_count = len(phoneme_mapping['features'])
    metric_mat = np.zeros((len(layer_metrics), phoneme_count))
    for li, lm in enumerate(layer_metrics):
        for feat in lm.keys():
            metric_mat[li, feat-1] = lm[feat][metric]

    im = ax.imshow(metric_mat, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label=metric.upper())

    feature_names = sorted(phoneme_mapping['features'].keys(), key=phoneme_mapping['features'])
    ax.set_xticks(list(range(phoneme_count)))
    ax.set_xticklabels(feature_names, rotation=90)

    return fig


def log_layer_feature_metrics(layer_metrics: List[dict], phoneme_mapping: dict, fname: pathlib.Path):
    feature_names = sorted(phoneme_mapping['features'].keys(), key=phoneme_mapping['features'])
    lines = [
        ['model_type', 'layer', 'feature', 'true_positive', 'false_positive', 'false_negative', 'true_negative', 'accuracy', 'precision', 'recall', 'f1']
    ]
    for li, metric in enumerate(layer_metrics):
        for fi in metric.keys():
            entry_line = [
                li, feature_names[fi-1],
                metric[fi]['matrix']['tp'],
                metric[fi]['matrix']['fp'],
                metric[fi]['matrix']['fn'],
                metric[fi]['matrix']['tn'],
                metric[fi]['accuracy'],
                metric[fi]['precision'],
                metric[fi]['recall'],
                metric[fi]['f1'],
            ]
            lines.append(list(map(str, entry_line)))
    s_lines = ['\t'.join(l) for l in lines]
    with open(fname, 'w+') as fp:
        fp.write('\n'.join(s_lines))


def do_train_run(cfg: dict, model_type: str, output_file: pathlib.Path, cpus: int = os.cpu_count()) -> nn.Module:
    wrun = wandb.init(
        entity='aaronjencks-personal',
        project='CSE5525-Final-Probing',
        name=model_type,
        config=cfg['hyperparameters']
    )

    logger.info('loading tokenizer')
    vocab, merges = utils.get_tokenizer_paths(cfg, model_type)
    tokenizer = utils.load_tokenizer(vocab, merges)
    logger.info('loading training dataset')
    train_ds = load_and_preprocess(cfg, 'train', tokenizer, model_type, cpus)
    logger.info('loading evaluation dataset')
    eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, model_type, cpus)
    logger.info('loading evaluation dataset')
    test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)
    logger.info('loading phoneme mapping file')
    mapping_path = pathlib.Path(cfg['mappings'])
    logger.info(f'loading mapping from {mapping_path}')
    with open(mapping_path, 'r') as fp:
        phoneme_mappings = json.load(fp)
    phoneme_count = len(phoneme_mappings['features'])
    logger.info('loading model')
    base_model = utils.load_pretrained_model(cfg, model_type, device=DEVICE)
    model = ProbedGPT(base_model, phoneme_count)
    logger.info('generating optimizer')
    hyperparameters = cfg['hyperparameters']
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # For multi-hot encoded vector
    optimizer = torch.optim.AdamW(model.probes.parameters(), lr=hyperparameters['learning_rate'])

    logger.info('starting training')
    dataset_order = list(range(len(train_ds)))
    for epoch in range(hyperparameters['epochs']):
        random.shuffle(dataset_order)
        epoch_loss = 0.0
        errors = 0
        for idx in tqdm(dataset_order, desc=f'train epoch {epoch}'):
            row = train_ds[idx]
            input_ids = torch.LongTensor(row['input_ids'], device=DEVICE)
            start_position = row['start_positions']
            end_position = row['end_positions']
            if start_position < 0 or end_position < 0:
                logger.warning(f'skipping row with no answer: {idx}')
                errors += 1
                continue
            labels = row['features'][0]
            label_vector, label_mask = label_to_hot_vector_w_mask(labels, phoneme_count)

            optimizer.zero_grad()
            probe_outputs = model(input_ids, start_position, end_position)
            layer_losses = []
            for layer_probe in probe_outputs:
                pooled_probe = layer_probe.mean(dim=0)
                loss_vec = criterion(pooled_probe, label_vector)
                masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)
                layer_losses.append(masked_loss)
            loss = torch.stack(layer_losses).mean()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_ds) - errors
        eval_loss, eval_metrics = do_eval_epoch(model, eval_ds, phoneme_count)

        log_entry = {
            'train/loss': epoch_loss,
            'eval/loss': eval_loss,
        }
        for layer in range(len(eval_metrics)):
            l_acc, l_prec, l_rec, l_f1 = compute_macro_metrics(eval_metrics, layer)
            log_entry[f'eval/accuracy/layer_{layer:02d}'] = l_acc
            log_entry[f'eval/precision/layer_{layer:02d}'] = l_prec
            log_entry[f'eval/recall/layer_{layer:02d}'] = l_rec
            log_entry[f'eval/f1/layer_{layer:02d}'] = l_f1
        eval_hm = compute_layer_feature_heatmap(eval_metrics, phoneme_mappings)
        log_entry['eval/f1_heatmap'] = wandb.Image(eval_hm)
        wandb.log(log_entry, step=epoch)
        plt.close(eval_hm)

    test_loss, test_metrics = do_eval_epoch(model, test_ds, phoneme_count)
    log_entry = {
        'test/loss': test_loss,
    }
    for layer in range(len(test_metrics)):
        l_acc, l_prec, l_rec, l_f1 = compute_macro_metrics(test_metrics, layer)
        log_entry[f'test/accuracy/layer_{layer:02d}'] = l_acc
        log_entry[f'test/precision/layer_{layer:02d}'] = l_prec
        log_entry[f'test/recall/layer_{layer:02d}'] = l_rec
        log_entry[f'test/f1/layer_{layer:02d}'] = l_f1
    test_hm = compute_layer_feature_heatmap(test_metrics, phoneme_mappings)
    wandb.log(log_entry)
    plt.close(test_hm)
    log_layer_feature_metrics(test_metrics, phoneme_mappings, output_file)

    wrun.finish()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], help='The model type')
    ap.add_argument('--eval-only', type=pathlib.Path, default=None, help='If supplied, specifies a checkpoint to evaluate, training is skipped, assumes that it is a trainer checkpoint')
    ap.add_argument('--output-log', type=str, default='probing_results', help='The file to store the final probing accuracies in')
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.default_config)

    logger.info(f'training model on {DEVICE}')

    for mt in args.model_type:
        output_path = pathlib.Path('data') / f'{args.output_log}_{mt}.tsv'
        model = do_train_run(cfg, mt, output_path, args.cpus)
