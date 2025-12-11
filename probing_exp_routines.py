import json
import logging
import os
import pathlib
import random
from typing import List, Tuple, Dict

from datasets import Dataset
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import wandb

from probing_exp_utils import label_to_hot_vector_w_mask, load_hidden_states, extract_token_representation, LinearProbe, \
    DEVICE, find_latest_checkpoint, load_checkpoint, save_checkpoint, compute_macro_metrics, \
    compute_layer_feature_heatmap, log_layer_feature_metrics
import utils
from probing_new import load_and_preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def do_eval_epoch(probes: nn.ModuleList, eval_ds: Dataset, phoneme_count: int,
                  model_type: str, split: str, hidden_states_dir: pathlib.Path,
                  num_layers: int, average_span: bool = False) -> Tuple[float, List[dict], List[Dict[int, List[bool]]]]:
    for probe in probes:
        probe.eval()

    dataset_order = list(range(len(eval_ds)))
    random.shuffle(dataset_order)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    layer_metrics = [{} for _ in range(num_layers)]
    layer_predictions = [{} for _ in range(num_layers)]
    epoch_loss = 0
    errors = 0

    with torch.no_grad():
        for idx in tqdm(dataset_order, desc=f'eval {split}'):
            row = eval_ds[idx]
            start_position = row['start_positions']
            end_position = row['end_positions']

            if start_position < 0 or end_position < 0:
                logger.warning(f'skipping row with no answer: {idx}')
                errors += 1
                continue

            labels = set(row['features'][0])
            label_vector, label_mask = label_to_hot_vector_w_mask(row['features'][0], phoneme_count)

            layer_losses = []

            hidden_states = load_hidden_states(idx, model_type, split, hidden_states_dir)
            if len(hidden_states[0]) == 0:
                logger.warning(f'skipping row with no span: {idx}')
                errors += 1
                continue

            for layer_idx in range(num_layers):
                try:
                    token_repr = extract_token_representation(hidden_states, layer_idx, average_span)

                    probe_output = probes[layer_idx](token_repr.unsqueeze(0))
                    pooled_probe = probe_output.squeeze(0)

                    loss_vec = criterion(pooled_probe, label_vector)
                    masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)
                    layer_losses.append(masked_loss)

                    metric = layer_metrics[layer_idx]
                    pred = (torch.sigmoid(pooled_probe) > 0.5).bool()
                    layer_predictions[layer_idx][idx] = pred.detach().cpu().numpy().tolist()

                    pred_nonzero = torch.nonzero(pred, as_tuple=False).squeeze()

                    if pred_nonzero.dim() == 0:
                        pred_indices = set([pred_nonzero.item()])
                    elif pred_nonzero.dim() == 1:
                        pred_indices = set(pred_nonzero.tolist())
                    else:
                        pred_indices = set()

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

                except FileNotFoundError as e:
                    logger.error(f"Could not load hidden states for idx={idx}, layer={layer_idx}: {e}")
                    errors += 1
                    continue

            if layer_losses:
                loss = torch.stack(layer_losses).mean()
                epoch_loss += loss.item()

    for layer, layer_metric in enumerate(layer_metrics):
        for fi in range(1, phoneme_count + 1):
            if fi not in layer_metric:
                logger.warning(f'layer {layer} never saw feature {fi}')
                continue
            matrix = layer_metric[fi]['matrix']
            total = sum(matrix.values())
            if total == 0:
                continue

            layer_metric[fi]['accuracy'] = (matrix['tp'] + matrix['tn']) / total

            denom = matrix['tp'] + matrix['fp']
            layer_metric[fi]['precision'] = matrix['tp'] / denom if denom > 0 else 0

            denom = matrix['tp'] + matrix['fn']
            layer_metric[fi]['recall'] = matrix['tp'] / denom if denom > 0 else 0

            precision = layer_metric[fi]['precision']
            recall = layer_metric[fi]['recall']
            denom = precision + recall
            layer_metric[fi]['f1'] = 2 * (precision * recall) / denom if denom > 0 else 0

    avg_loss = epoch_loss / max(len(eval_ds) - errors, 1)
    return avg_loss, layer_metrics, layer_predictions


def do_train_run(cfg: dict, model_type: str, output_file: pathlib.Path,
                 num_layers: int, hidden_dim: int, checkpoint_dir: pathlib.Path,
                 resume: bool, average_span: bool = False, cpus: int = os.cpu_count()) -> nn.ModuleList:
    logger.info(f'training probes on {DEVICE}')

    extraction_method = 'avgspan' if average_span else 'lasttoken'
    project_name = f'ipa-final-probing-linear-classifier-{model_type}-{extraction_method}'
    run_name = f'{model_type}_{extraction_method}_preextracted'

    wrun = None
    if cfg['wandb']['enabled']:
        wrun = wandb.init(
            entity=cfg['wandb']['entity'],
            project=project_name,
            name=run_name,
            config=cfg['hyperparameters'],
            resume='allow' if resume else False
        )

    logger.info('loading tokenizer')
    vocab, merges = utils.get_tokenizer_paths(cfg, model_type)
    tokenizer = utils.load_tokenizer(vocab, merges)

    logger.info('loading datasets')
    train_ds = load_and_preprocess(cfg, 'train', tokenizer, model_type, cpus)
    eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, model_type, cpus)
    # test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)

    logger.info('loading phoneme mapping file')
    mapping_path = pathlib.Path(cfg['mappings'])
    with open(mapping_path, 'r') as fp:
        phoneme_mappings = json.load(fp)
    phoneme_count = len(phoneme_mappings['features'])

    logger.info(f'initializing {num_layers} linear probes')
    probes = nn.ModuleList([
        LinearProbe(hidden_dim, phoneme_count).to(DEVICE)
        for _ in range(num_layers)
    ])

    hyperparameters = cfg['hyperparameters']
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # We need one for EACH probe
    optimizers = [
        torch.optim.AdamW(probes[i].parameters(), lr=hyperparameters['learning_rate'])
        for i in range(num_layers)
    ]

    hidden_states_dir = pathlib.Path(cfg['hidden_states'])
    logger.info(f'Using hidden states directory: {hidden_states_dir}')

    start_epoch = 0
    if resume:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, model_type)
        if checkpoint_path:
            try:
                start_epoch, _, _, _ = load_checkpoint(checkpoint_path, probes, optimizers)
                start_epoch += 1
                logger.info(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
                start_epoch = 0
        else:
            logger.info("No checkpoint found, starting training from scratch")
    else:
        logger.info("Starting training from scratch (resume=False)")

    logger.info('starting training')
    dataset_order = list(range(len(train_ds)))
    one_epoch = False

    for epoch in range(start_epoch, hyperparameters['epochs']):
        random.shuffle(dataset_order)
        epoch_loss = 0.0
        errors = 0
        one_epoch = True

        for probe in probes:
            probe.train()

        for idx in tqdm(dataset_order, desc=f'train epoch {epoch}'):
            row = train_ds[idx]
            start_position = row['start_positions']
            end_position = row['end_positions']

            if start_position < 0 or end_position < 0:
                logger.warning(f'skipping row with no answer: {idx}')
                errors += 1
                continue

            labels = row['features'][0]
            label_vector, label_mask = label_to_hot_vector_w_mask(labels, phoneme_count)

            layer_losses = []

            hidden_states = load_hidden_states(idx, model_type, 'train', hidden_states_dir)
            if len(hidden_states[0]) == 0:
                logger.warning(f'skipping row with no span: {idx}')
                errors += 1
                continue

            for layer_idx in range(num_layers):
                try:
                    optimizers[layer_idx].zero_grad()

                    token_repr = extract_token_representation(hidden_states, layer_idx, average_span)

                    probe_output = probes[layer_idx](token_repr.unsqueeze(0))
                    pooled_probe = probe_output.squeeze(0)

                    loss_vec = criterion(pooled_probe, label_vector)
                    masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)

                    masked_loss.backward()
                    optimizers[layer_idx].step()

                    layer_losses.append(masked_loss.detach())
                except FileNotFoundError as e:
                    logger.error(f"Could not load hidden states for idx={idx}, layer={layer_idx}: {e}")
                    errors += 1
                    break

            if not layer_losses:
                continue

            loss = torch.stack(layer_losses).mean()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_ds) - errors, 1)

        eval_loss, eval_metrics, _ = do_eval_epoch(
            probes, eval_ds, phoneme_count, model_type, 'validation',
            hidden_states_dir, num_layers, average_span
        )

        # Save checkpoint after each epoch
        save_checkpoint(checkpoint_dir, model_type, epoch, probes, optimizers,
                        epoch_loss, eval_loss, eval_metrics, num_layers)

        log_entry = {
            'train/loss': epoch_loss,
            'eval/loss': eval_loss,
        }
        for layer in range(num_layers):
            l_acc, l_prec, l_rec, l_f1 = compute_macro_metrics(eval_metrics, layer)
            log_entry[f'eval/accuracy/layer_{layer:02d}'] = l_acc
            log_entry[f'eval/precision/layer_{layer:02d}'] = l_prec
            log_entry[f'eval/recall/layer_{layer:02d}'] = l_rec
            log_entry[f'eval/f1/layer_{layer:02d}'] = l_f1

        eval_hm = compute_layer_feature_heatmap(
            eval_metrics, phoneme_mappings,
            pathlib.Path(cfg['heatmap_prefix']), run_name
        )
        if wrun is not None:
            log_entry['eval/f1_heatmap'] = wandb.Image(eval_hm)
            wandb.log(log_entry, step=epoch)
        plt.close(eval_hm)

    logger.info('Finished training')
    logger.info('Probe Weights:')
    for layer in range(num_layers):
        p = probes[layer]
        logger.info(f'layer {layer}: {p.linear.weight.detach().cpu().numpy()}')

    # test_loss, test_metrics = do_eval_epoch(
    #     probes, eval_ds, phoneme_count, model_type, 'validation',
    #     hidden_states_dir, num_layers, average_span
    # )
    #
    # log_entry = {'test/loss': test_loss}
    # for layer in range(num_layers):
    #     l_acc, l_prec, l_rec, l_f1 = compute_macro_metrics(test_metrics, layer)
    #     log_entry[f'test/accuracy/layer_{layer:02d}'] = l_acc
    #     log_entry[f'test/precision/layer_{layer:02d}'] = l_prec
    #     log_entry[f'test/recall/layer_{layer:02d}'] = l_rec
    #     log_entry[f'test/f1/layer_{layer:02d}'] = l_f1
    #
    # test_hm = compute_layer_feature_heatmap(test_metrics, phoneme_mappings)
    # wandb.log(log_entry)
    # plt.close(test_hm)

    if one_epoch:
        log_layer_feature_metrics(eval_metrics, phoneme_mappings, output_file)

    if wrun is not None:
        wrun.finish()
    return probes
