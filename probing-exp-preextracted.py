import argparse
import json
import logging
import os
import pathlib
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
import wandb

import config
import utils
from probing_new import load_and_preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# python /users/PAS3184/chiang248/cse5525/ipa-gpt-interpret/probing-exp-preextracted.py config/default.json --model-type ipa normal --num-layers 12 --hidden-dim 768 --output-log probing_results_preextracted

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


def load_hidden_states(dataset_idx: int, layer_idx: int, model_type: str, split: str, 
                       hidden_states_dir: pathlib.Path) -> np.ndarray:
    model_dir = hidden_states_dir / f"token_hidden_states_{model_type}_{split}"
    file_path = model_dir / f"{dataset_idx}_{layer_idx}_{model_type}.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Hidden state file not found: {file_path}")
    
    # Load npz file
    with np.load(file_path) as data:
        hidden_states = data['arr_0']
    
    return hidden_states


def load_metadata(model_type: str, split: str, hidden_states_dir: pathlib.Path) -> Dict:
    model_dir = hidden_states_dir / f"token_hidden_states_{model_type}_{split}"
    metadata_path = model_dir / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def label_to_hot_vector_w_mask(features: List[int], phoneme_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    binary_vector = torch.zeros(phoneme_count, device=DEVICE)
    mask = torch.ones(phoneme_count, device=DEVICE)
    for f in features:
        if f > 0:
            binary_vector[f-1] = 1
        else:
            mask[abs(f)-1] = 0
    return binary_vector.float(), mask.float()


def extract_token_representation(hidden_states: np.ndarray, 
                                 start_position: int, 
                                 end_position: int,
                                 use_last_token: bool = True) -> torch.Tensor:
    hidden_states_tensor = torch.from_numpy(hidden_states).float().to(DEVICE)
    
    if use_last_token:
        token_repr = hidden_states_tensor[-1, :]
    else:
        token_repr = hidden_states_tensor[end_position, :]
    
    return token_repr


def compute_macro_metrics(layer_metrics: List[dict], layer_idx: int) -> Tuple[float, float, float, float]:
    layer_metric = layer_metrics[layer_idx]
    accuracies, precisions, recalls, f1s = [], [], [], []
    
    for fi, metric in layer_metric.items():
        if 'accuracy' in metric:
            accuracies.append(metric['accuracy'])
        if 'precision' in metric:
            precisions.append(metric['precision'])
        if 'recall' in metric:
            recalls.append(metric['recall'])
        if 'f1' in metric:
            f1s.append(metric['f1'])
    
    return (
        np.mean(accuracies) if accuracies else 0.0,
        np.mean(precisions) if precisions else 0.0,
        np.mean(recalls) if recalls else 0.0,
        np.mean(f1s) if f1s else 0.0
    )


def do_eval_epoch(probes: nn.ModuleList, eval_ds: Dataset, phoneme_count: int, 
                 model_type: str, split: str, hidden_states_dir: pathlib.Path,
                 num_layers: int) -> Tuple[float, List[dict]]:
    for probe in probes:
        probe.eval()
    
    dataset_order = list(range(len(eval_ds)))
    random.shuffle(dataset_order)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    layer_metrics = [{} for _ in range(num_layers)]
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
            
            for layer_idx in range(num_layers):
                try:
                    hidden_states = load_hidden_states(idx, layer_idx, model_type, split, hidden_states_dir)
                    
                    token_repr = extract_token_representation(hidden_states, start_position, end_position)
                    
                    probe_output = probes[layer_idx](token_repr.unsqueeze(0))
                    pooled_probe = probe_output.squeeze(0)
                    
                    loss_vec = criterion(pooled_probe, label_vector)
                    masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)
                    layer_losses.append(masked_loss)
                    
                    metric = layer_metrics[layer_idx]
                    pred = (torch.sigmoid(pooled_probe) > 0.5).bool()

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
    return avg_loss, layer_metrics


def compute_layer_feature_heatmap(layer_metrics: List[dict], phoneme_mapping: dict, metric: str = 'f1') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Layer')
    ax.set_title(f'{metric.upper()} by layer x feature')

    phoneme_count = len(phoneme_mapping['features'])
    metric_mat = np.zeros((len(layer_metrics), phoneme_count))
    for li, lm in enumerate(layer_metrics):
        for feat in lm.keys():
            if metric in lm[feat]:
                metric_mat[li, feat-1] = lm[feat][metric]

    im = ax.imshow(metric_mat, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label=metric.upper())

    feature_names = sorted(phoneme_mapping['features'].keys(), key=lambda k: phoneme_mapping['features'][k])
    ax.set_xticks(list(range(phoneme_count)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticks(list(range(len(layer_metrics))))

    return fig


def log_layer_feature_metrics(layer_metrics: List[dict], phoneme_mapping: dict, fname: pathlib.Path):
    feature_names = sorted(phoneme_mapping['features'].keys(), key=lambda k: phoneme_mapping['features'][k])
    lines = [
        ['layer', 'feature', 'true_positive', 'false_positive', 'false_negative', 'true_negative', 
         'accuracy', 'precision', 'recall', 'f1']
    ]
    for li, metric in enumerate(layer_metrics):
        for fi in metric.keys():
            if 'matrix' not in metric[fi]:
                continue
            entry_line = [
                li, feature_names[fi-1],
                metric[fi]['matrix']['tp'],
                metric[fi]['matrix']['fp'],
                metric[fi]['matrix']['fn'],
                metric[fi]['matrix']['tn'],
                metric[fi].get('accuracy', 0),
                metric[fi].get('precision', 0),
                metric[fi].get('recall', 0),
                metric[fi].get('f1', 0),
            ]
            lines.append(list(map(str, entry_line)))
    
    s_lines = ['\t'.join(l) for l in lines]
    with open(fname, 'w+') as fp:
        fp.write('\n'.join(s_lines))


def do_train_run(cfg: dict, model_type: str, output_file: pathlib.Path, 
                num_layers: int, hidden_dim: int, cpus: int = os.cpu_count()) -> nn.ModuleList:
    wrun = wandb.init(
        entity='aaronjencks-the-ohio-state-university',
        project='ipa-probing-linear-probes-training',
        name=f'{model_type}_preextracted',
        config=cfg['hyperparameters']
    )

    logger.info('loading tokenizer')
    vocab, merges = utils.get_tokenizer_paths(cfg, model_type)
    tokenizer = utils.load_tokenizer(vocab, merges)
    
    logger.info('loading datasets')
    train_ds = load_and_preprocess(cfg, 'train', tokenizer, model_type, cpus)
    eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, model_type, cpus)
    test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)
    
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
    optimizer = torch.optim.AdamW(probes.parameters(), lr=hyperparameters['learning_rate'])
    
    hidden_states_dir = pathlib.Path('/fs/scratch/PAS2836/ipa_gpt/github/ipa-gpt-interpret/data/token_hidden_states')
    logger.info(f'Using hidden states directory: {hidden_states_dir}')
    
    logger.info('starting training')
    dataset_order = list(range(len(train_ds)))
    
    for epoch in range(hyperparameters['epochs']):
        random.shuffle(dataset_order)
        epoch_loss = 0.0
        errors = 0
        
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
            
            optimizer.zero_grad()
            layer_losses = []
            
            for layer_idx in range(num_layers):
                try:
                    hidden_states = load_hidden_states(idx, layer_idx, model_type, 'train', hidden_states_dir)
                    
                    token_repr = extract_token_representation(hidden_states, start_position, end_position)
                    
                    probe_output = probes[layer_idx](token_repr.unsqueeze(0))
                    pooled_probe = probe_output.squeeze(0)
                    
                    loss_vec = criterion(pooled_probe, label_vector)
                    masked_loss = (loss_vec * label_mask).sum() / label_mask.sum().clamp_min(1.0)
                    layer_losses.append(masked_loss)
                
                except FileNotFoundError as e:
                    logger.error(f"Could not load hidden states for idx={idx}, layer={layer_idx}: {e}")
                    errors += 1
                    break
            
            if not layer_losses:
                continue
            
            loss = torch.stack(layer_losses).mean()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= max(len(train_ds) - errors, 1)
        
        eval_loss, eval_metrics = do_eval_epoch(
            probes, eval_ds, phoneme_count, model_type, 'validation', 
            hidden_states_dir, num_layers
        )
        
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
        
        eval_hm = compute_layer_feature_heatmap(eval_metrics, phoneme_mappings)
        log_entry['eval/f1_heatmap'] = wandb.Image(eval_hm)
        wandb.log(log_entry, step=epoch)
        plt.close(eval_hm)
    
    test_loss, test_metrics = do_eval_epoch(
        probes, test_ds, phoneme_count, model_type, 'test', 
        hidden_states_dir, num_layers
    )
    
    log_entry = {'test/loss': test_loss}
    for layer in range(num_layers):
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
    return probes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], 
                   help='The model type')
    ap.add_argument('--num-layers', type=int, default=12, 
                   help='Number of layers in the model')
    ap.add_argument('--hidden-dim', type=int, default=768, 
                   help='Hidden dimension size')
    ap.add_argument('--output-log', type=str, default='probing_results_preextracted', 
                   help='The file to store the final probing accuracies in')
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.default_config)

    logger.info(f'training probes on {DEVICE}')

    for mt in args.model_type:
        output_path = pathlib.Path('data') / f'{args.output_log}_{mt}.tsv'
        probes = do_train_run(cfg, mt, output_path, args.num_layers, args.hidden_dim, args.cpus)