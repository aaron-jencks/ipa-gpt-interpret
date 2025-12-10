import json
import logging
import pathlib
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def load_hidden_states(dataset_idx: int, model_type: str, split: str,
                       hidden_states_dir: pathlib.Path) -> np.ndarray:
    """
    Loads the pre-extracted span-only hidden states for a given row.

    New file format: {idx}_{model_type}.npy
    Shape: (num_layers, span_len, hidden_dim)
    """
    model_dir = hidden_states_dir / f"token_hidden_states_{model_type}_{split}"
    file_path = model_dir / f"{dataset_idx}_{model_type}.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Hidden state file not found: {file_path}")

    # Load the whole thing (small because span_len <= 20)
    return np.load(file_path, mmap_mode="r")


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
            binary_vector[f - 1] = 1
        else:
            mask[abs(f) - 1] = 0
    return binary_vector.float(), mask.float()


def extract_token_representation(hidden_states: np.ndarray,
                                 layer_idx: int,
                                 average_span: bool = False) -> torch.Tensor:
    """
    hidden_states: (num_layers, span_len, hidden_dim)
    layer_idx: which layer we are probing
    """
    layer_hs = hidden_states[layer_idx]  # (span_len, hidden_dim)

    if average_span:
        vec = layer_hs.mean(axis=0)  # (hidden_dim,)
    else:
        vec = layer_hs[-1]  # last token (hidden_dim,)

    return torch.from_numpy(vec).float().to(DEVICE)


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


def save_checkpoint(checkpoint_dir: pathlib.Path, model_type: str, epoch: int,
                    probes: nn.ModuleList, optimizers: List[torch.optim.Optimizer],
                    train_loss: float, eval_loss: float, eval_metrics: List[dict],
                    num_layers: int):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_type': model_type,
        'num_layers': num_layers,
        'probe_state_dicts': [probe.state_dict() for probe in probes],
        'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'eval_metrics': eval_metrics,
    }

    checkpoint_path = checkpoint_dir / f"{model_type}_epoch{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Also save a "latest" checkpoint for easy resuming
    latest_path = checkpoint_dir / f"{model_type}_latest.pt"
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved latest checkpoint to {latest_path}")


def load_probe_checkpoint_no_optimizers(checkpoint_path: pathlib.Path, probes: nn.ModuleList):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    for probe, state_dict in zip(probes, checkpoint['probe_state_dicts']):
        probe.load_state_dict(state_dict)


def load_checkpoint(checkpoint_path: pathlib.Path, probes: nn.ModuleList,
                    optimizers: List[torch.optim.Optimizer]) -> Tuple[int, float, float, List[dict]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    for probe, state_dict in zip(probes, checkpoint['probe_state_dicts']):
        probe.load_state_dict(state_dict)

    for optimizer, state_dict in zip(optimizers, checkpoint['optimizer_state_dicts']):
        optimizer.load_state_dict(state_dict)

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    eval_loss = checkpoint['eval_loss']
    eval_metrics = checkpoint['eval_metrics']

    logger.info(f"Resumed from epoch {epoch}, train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}")

    return epoch, train_loss, eval_loss, eval_metrics


def find_latest_checkpoint(checkpoint_dir: pathlib.Path, model_type: str) -> Optional[pathlib.Path]:
    latest_path = checkpoint_dir / f"{model_type}_latest.pt"

    logger.info(f'attempting to find latest checkpoint at {latest_path}')

    if latest_path.exists():
        return latest_path

    logger.info(f'no latest checkpoint found, checking for epoch checkpoints')

    # Fallback
    checkpoints = list(checkpoint_dir.glob(f"{model_type}_epoch*.pt"))
    if checkpoints:
        logger.info(f'found {len(checkpoints)} checkpoints')
        checkpoints.sort(key=lambda p: int(p.stem.split('epoch')[1]))
        return checkpoints[-1]

    logger.info('no checkpoints found to load!')

    return None


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
                metric_mat[li, feat - 1] = lm[feat][metric]

    im = ax.imshow(metric_mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
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
                li, feature_names[fi - 1],
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
