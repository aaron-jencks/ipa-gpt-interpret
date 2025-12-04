import argparse
import json
import logging
import os
import pathlib
from typing import List

from datasets import Dataset
import numpy as np
import torch
from tqdm import tqdm

import config
import utils
from probing_new import load_and_preprocess, HiddenStateGPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_sample_data(model_type: str, output_dir: pathlib.Path, samples: List[dict]) -> List[str]:
    filenames = []
    for batch_samples in tqdm(samples, desc="Saving samples"):
        dataset_idx = batch_samples["dataset_idx"]
        layer_states = np.stack(batch_samples["layer_states"], axis=0)
        filename = output_dir / f"{dataset_idx}_{model_type}.npy"
        np.save(filename, layer_states)
        filenames.append(str(filename))
    return filenames


@torch.no_grad()
def extract_hidden_states_per_token(
    model: HiddenStateGPT,
    dataset: Dataset,
    output_dir: pathlib.Path,
    split: str,
    model_type: str,
    batch_size: int = 128,
    accumulation_size: int = 10_000,
) -> dict:
    model.eval()

    token_states_dir = output_dir / f'token_hidden_states_{model_type}_{split}'
    token_states_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'n_layers': 12,
        'model_type': model_type,
        'filenames': [],
    }
    
    logger.info(f"Extracting per-token hidden states from {len(dataset)} samples")

    samples = []
    
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        
        # Gather batch data
        batch_input_ids = []
        batch_valid_indices = []
        batch_slices = []
        
        for idx in range(batch_start, batch_end):
            row = dataset[idx]
            if row['start_positions'] < 0 or row['end_positions'] < 0:
                continue
            batch_input_ids.append(row['input_ids'])
            batch_valid_indices.append(idx)
            batch_slices.append((row['start_positions'], row['end_positions']))
        
        if len(batch_input_ids) == 0:
            continue
        
        input_ids_tensor = torch.tensor(batch_input_ids, device=DEVICE).long()
        layer_hidden_states = model(input_ids_tensor)
        
        # Process each sample in the batch
        for i in range(len(batch_input_ids)):
            start, stop = batch_slices[i]
            samples.append({
                'dataset_idx': batch_valid_indices[i],
                'layer_states': [layer_hs[i, start:stop, :].cpu().numpy().astype(np.float16) for layer_hs in layer_hidden_states]
            })
            if len(samples) == (accumulation_size // batch_size):
                metadata['filenames'] += save_sample_data(model_type, token_states_dir, samples)
                samples = []
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    if len(samples) > 0:
        metadata['filenames'] += save_sample_data(model_type, token_states_dir, samples)
    
    logger.info(f"Extracted hidden states for {len(metadata['filenames'])} samples")
    metadata_filename = token_states_dir / 'metadata.json'
    with open(metadata_filename, 'w+') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_filename}")
    
    return metadata


def main(
        cfg: dict, model_type: str, output_dir: pathlib.Path,
        batch_size: int = 128, accumulation_size: int = 10_000,
        cpus: int = os.cpu_count()
):
    vocab, merges = utils.get_tokenizer_paths(cfg, model_type)
    tokenizer = utils.load_tokenizer(vocab, merges)
    
    logger.info('Loading training dataset')
    train_ds = load_and_preprocess(cfg, 'train', tokenizer, model_type, cpus)
    
    logger.info('Loading evaluation dataset')
    eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, model_type, cpus)
    
    # logger.info('Loading test dataset')
    # test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)
    
    logger.info('Loading phoneme mapping file')
    mapping_path = pathlib.Path(cfg['mappings'])
    with open(mapping_path, 'r', encoding='utf-8') as fp:
        phoneme_mappings = json.load(fp)
    phoneme_count = len(phoneme_mappings['features'])
    
    logger.info('Loading model')
    base_model = utils.load_pretrained_model(cfg, model_type, device=DEVICE)
    model = HiddenStateGPT(base_model, phoneme_count)
    model.compile()  # compile the model for speed
    model = model.to(DEVICE)
    
    logger.info("Extracting Training Set")
    train_metadata = extract_hidden_states_per_token(
        model, train_ds, output_dir, 'train', model_type, batch_size=batch_size,
        accumulation_size=accumulation_size
    )
    
    logger.info("Extracting Validation Set")
    eval_metadata = extract_hidden_states_per_token(
        model, eval_ds, output_dir, 'validation', model_type, batch_size=batch_size,
        accumulation_size=accumulation_size
    )
    
    # logger.info("Extracting Test Set")
    # test_metadata = extract_hidden_states_per_token(
    #     model, test_ds, output_dir, 'test', model_type, batch_size=batch_size,
    #     accumulation_size=accumulation_size
    # )
    
    # Print summary statistics
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Train samples: {len(train_metadata['filenames'])}")
    logger.info(f"Eval samples: {len(eval_metadata['filenames'])}")
    # logger.info(f"Test samples: {len(test_metadata['filenames'])}")
    logger.info(f"Layers per sample: {train_metadata['n_layers']}")
    logger.info(f"Total files created: {(len(train_metadata['filenames']) + len(eval_metadata['filenames']))}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract per-token hidden states and save to numpy files')
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], 
                    help='The model type(s) to process')
    ap.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('data/token_hidden_states'),
                    help='Directory to save per-token hidden states')
    ap.add_argument('--batch-size', type=int, default=128,
                    help='Batch size for hidden state extraction')
    ap.add_argument('--accumulation-size', type=int, default=10_000, help='The number of samples to store before saving')
    args = ap.parse_args()
    
    cfg = config.load_config(args.config, args.default_config)
    
    logger.info(f'Running on {DEVICE}')
    logger.info(f'Output directory: {args.output_dir}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Model types: {args.model_type}')
    
    for mt in args.model_type:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model type: {mt}")
        logger.info(f"{'='*60}\n")
        main(cfg, mt, args.output_dir, args.batch_size, args.accumulation_size, args.cpus)
