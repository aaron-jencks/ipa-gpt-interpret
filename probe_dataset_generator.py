import argparse
import json
import logging
import os
import pathlib
from typing import Dict
import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
import config
import utils
from probing_new import load_and_preprocess, HiddenStateGPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_hidden_states_per_token(
    model: ProbedGPT, 
    dataset: Dataset,
    output_dir: pathlib.Path,
    model_type: str,
    batch_size: int = 128
) -> dict:
    model.eval()
    

    token_states_dir = output_dir / f'token_hidden_states_{model_type}'
    token_states_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'samples': [],
        'n_layers': None,
        'model_type': model_type
    }
    
    logger.info(f"Extracting per-token hidden states from {len(dataset)} samples")
    
    sample_id = 0
    
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        
        # Gather batch data
        batch_input_ids = []
        batch_starts = []
        batch_ends = []
        batch_features = []
        batch_valid_indices = []
        
        for idx in range(batch_start, batch_end):
            row = dataset[idx]
            if row['start_positions'] < 0 or row['end_positions'] < 0:
                continue
            
            batch_input_ids.append(row['input_ids'])
            batch_starts.append(row['start_positions'])
            batch_ends.append(row['end_positions'])
            batch_features.append(row['features'][0])
            batch_valid_indices.append(idx)
        
        if len(batch_input_ids) == 0:
            continue
        
        input_ids_tensor = torch.tensor(batch_input_ids, device=DEVICE).long()
        
        with torch.no_grad():
            layer_hidden_states = model(input_ids_tensor)
        

        if metadata['n_layers'] is None:
            metadata['n_layers'] = len(layer_hidden_states)
        
        # Process each sample in the batch
        for i in range(len(batch_input_ids)):
            start_pos = batch_starts[i]
            end_pos = batch_ends[i]
            span_length = end_pos - start_pos + 1
            
            sample_metadata = {
                'sample_id': sample_id,
                'original_dataset_idx': batch_valid_indices[i],
                'start_position': start_pos,
                'end_position': end_pos,
                'span_length': span_length,
                'features': batch_features[i],
                'layer_files': []
            }
            

            for layer_idx, layer_hs in enumerate(layer_hidden_states):
                span_tokens = layer_hs[i, start_pos:end_pos+1, :]
                filename = f'{sample_id}_{layer_idx}_{model_type}.npy'
                filepath = token_states_dir / filename
                np.save(filepath, span_tokens.cpu().numpy())
                
                sample_metadata['layer_files'].append(str(filepath))
            metadata['samples'].append(sample_metadata)
            sample_id += 1
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
    logger.info(f"Extracted hidden states for {len(metadata['samples'])} samples")
    logger.info(f"Created {len(metadata['samples']) * metadata['n_layers']} layer files")

    metadata_path = output_dir / f'metadata_{model_type}.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    return metadata


def main(cfg: dict, model_type: str, output_dir: pathlib.Path, batch_size: int =128, cpus: int = os.cpu_count()):
    vocab, merges = utils.get_tokenizer_paths(cfg, model_type)
    tokenizer = utils.load_tokenizer(vocab, merges)
    
    logger.info('Loading training dataset')
    train_ds = load_and_preprocess(cfg, 'train', tokenizer, model_type, cpus)
    
    logger.info('Loading evaluation dataset')
    eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, model_type, cpus)
    
    logger.info('Loading test dataset')
    test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)
    
    logger.info('Loading phoneme mapping file')
    mapping_path = pathlib.Path(cfg['mappings'])
    with open(mapping_path, 'r', encoding='utf-8') as fp:
        phoneme_mappings = json.load(fp)
    phoneme_count = len(phoneme_mappings['features'])
    
    logger.info('Loading model')
    base_model = utils.load_pretrained_model(cfg, model_type, device=DEVICE)
    model = HiddenStateGPT(base_model, phoneme_count)
    model = model.compile().to(DEVICE)  # compile the model for speed
    
    logger.info("Extracting Training Set")
    train_metadata = extract_hidden_states_per_token(
        model, train_ds, output_dir, model_type, batch_size=batch_size
    )
    
    logger.info("Extracting Validation Set")
    eval_metadata = extract_hidden_states_per_token(
        model, eval_ds, output_dir, model_type, batch_size=batch_size
    )
    
    logger.info("Extracting Test Set")
    test_metadata = extract_hidden_states_per_token(
        model, test_ds, output_dir, model_type, batch_size=batch_size
    )
    
    # Print summary statistics
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Train samples: {len(train_metadata['samples'])}")
    logger.info(f"Eval samples: {len(eval_metadata['samples'])}")
    logger.info(f"Test samples: {len(test_metadata['samples'])}")
    logger.info(f"Layers per sample: {train_metadata['n_layers']}")
    logger.info(f"Total files created: {(len(train_metadata['samples']) + len(eval_metadata['samples']) + len(test_metadata['samples'])) * train_metadata['n_layers']}")
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
        main(cfg, mt, args.output_dir, args.batch_size, args.cpus)
