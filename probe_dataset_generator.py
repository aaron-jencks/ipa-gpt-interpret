import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple
import torch
from tqdm import tqdm
from datasets import Dataset
import config
import utils
from probing_new import load_and_preprocess, ProbedGPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_hidden_states(
    model: ProbedGPT, 
    dataset: Dataset, 
    batch_size: int = 32
) -> Tuple[List[torch.Tensor], List[int], List[int], List[List[int]]]:
    model.eval()
    
    all_hidden_states = []
    start_positions = []
    end_positions = []
    features = []
    
    logger.info(f"Extracting hidden states from {len(dataset)} samples with batch_size={batch_size}")
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting hidden states"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(dataset))
        
        # Gather batch data
        batch_input_ids = []
        batch_starts = []
        batch_ends = []
        batch_features = []
        
        for idx in range(batch_start, batch_end):
            row = dataset[idx]
            if row['start_positions'] < 0 or row['end_positions'] < 0:
                continue
            
            batch_input_ids.append(row['input_ids'])
            batch_starts.append(row['start_positions'])
            batch_ends.append(row['end_positions'])
            batch_features.append(row['features'][0])
        
        if len(batch_input_ids) == 0:
            continue
        
        input_ids_tensor = torch.tensor(batch_input_ids, device=DEVICE).long()
        
        with torch.no_grad():
            layer_hidden_states = model(input_ids_tensor)
        

        for i in range(len(batch_input_ids)):
            sample_hidden = [layer_hs[i] for layer_hs in layer_hidden_states]
            # (n_layers, seq_len, n_embd)
            sample_hidden = torch.stack(sample_hidden)
            
            all_hidden_states.append(sample_hidden.cpu())
            start_positions.append(batch_starts[i])
            end_positions.append(batch_ends[i])
            features.append(batch_features[i])
    
    logger.info(f"Extracted hidden states for {len(all_hidden_states)} samples")
    return all_hidden_states, start_positions, end_positions, features


def create_probe_dataset_dict(
    hidden_states: List[torch.Tensor],
    start_positions: List[int],
    end_positions: List[int],
    features: List[List[int]],
    phoneme_count: int
) -> dict:
    
    pooled_hidden_states_list = []
    labels_list = []
    masks_list = []
    
    logger.info(f"Creating probe dataset from {len(hidden_states)} samples")
    
    for i in tqdm(range(len(hidden_states)), desc="Creating probe dataset"):
        # (n_layers, seq_len, n_embd)
        hs = hidden_states[i]
        start = start_positions[i]
        end = end_positions[i]
        
        span_hidden = hs[:, start:end+1, :]
        pooled = span_hidden.mean(dim=1)
        
        label_vector = torch.zeros(phoneme_count)
        mask = torch.ones(phoneme_count)
        for f in features[i]:
            if f > 0:
                label_vector[f-1] = 1
            else:
                mask[abs(f)-1] = 0
        
        pooled_hidden_states_list.append(pooled)
        labels_list.append(label_vector)
        masks_list.append(mask)
    

    pooled_hidden_states = torch.stack(pooled_hidden_states_list)
    labels = torch.stack(labels_list)
    masks = torch.stack(masks_list)
    
    logger.info(f"Created Probe dataset with {len(pooled_hidden_states)} samples")
    
    return {
        'pooled_hidden_states': pooled_hidden_states,
        'labels': labels,
        'masks': masks,
        'phoneme_count': phoneme_count,
        'n_layers': pooled_hidden_states.shape[1]
    }


def save_probe_dataset(dataset_dict: dict, save_path: pathlib.Path):
    logger.info(f"Saving probe dataset to {save_path}")
    torch.save(dataset_dict, save_path)
    logger.info(f"Dataset saved successfully")


def main(cfg: dict, model_type: str, output_dir: pathlib.Path, batch_size: int = 32, cpus: int = os.cpu_count()):
    # All the loading 
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
    model = ProbedGPT(base_model, phoneme_count).to(DEVICE)
    

    # Extract hidden states
    train_hidden, train_starts, train_ends, train_features = extract_hidden_states(
        model, train_ds, batch_size=batch_size
    )
    eval_hidden, eval_starts, eval_ends, eval_features = extract_hidden_states(
        model, eval_ds, batch_size=batch_size
    )
    test_hidden, test_starts, test_ends, test_features = extract_hidden_states(
        model, test_ds, batch_size=batch_size
    )  
    
    train_probe_dict = create_probe_dataset_dict(
        train_hidden, train_starts, train_ends, train_features, phoneme_count
    )
    eval_probe_dict = create_probe_dataset_dict(
        eval_hidden, eval_starts, eval_ends, eval_features, phoneme_count
    )
    test_probe_dict = create_probe_dataset_dict(
        test_hidden, test_starts, test_ends, test_features, phoneme_count
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_probe_dataset(train_probe_dict, output_dir / f'train_probe_dataset_{model_type}.pt')
    save_probe_dataset(eval_probe_dict, output_dir / f'eval_probe_dataset_{model_type}.pt')
    save_probe_dataset(test_probe_dict, output_dir / f'test_probe_dataset_{model_type}.pt')
    
    logger.info(f'Train samples: {len(train_probe_dict["pooled_hidden_states"])}')
    logger.info(f'Eval samples: {len(eval_probe_dict["pooled_hidden_states"])}')
    logger.info(f'Test samples: {len(test_probe_dict["pooled_hidden_states"])}')
    logger.info(f'Hidden state shape: {train_probe_dict["pooled_hidden_states"].shape}')
    logger.info(f'Labels shape: {train_probe_dict["labels"].shape}')



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract hidden states and make probe datasets')
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], 
                    help='The model type(s) to process')
    ap.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path('data/probe_datasets'),
                    help='Directory to save probe datasets')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for hidden state extraction')
    args = ap.parse_args()
    
    cfg = config.load_config(args.config, args.default_config)
    
    logger.info(f'Running on {DEVICE}')
    
    for mt in args.model_type:
        main(cfg, mt, args.output_dir, args.batch_size, args.cpus)