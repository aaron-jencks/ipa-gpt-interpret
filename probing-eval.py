import argparse
import json
import logging
import pathlib
from logging import Logger

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

import config
import utils
from probing_exp_routines import do_eval_epoch
from probing_exp_utils import LinearProbe, find_latest_checkpoint, load_checkpoint, save_checkpoint, \
    compute_macro_metrics, compute_layer_feature_heatmap, log_layer_feature_metrics, load_probe_checkpoint_no_optimizers
from probing_new import load_and_preprocess

logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'],
                    help='The model type')
    ap.add_argument('--output-log', type=str, default='probing_results_preextracted',
                    help='The file to store the final probing accuracies in')
    ap.add_argument('--average-span', action='store_true',
                    help='Average all tokens in the span instead of using last token')
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.default_config)

    logger.info(f'training probes on {DEVICE}')

    checkpoint_dir = pathlib.Path(cfg['checkpoints']['probe_prefix'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for mt in args.model_type:
        output_path = pathlib.Path('data') / f'{args.output_log}_{mt}.tsv'

        extraction_method = 'avgspan' if args.average_span else 'lasttoken'
        project_name = f'ipa-final-probing-linear-classifier-{mt}-{extraction_method}'
        run_name = f'{mt}_{extraction_method}_preextracted'

        wrun = wandb.init(
            entity='aaronjencks-the-ohio-state-university',
            project=project_name,
            name=run_name,
            config=cfg['hyperparameters'],
        )

        logger.info('loading tokenizer')
        vocab, merges = utils.get_tokenizer_paths(cfg, mt)
        tokenizer = utils.load_tokenizer(vocab, merges)

        logger.info('loading datasets')
        train_ds = load_and_preprocess(cfg, 'train', tokenizer, mt, args.cpus)
        eval_ds = load_and_preprocess(cfg, 'validation', tokenizer, mt, args.cpus)
        # test_ds = load_and_preprocess(cfg, 'test', tokenizer, model_type, cpus)

        logger.info('loading phoneme mapping file')
        mapping_path = pathlib.Path(cfg['mappings'])
        with open(mapping_path, 'r') as fp:
            phoneme_mappings = json.load(fp)
        phoneme_count = len(phoneme_mappings['features'])

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

        eval_loss, eval_metrics = do_eval_epoch(
            probes,
            eval_ds, phoneme_count, mt, 'validation',
            hidden_states_dir, num_layers=12, average_span=args.average_span
        )

        log_entry = {
            'eval/loss': eval_loss,
        }
        for layer in range(12):
            l_acc, l_prec, l_rec, l_f1 = compute_macro_metrics(eval_metrics, layer)
            log_entry[f'eval/accuracy/layer_{layer:02d}'] = l_acc
            log_entry[f'eval/precision/layer_{layer:02d}'] = l_prec
            log_entry[f'eval/recall/layer_{layer:02d}'] = l_rec
            log_entry[f'eval/f1/layer_{layer:02d}'] = l_f1

        eval_hm = compute_layer_feature_heatmap(eval_metrics, phoneme_mappings)
        log_entry['eval/f1_heatmap'] = wandb.Image(eval_hm)
        wandb.log(log_entry)
        plt.close(eval_hm)

        log_layer_feature_metrics(eval_metrics, phoneme_mappings, output_path)

        wrun.finish()
