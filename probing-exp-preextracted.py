import argparse
import logging
import pathlib

import config
import utils
from probing_exp_routines import do_train_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], 
                   help='The model type')
    ap.add_argument('--output-log', type=str, default='probing_results_preextracted', 
                   help='The file to store the final probing accuracies in')
    ap.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                   help='Directory to save/load checkpoints')
    ap.add_argument('--resume', action='store_true',
                   help='Resume from latest checkpoint if available')
    ap.add_argument('--average-span', action='store_true',
                   help='Average all tokens in the span instead of using last token')
    args = ap.parse_args()
    cfg = config.load_config(args.config, args.default_config)
    
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for mt in args.model_type:
        output_path = pathlib.Path('data') / f'{args.output_log}_{mt}.tsv'
        probes = do_train_run(cfg, mt, output_path, 12, 768,
                            checkpoint_dir, args.resume, args.average_span, args.cpus)