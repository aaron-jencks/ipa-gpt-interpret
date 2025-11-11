import argparse
import logging
import os
import pathlib
from typing import List

from datasets import load_dataset, Dataset
import numpy as np
from transformers import GPT2TokenizerFast

import config
import utils
from model import GPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def numpy_topk(arr, k):
    ind = np.argpartition(arr, -k)[-k:]
    return ind[np.argsort(arr[ind])]


# 1) Minimal postprocess: logits -> span text
def postprocess_probe_predictions(cfg, examples, model: GPT) -> List[int]:
    logger.info('extracting contexts and offsets')

    if len(raw_predictions) == 2:
        start_logits, end_logits = raw_predictions
    elif len(raw_predictions) == 3:
        start_logits, end_logits, _ = raw_predictions
    else:
        raise ValueError('unrecognized raw_predictions value')
    assert len(features) == len(examples)

    logger.info('starting evaluation loop')

    preds = {}
    use_ids = "id" in examples.column_names

    for i in range(len(features)):
        context = contexts[i]
        offsets = offset_maps[i]

        s_log = start_logits[i]
        e_log = end_logits[i]

        first_score = True
        best_score, best_idx = -1e9, -1
        tried_answers = {}

        # top-k search that enforces e >= s and max_answer_length
        start_idxes = numpy_topk(s_log, n_best_size)
        end_idxes = numpy_topk(e_log, n_best_size)

        for s in start_idxes:
            tried_answers[s] = None
            for e in end_idxes:
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue
                s_off, e_off = offsets[s], offsets[e]
                if s_off == (0, 0) or e_off == (0, 0):
                    continue  # skip non-context/special/pad
                score = s_log[s] + e_log[e]
                text = context[s_off[0]:e_off[1]]
                answer_dict = {
                    'start': s_off[0],
                    'end': e_off[1],
                    'text': text,
                    'score': score,
                    'logits': (s_log, e_log),
                    'logit_indices': (s, e)
                }
                if tried_answers[s] is None or score > tried_answers[s]['score']:
                    tried_answers[s] = answer_dict
                if first_score or score > best_score:
                    first_score = False
                    best_score = score
                    best_idx = s

        ex_key = examples["id"][i] if use_ids else str(i)
        preds[ex_key] = {
            'answers': tried_answers,
            'best_idx': best_idx,
        }

    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap = utils.setup_default_args(ap)
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], help='The model type')
    ap.add_argument('--eval-only', type=pathlib.Path, default=None, help='If supplied, specifies a checkpoint to evaluate, training is skipped, assumes that it is a trainer checkpoint')
    args = ap.parse_args()
    cfg, db = config.load_config(args.config, args.default_config)

    for mt in args.model_type:
        do_train_run(
            args.job_number, cfg, db,
            args.train_langs, args.eval_langs, mt,
            args.training_eval_size,
            args.sample_examples, args.display_incorrect,
            args.cpus,
            args.debug, args.eval_only
        )