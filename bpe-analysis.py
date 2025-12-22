import argparse
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
import os
import pathlib
from queue import Full
from typing import Dict, Set

from datasets import concatenate_datasets, load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GLOBAL_DATASET = None
VOCAB_SIZE = 50_001


@dataclass
class Config:
    vocab: pathlib.Path
    merges: pathlib.Path
    feature: str


def counting_daemon(qin: mp.Queue, arr: mp.Array, avg: mp.Array, qout: mp.Queue, lang_feature: str, lang_codes: Dict[str, int], cfg: Config):
    tokenizer = GPT2TokenizerFast(
        str(cfg.vocab), str(cfg.merges),
        add_prefix_space=True
    )

    while True:
        slice = qin.get()
        if slice is None:
            break
        slice_start, slice_end = slice
        records = GLOBAL_DATASET[slice_start:slice_end]
        langs = records[lang_feature]
        tokens = tokenizer(records[cfg.feature])['input_ids']
        for ri, row_tokens in enumerate(tokens):
            lang = langs[ri]
            lang_code = lang_codes[lang]
            lang_offset = lang_code * VOCAB_SIZE
            lang_avg, lang_count = avg[lang_code * 2:lang_code * 2 + 1]
            avg[lang_code * 2] = (lang_avg * lang_count + len(row_tokens)) / (lang_count + 1)
            avg[lang_code * 2 + 1] += 1
            for token in row_tokens:
                arr[lang_offset + token] += 1
        qout.put(slice_end - slice_start)


def log_inventories(
        directory: pathlib.Path,
        tokenizer: GPT2TokenizerFast, vocab: Dict[int, str],
        supports: Dict[str, Dict[int, int]],
        disjoint: Dict[str, Set[int]], shared: Set[int],
        avgs: Dict[str, float]
):
    logger.info(f'saving phonetic inventories to {directory}')

    for lang in disjoint.keys():
        lines = [
            '"token","string","byte","support"'
        ]
        for token in disjoint[lang]:
            ts = tokenizer.decode([token]).replace('\n', '\\n')
            bs = 'N/A'
            if ts == '�' or len(ts) == 1:
                bs = ' '.join(f'x{byte:02x}' for byte in vocab[token].encode('latin-1'))
            lines.append(f'{token},"{ts}","{bs}",{supports[lang][token]}')
        with open(directory / f'{lang}.csv', 'w+') as fp:
            fp.write('\n'.join(lines))

    shared_lines = [
        '"token","string","byte","support"'
    ]
    for token in shared:
        ts = tokenizer.decode([token]).replace('\n', '\\n')
        bs = 'N/A'
        if ts == '�' or len(ts) == 1:
            bs = ' '.join(f'x{byte:02x}' for byte in vocab[token].encode('latin-1'))
        support = sum(supports[lang][token] for lang in disjoint.keys())
        shared_lines.append(f'{token},"{ts}","{bs}",{support}')
    with open(directory / 'shared.csv', 'w+') as fp:
        fp.write('\n'.join(shared_lines))

    with open(directory / 'metrics.json', 'w+') as fp:
        metrics = {}
        for lang in disjoint.keys():
            metrics[lang] = {
                'average_length': avgs[lang],
                'count': len(disjoint[lang]),
            }
        metrics['shared'] = {
            'count': len(shared)
        }
        json.dump(metrics, fp, indent=4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='analyzes the bpe tokenizer token inventories of the language in the dataset')
    ap.add_argument('codes', type=pathlib.Path, help='indicates where the vocab and merges are stored')
    ap.add_argument('name', type=str, help='indicates the prefix to the vocab and merge files')
    ap.add_argument('--feature', type=str, default='text', help='the feature column name to apply')
    ap.add_argument('--lang-feature', type=str, default='language', help='the feature column name to extract the language names from')
    ap.add_argument('--cpus', type=int, default=os.cpu_count(), help='the number of cores to use')
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('./cache/huggingface'),
                        help='the location of the cache folder for huggingface')
    ap.add_argument('--dataset', type=str, default='openwebtext', help='the dataset to use')
    ap.add_argument('--batch-size', type=int, default=2000, help='the batch size of the tokenizing')
    ap.add_argument('--result-directory', type=pathlib.Path, default=pathlib.Path('data/token-analysis'), help='the directory to save the analysis results to')
    args = ap.parse_args()

    procs = []

    vocab_fname = args.codes / "{}-vocab.json".format(args.name)
    merges_fname = args.codes / "{}-merges.txt".format(args.name)

    logger.info(f'reading data from {str(vocab_fname)} and {str(merges_fname)}')

    logger.info('loading vocab indices...')
    with open(vocab_fname, 'rb') as fp:
        bdata = fp.read().decode('latin-1')
        vocab_data = json.loads(bdata)
    vocab_indices = {v: k for k, v in vocab_data.items()}
    tokenizer = GPT2TokenizerFast(
        str(vocab_fname), str(merges_fname),
        add_prefix_space=True
    )

    logger.info('loading dataset...')
    dataset = load_dataset(args.dataset, cache_dir=args.cache, num_proc=args.cpus)
    GLOBAL_DATASET = concatenate_datasets([dataset[split] for split in dataset])

    language_codes = {v: k for k, v in enumerate(GLOBAL_DATASET.unique(args.lang_feature))}

    logger.info('setting up output...')
    output_array = mp.Array('i', VOCAB_SIZE * len(language_codes))
    output_averages = mp.Array('d', len(language_codes) * 2)
    qout = mp.Queue()

    logger.info('generating processors...')
    processing_config = Config(vocab_fname, merges_fname, args.feature)

    queues = [mp.Queue() for _ in range(args.cpus)]
    procs = []
    for pi in range(args.cpus):
        proc = mp.Process(target=counting_daemon, args=(queues[pi], output_array, output_averages, qout, args.lang_feature, language_codes, processing_config))
        proc.start()
        procs.append(proc)

    qi = 0
    total_batches = 0
    for idx in tqdm(range(0, len(GLOBAL_DATASET), args.batch_size), desc='queueing up batches'):
        batch_end = min(idx + args.batch_size, len(GLOBAL_DATASET))
        total_batches += 1
        while True:
            try:
                queues[qi].put_nowait((idx, batch_end))
                qi = (qi + 1) % len(queues)
                break
            except Full as e:
                qi = (qi + 1) % len(queues)

    logger.info('finished feeding data, waiting for results...')
    for _ in tqdm(range(total_batches), desc='processing batches'):
        qout.get()
    for q in queues:
        q.put(None)
    pbar = tqdm(total=len(procs), desc='waiting for daemons to finish...')
    for p in procs:
        p.join()

    logger.info('analyzing data...')

    shared_inventory = None
    unfiltered_inventories = {}
    disjoint_inventories = {}
    for lang in language_codes.keys():
        lang_offset = language_codes[lang] * VOCAB_SIZE
        unfiltered_inventories[lang] = {t: s for t, s in enumerate(output_array[lang_offset:lang_offset + VOCAB_SIZE]) if s > 0}
        if shared_inventory is None:
            shared_inventory = set(unfiltered_inventories[lang].keys())
        else:
            shared_inventory &= set(unfiltered_inventories[lang].keys())
    for lang in language_codes.keys():
        disjoint_inventories[lang] = set(unfiltered_inventories[lang].keys()) - shared_inventory

    print('token inventories:')
    for lang in language_codes.keys():
        print(f'{lang}: Top 10 Tokens\n   \tToken\tCount')
        tokens = sorted(list(disjoint_inventories[lang]), key=lambda x: unfiltered_inventories[lang][x], reverse=True)
        print('token inventory count:', len(tokens))
        for ti, tok in enumerate(tokens[:10]):
            print(f'{ti + 1:02d}. & {tok} & "\ipa{{{tokenizer.decode([tok])}}}" & {unfiltered_inventories[lang][tok]:,d} \\\\')

    print('shared inventory')
    print(f'size of shared inventory: {len(shared_inventory)}')
    print('Top 10 Tokens\n   \tToken\tCount')
    tokens = sorted(
        list(shared_inventory),
        key=lambda x: sum([unfiltered_inventories[lang][x] for lang in language_codes.keys()]),
        reverse=True
    )
    for ti, tok in enumerate(tokens[:10]):
        print(f'{ti + 1:02d}. & {tok} & "\ipa{{{tokenizer.decode([tok])}}}" &  {sum([unfiltered_inventories[lang][tok] for lang in language_codes.keys()]):,d} \\\\')

    os.makedirs(args.result_directory, exist_ok=True)
    log_inventories(
        args.result_directory,
        tokenizer, vocab_indices,
        unfiltered_inventories,
        disjoint_inventories, shared_inventory,
        output_averages,
    )
