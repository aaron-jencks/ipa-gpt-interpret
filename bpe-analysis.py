import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import pathlib
from queue import Full

from datasets import concatenate_datasets, load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    vocab: pathlib.Path
    merges: pathlib.Path
    feature: str


def counting_daemon(qin: mp.Queue, qout: mp.Queue, cfg: Config):
    tokenizer = GPT2TokenizerFast(
        str(cfg.vocab), str(cfg.merges),
        add_prefix_space=True
    )

    while True:
        records = qin.get()
        if records is None:
            break
        langs = records['language']
        tokens = tokenizer(records[cfg.feature])['input_ids']
        batch_inventory = {}
        for ri, row_tokens in enumerate(tokens):
            lang = langs[ri]
            if lang not in batch_inventory:
                batch_inventory[lang] = {}
            for token in row_tokens:
                if token not in batch_inventory[lang]:
                    batch_inventory[lang][token] = 1
                else:
                    batch_inventory[lang][token] += 1
        qout.put(batch_inventory)


def collator_daemon(qin: mp.Queue, qout: mp.Queue, batch_size: int, total_length: int):
    result = {}

    pbar = tqdm(total=total_length, desc='collecting data')
    while True:
        job = qin.get()
        if job is None:
            break
        batch_inventory = job
        for lang in batch_inventory:
            inventory = batch_inventory[lang]
            if lang not in result:
                result[lang] = inventory
            else:
                for token in inventory:
                    if token not in result[lang]:
                        result[lang][token] = inventory[token]
                    else:
                        result[lang][token] += inventory[token]
        pbar.update(batch_size)
    pbar.close()

    qout.put(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='analyzes the bpe tokenizer token inventories of the language in the dataset')
    ap.add_argument('codes', type=pathlib.Path, help='indicates where the vocab and merges are stored')
    ap.add_argument('name', type=str, help='indicates the prefix to the vocab and merge files')
    ap.add_argument('--feature', type=str, default='text', help='the feature column name to apply')
    ap.add_argument('--cpus', type=int, default=os.cpu_count(), help='the number of cores to use')
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('./cache/huggingface'),
                        help='the location of the cache folder for huggingface')
    ap.add_argument('--dataset', type=str, default='openwebtext', help='the dataset to use')
    ap.add_argument('--batch-size', type=int, default=2000, help='the batch size of the tokenizing')
    args = ap.parse_args()

    procs = []

    vocab_fname = args.codes / "{}-vocab.json".format(args.name)
    merges_fname = args.codes / "{}-merges.txt".format(args.name)

    logger.info(f'reading data from {str(vocab_fname)} and {str(merges_fname)}')

    logger.info('loading vocab indices...')
    with open(vocab_fname) as fp:
        vocab_data = json.load(fp)
    vocab_indices = {v: k for k, v in vocab_data.items()}
    tokenizer = GPT2TokenizerFast(
        str(vocab_fname), str(merges_fname),
        add_prefix_space=True
    )

    logger.info('loading dataset...')
    dataset = load_dataset(args.dataset, cache_dir=args.cache)
    merged_dataset = concatenate_datasets([dataset[split] for split in dataset])

    logger.info('generating processors...')
    processing_config = Config(vocab_fname, merges_fname, args.feature)

    queues = [mp.Queue() for _ in range(args.cpus)]
    qout = mp.Queue()
    final_qout = mp.Queue()
    collator = mp.Process(target=collator_daemon, args=(qout, final_qout, args.batch_size, len(merged_dataset)))
    collator.start()
    procs = []
    for pi in range(args.cpus):
        proc = mp.Process(target=counting_daemon, args=(queues[pi], qout, processing_config))
        proc.start()
        procs.append(proc)

    qi = 0
    for idx in range(0, len(merged_dataset), args.batch_size):
        batch_end = min(idx + args.batch_size, len(merged_dataset))
        batch = merged_dataset[idx:batch_end]
        while True:
            try:
                queues[qi].put_nowait(batch)
                qi = (qi + 1) % len(queues)
                break
            except Full as e:
                qi = (qi + 1) % len(queues)

    logger.info('finished feeding data, waiting for results...')
    for q in queues:
        q.put(None)
    for p in procs:
        p.join()
    qout.put(None)
    inventories = final_qout.get()
    collator.join()

    logger.info('analyzing data...')
    logger.info('token inventories:')
    for lang in inventories:
        logger.info(f'{lang}: Top 10 Tokens\n   \tToken\tCount')
        tokens = sorted(list(inventories[lang].keys()), key=lambda x: inventories[lang][x], reverse=True)
        print('token inventory count:', len(tokens))
        for ti, tok in enumerate(tokens[:10]):
            print(f'{ti + 1:02d}. & {tok} & "\ipa{{{tokenizer.decode([tok])}}}" & {inventories[lang][tok]:,d} \\\\')

    logger.info('shared inventory')
    lang_list = list(inventories.keys())
    assert len(lang_list) == 2, "multilingual models not supported"
    merged_inventory = deepcopy(inventories[lang_list[0]])
    for token in inventories[lang_list[1]]:
        if token in merged_inventory:
            merged_inventory[token] += inventories[lang_list[1]][token]

    logger.info(f'size of shared inventory: {len(merged_inventory)}')
    logger.info('Top 10 Tokens\n   \tToken\tCount')
    tokens = sorted(list(merged_inventory.keys()), key=lambda x: merged_inventory[x], reverse=True)
    for ti, tok in enumerate(tokens[:10]):
        print(f'{ti + 1:02d}. & {tok} & "\ipa{{{tokenizer.decode([tok])}}}" &  {merged_inventory[tok]:,d} \\\\')
