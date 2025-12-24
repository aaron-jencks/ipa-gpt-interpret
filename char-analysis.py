import argparse
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import pathlib
from queue import Full
from typing import Dict, Set
from collections import Counter, defaultdict

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLOBAL_DATASET = None


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    feature: str
    lang_feature: str
    punctuation: Set[str]
    orthographies: Dict[str, Set[str]]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_char_set(path: pathlib.Path) -> Set[str]:
    # raw character stream — every character counts
    with open(path, "r", encoding="utf-8") as fp:
        return set(fp.read())


def csv_escape_char(ch: str) -> str:
    # make control chars visible + escape quotes for pandas
    if ch == "\n":
        ch = "\\n"
    elif ch == "\t":
        ch = "\\t"
    elif ch == "\r":
        ch = "\\r"
    return ch.replace('"', '""')


# ─────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────

def counting_daemon(
    qin: mp.Queue,
    qdone: mp.Queue,
    qout: mp.Queue,
    cfg: Config,
):
    # local accumulation ONLY (fast)
    local_counts = defaultdict(Counter)

    while True:
        sl = qin.get()
        if sl is None:
            break

        slice_start, slice_end = sl
        records = GLOBAL_DATASET[slice_start:slice_end]
        texts = records[cfg.feature]
        langs = records[cfg.lang_feature]

        for text, lang in zip(texts, langs):
            if lang not in cfg.orthographies:
                continue

            valid_chars = cfg.orthographies[lang] | cfg.punctuation
            ctr = local_counts[lang]

            for ch in text:
                if ch in valid_chars:
                    ctr[ch] += 1

        # signal "one batch finished"
        qdone.put(1)

    # send final local counts back ONCE
    qout.put(dict(local_counts))


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

def log_inventories(
    directory: pathlib.Path,
    supports: Dict[str, Counter],
    disjoint: Dict[str, Set[str]],
    shared: Set[str],
):
    logger.info(f"saving character inventories to {directory}")
    os.makedirs(directory, exist_ok=True)

    # per-language disjoint inventories
    for lang in disjoint.keys():
        lines = ['"char","support"']
        for ch in sorted(
            disjoint[lang],
            key=lambda c: supports[lang][c],
            reverse=True,
        ):
            esc = csv_escape_char(ch)
            lines.append(f'"{esc}",{supports[lang][ch]}')

        with open(directory / f"{lang}.csv", "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))

    # shared inventory
    header = '"char","support",' + ",".join(f"{l}_support" for l in disjoint.keys())
    shared_lines = [header]

    for ch in sorted(
        shared,
        key=lambda c: sum(supports[l][c] for l in disjoint.keys()),
        reverse=True,
    ):
        esc = csv_escape_char(ch)
        per_lang = [supports[l][ch] for l in disjoint.keys()]
        shared_lines.append(
            f'"{esc}",{sum(per_lang)},' + ",".join(map(str, per_lang))
        )

    with open(directory / "shared.csv", "w", encoding="utf-8") as fp:
        fp.write("\n".join(shared_lines))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Character-level inventory analysis (correct multiprocessing + progress)"
    )

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--feature", type=str, default="text")
    ap.add_argument("--lang-feature", type=str, default="language")
    ap.add_argument("--cpus", type=int, default=os.cpu_count())
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("./cache/huggingface"))
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--result-directory", type=pathlib.Path, required=True)

    ap.add_argument("--punctuation-file", type=pathlib.Path, required=True)
    ap.add_argument(
        "--orthography",
        action="append",
        required=True,
        help="language=path/to/orthography.txt",
    )

    args = ap.parse_args()

    punctuation = load_char_set(args.punctuation_file)

    orthographies = {}
    for spec in args.orthography:
        lang, path = spec.split("=", 1)
        orthographies[lang] = load_char_set(pathlib.Path(path))

    logger.info("loading dataset...")
    dataset = load_dataset(args.dataset, cache_dir=args.cache, num_proc=args.cpus)
    GLOBAL_DATASET = concatenate_datasets([dataset[s] for s in dataset])

    languages = set(orthographies.keys())

    cfg = Config(
        feature=args.feature,
        lang_feature=args.lang_feature,
        punctuation=punctuation,
        orthographies=orthographies,
    )

    queues = [mp.Queue() for _ in range(args.cpus)]
    qdone = mp.Queue()  # per-batch completion signals
    qout = mp.Queue()   # final per-worker counters
    procs = []

    # spawn workers
    for i in range(args.cpus):
        p = mp.Process(
            target=counting_daemon,
            args=(queues[i], qdone, qout, cfg),
        )
        p.start()
        procs.append(p)

    # feed work + count batches
    total_batches = 0
    qi = 0
    for idx in tqdm(range(0, len(GLOBAL_DATASET), args.batch_size), desc="queueing batches"):
        end = min(idx + args.batch_size, len(GLOBAL_DATASET))
        total_batches += 1
        while True:
            try:
                queues[qi].put_nowait((idx, end))
                qi = (qi + 1) % len(queues)
                break
            except Full:
                qi = (qi + 1) % len(queues)

    # stop workers
    for q in queues:
        q.put(None)

    # progress: wait for every batch to be completed
    for _ in tqdm(range(total_batches), desc="processing batches"):
        qdone.get()

    # merge results from workers (exactly one message per worker)
    counts = {lang: Counter() for lang in languages}
    for _ in range(len(procs)):
        worker_counts = qout.get()
        for lang, ctr in worker_counts.items():
            counts[lang].update(ctr)

    for p in procs:
        p.join()

    # inventory analysis
    shared_inventory = None
    for lang in languages:
        inv = set(counts[lang].keys())
        shared_inventory = inv if shared_inventory is None else (shared_inventory & inv)

    disjoint = {
        lang: set(counts[lang].keys()) - shared_inventory
        for lang in languages
    }

    log_inventories(
        args.result_directory,
        counts,
        disjoint,
        shared_inventory,
    )
