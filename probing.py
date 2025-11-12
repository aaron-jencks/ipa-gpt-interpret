import logging
import os
from typing import Tuple, List
from typing_extensions import Self

from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
from transformers import GPT2TokenizerFast

from model import GPT


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess(
        cfg: dict, split: str,
        tokenizer: GPT2TokenizerFast, model_type: str,
        cpus: int = os.cpu_count()
) -> Dataset:
    dataset_name = cfg['datasets']['probing']

    logger.info(f'Loading dataset "{dataset_name}"')

    ds = load_dataset(dataset_name, split=split, cache_dir=cfg['datasets']['cache'])
    feat = 'text'
    if model_type == 'ipa':
        feat += '-phoneme'
    eval_feat = 'windows'
    if model_type == 'ipa':
        eval_feat += '-phoneme'

    def preprocess(examples):
        encoded = tokenizer(examples[feat])
        return {
            'encoding_length': [len(row) for row in encoded['input_ids']],
        }

    ds_pre = ds.map(preprocess, batched=True, num_proc=cpus)
    ds_pre = ds_pre.filter(lambda r: r['encoding_length'] <= 1024)

    # source: https://huggingface.co/docs/transformers/en/tasks/question_answering
    def preprocess(examples):
        inputs = tokenizer(
            examples[feat],
            max_length=1024,
            truncation=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        offset_mapping = inputs["offset_mapping"]
        answers = examples[eval_feat]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer['start']) == 0:
                logger.warning(f'row {i} does not have an answer')
                start_positions.append(-1)
                end_positions.append(-1)
                continue
            start_char = answer['start'][0]
            end_char = answer['end'][0]

            # Otherwise it's the start and end token positions
            idx = 0
            while idx <= len(inputs['input_ids']) and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = len(inputs['input_ids']) - 1
            while idx >= 0 and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return ds_pre.map(preprocess, batched=True, num_proc=cpus)


class ProbedGPT(nn.Module):
    def __init__(self, inner: GPT, phoneme_count: int):
        super().__init__()
        self.inner = inner
        self.probes = nn.ModuleList([
            nn.Linear(self.inner.config.n_embd, phoneme_count) for _ in range(inner.config.n_layer)
        ])
        self.freeze_inner_model()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.inner.eval()
        return self

    def freeze_inner_model(self):
        self.inner.eval()
        for param in self.inner.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.tensor, start_position: int, end_position: int):
        assert len(input_ids.shape) == 1, 'cannot handling batches for probing spans'
        l = input_ids.shape[0]
        assert l <= self.inner.config.block_size, f"Cannot forward sequence of length {l}, block size is only {self.inner.config.block_size}"
        device = next(self.parameters()).device
        pos = torch.arange(0, l, dtype=torch.long, device=device)  # shape (l)

        input_ids = input_ids.unsqueeze(0)  # add the batch dimension

        with torch.no_grad():
            tok_emb = self.inner.transformer.wte(input_ids)  # token embeddings of shape (B, l, n_embd)
            pos_emb = self.inner.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.inner.transformer.drop(tok_emb + pos_emb)

            block_outputs = []
            for block in self.inner.transformer.h:
                x = block(x)
                block_outputs.append(x)

        probe_outputs = []  # (n_layers, span_size, phoneme_count)

        for bi, block_output in enumerate(block_outputs):
            probe = self.probes[bi]
            span_length = end_position - start_position
            layer_outputs = [probe(block_output[0, start_position + i, :]) for i in range(span_length + 1)]
            probe_outputs.append(torch.stack(layer_outputs))

        return probe_outputs