# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

#

import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import numpy as np
from time import perf_counter_ns
from dataclasses import dataclass, field
import numpy as np
from contextlib import contextmanager
import os

SEC_TO_NS_SCALE = 1000000000

SPLIT_PATHS = {
    ('imdb', 'train'): './datasets/aclImdb/train',
    ('imdb', 'test'): './datasets/aclImdb/test',
    ('sst2', 'train'): './datasets/sst/train.tsv',
    ('sst2', 'test'): './datasets/sst/dev.tsv'
}


@dataclass
class Benchmark:
    summary_msg: str = field(default_factory=str)

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self, step):
        start = perf_counter_ns()
        yield
        ns = perf_counter_ns() - start
        msg = f"\n{'*' * 70}\n'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        print(msg)
        self.summary_msg += msg + '\n'

    def summary(self):
        print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    smoke_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to execute in sanity check mode."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of testing examples to this "
                    "value if set."
        },
    )
    instance_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "for multi-instance inference, to indicate which instance this is."
        },
    )
    dataset: Optional[str] = field(
        default='imdb',
        metadata={
            "help": "Select dataset ('imdb' / 'sst2'). Default is 'imdb'"
        },
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    profiler: int = field(
        default=0,
        metadata={
            "help": "wether using pytorch profiler"
        },
    )
    profiler_name: str = field(
        default="test",
        metadata={
            "help": "log name for pytorch profiler"
        },
    )
    ipex: bool = field(
        default=False,
        metadata={
            "help": "Use Intel® Extension for PyTorch for fine-Tuning."
        },
    )
    ipex_bf16: int = field(
        default=0,
        metadata={
            "help": "Auto mixed precision using bfloat16."
        },
    )
    ipex_fp32: int = field(
        default=0,
        metadata={
            "help": "Auto mixed precision using bfloat16."
        },
    )
    bf16_ipex_ft: int = field(
        default=False,
        metadata={
            "help": "Auto mixed precision using bfloat16 to fine-tuning."
        },
    )
    fp32_ipex_ft: int = field(
        default=False,
        metadata={
            "help": "use ipex optimization for fp32 fine-tuning."
        },
    )
    int8_bf16: int = field(
        default=0,
        metadata={
            "help": "Auto mixed precision using int8+bfloat16."
        },
    )
    multi_instance: bool = field(
        default=False,
        metadata={
            "help": "Whether to use multi-instance mode"
        },
    )
    int8: int = field(
        default=0,
        metadata={
            "help": "Whether to do inference with int8 model"
        },
    )
    dist_backend: Optional[str] = field(
        default="ccl", metadata={"help": "Distributed backend to use"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    real_time: bool = field(
        default=False, metadata={"help": "Whether to pre-process the inputs in real-time."}
    )
    few_shot: bool = field(
        default=False,
        metadata={
            "help": "Employ few-shot pattern-based MLM training on a small subset of the data."
        },
    )
    pattern_id: bool = field(
        default=0, metadata={"help": "Few-shot: pattern id of the pattern to use for few-shot training."}
    )
    label_loss: bool = field(
        default=True, metadata={"help": "Few-shot: whether to use label loss."}
    )
    random_mlm: bool = field(
        default=False, metadata={"help": "Few-shot: whether to use random MLM loss."}
    )
    alpha: float = field(
        default=0.6, metadata={"help": "Few-shot: alpha value for loss computation: ."}
    )
    torchscript: bool = field(
        default=False, metadata={"help": "Enable Torchscript."}
    )


class PredsLabels:
    def __init__(self, preds, labels):
        self.predictions = preds
        self.label_ids = labels


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def check_dataset(name: str):
    if name == 'imdb':
        return [name]
    elif name == 'sst2':
        return ['glue', 'sst2']
    else:
        error_msg = f'Now only imdb and sst2 dataset are supported. Your dataset is {name}.'
        raise ValueError(error_msg)


def read_dataset(name: str, split: str = "test", generator: bool = False,
                 return_labels: bool = True, batch_size: int = 1, max_samples: int = None):
    split_path = SPLIT_PATHS[(name, split)]
    args = split_path, return_labels, batch_size, max_samples
    gen = imdb_gen(*args) if name == 'imdb' else sst_gen(*args)

    if generator:
        return gen

    texts, labels = [], []
    for text_batch, label_batch in gen:
        texts.extend(text_batch)
        if return_labels:
            labels.extend(label_batch)
    return (texts, labels) if return_labels else texts


def imdb_gen(split_path, return_label, batch_size, max_samples):
    text_batch, label_batch = [], []
    for label_dir in "pos", "neg":
        for i, text_file in enumerate((Path(split_path) / label_dir).iterdir()):
            text_batch.append(text_file.read_text())
            if return_label:
                label_batch.append(0 if label_dir == 'neg' else 1)
            if len(text_batch) == batch_size:
                yield (text_batch, label_batch) if return_label else text_batch
                text_batch, label_batch = [], []
            if max_samples is not None and i == max_samples / 2:
                break
    if text_batch:
        yield (text_batch, label_batch) if return_label else text_batch
        text_batch, label_batch = [], []


def sst_gen(split_path, return_label, batch_size, max_samples):
    text_batch, label_batch = [], []
    i = 0
    with open(split_path) as f:
        for line in f.readlines()[1:]:
            if line:
                i += 1
                text, label = line.strip().split(" \t")
                text_batch.append(text)
                if return_label:
                    label_batch.append(int(label))
            if len(text_batch) == batch_size:
                yield (text_batch, label_batch) if return_label else text_batch
                text_batch, label_batch = [], []
            if max_samples is not None and i == max_samples:
                break
    if text_batch:
        yield (text_batch, label_batch) if return_label else text_batch
        text_batch, label_batch = [], []


def to_tensor_dataset(framework, encodings, labels=None):
    if framework == 'tf':
        from tensorflow.data import Dataset

        data = (dict(encodings), labels) if labels else dict(encodings)
        dataset = Dataset.from_tensor_slices(data)

    if framework == 'pt':
        from torch import tensor
        from torch.utils.data import Dataset

        class IMDbDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        dataset = IMDbDataset(encodings, labels)

    return dataset


def save_train_metrics(train_result, trainer, max_train):
    # pytorch only
    if train_result:
        metrics = train_result.metrics
        metrics["train_samples"] = max_train
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def save_test_metrics(metrics, max_test, output_dir):
    metrics['test_samples'] = max_test
    with open(Path(output_dir) / 'test_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return "\n\n******** TEST METRICS ********\n" + '\n'.join(f'{k}: {v}' for k, v in metrics.items())


def read_imdb_split(split_dir):
    texts, labels = [], []
    for label_dir in "pos", "neg":
        for text_file in (Path(split_dir) / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == 'neg' else 1)
    return texts, labels


def read_sst_file(sst_file):
    texts, labels = [], []
    with open(sst_file) as f:
        for line in f.readlines()[1:]:
            if line:
                text, label = line.strip().split(" \t")
                texts.append(text)
                labels.append(int(label))
    return texts, labels
