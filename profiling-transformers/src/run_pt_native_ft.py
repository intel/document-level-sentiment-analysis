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

from pathlib import Path
import os
import logging
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch import tensor

try:
    import intel_extension_for_pytorch as ipex
finally:
    pass

import transformers
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    set_seed,
)

from utils import (
    Arguments,
    read_dataset,
    to_tensor_dataset,
    Benchmark,
    compute_metrics,
    PredsLabels
)

transformers.logging.set_verbosity_info()

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    output_dir = Path(training_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    bench = Benchmark()
    track = bench.track

    set_seed(training_args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if int(os.environ.get('PMI_SIZE', '0')) > 1 and not args.multi_instance:
        if args.dist_backend == 'ccl':
            try:
                import oneccl_bindings_for_pytorch
            except:
                print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
                raise
        elif args.dist_backend == 'mpi':
            if not torch.distributed.is_mpi_available():
                try:
                    import torch_mpi
                except:
                    print("MPI backend requested but not available try installing torch_mpi module")
                    raise
        else:
            raise ValueError(f"{args.dist_backend} backend requested but not supported")

        os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
        os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')
        torch.distributed.init_process_group(backend=args.dist_backend)
        device = torch.device("cpu")
        training_args.local_rank = torch.distributed.get_rank()
        if training_args.local_rank == 0: print(f"##################Using {args.dist_backend.upper()} dist run with {torch.distributed.get_world_size()} ranks", flush=True)

    def to_inputs(batch: dict) -> dict:
        return {k: (v if torch.is_tensor(v) else tensor(v)).to(device=device) \
                for k, v in batch.items()}

    ################################# Load Data #################################

    with track('Load Data'):
        if training_args.do_train:
            # Train Data
            train_texts, train_labels = read_dataset(args.dataset, 'train')
            max_train = args.max_train_samples if args.max_train_samples else len(train_texts)
            if args.smoke_test:
                training_args.max_steps = 3
                training_args.num_train_epochs = 1
                max_train = 104
            train_texts, train_labels = train_texts[:max_train], train_labels[:max_train]

        if training_args.do_predict:
            max_test = 100 if args.smoke_test else (args.max_test_samples if args.max_test_samples else None)

            if not args.real_time:
                # Test Data
                test_texts, test_labels = read_dataset(args.dataset, 'test')
                if args.multi_instance:
                    start_index = (args.instance_index - 1) * args.max_test_samples
                    end_index = args.instance_index * args.max_test_samples
                    test_texts, test_labels = test_texts[start_index:end_index], test_labels[start_index:end_index]
                    print("start_index is ", start_index)
                    print("end_index is ", end_index)
                    print("test text length is ", len(test_texts))
                    print("test labels  length is ", len(test_labels))
                else:
                    test_texts, test_labels = test_texts[:max_test], test_labels[:max_test]

    ################################# Pre-process #################################
    with track('Pre-process'):
        with track('----Init tokenizer'):
            # Tokenization + Feature Extraction
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
            )
            max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
            token_args = dict(truncation=True, padding=True, max_length=max_seq_len)

            if training_args.do_train:
                with track('----Training data encoding'):
                    train_encodings = tokenizer(train_texts, **token_args)
                with track('----Training tensor data convert'):
                    train_dataset = to_tensor_dataset('pt', train_encodings, train_labels)

            if training_args.do_predict and not args.real_time:
                with track('----PyTorch test data encoding'):
                    test_encodings = tokenizer(test_texts, padding='max_length', max_length=max_seq_len,
                                               truncation=True)
                with track('----PyTorch test tensor data convert'):
                    test_dataset = to_tensor_dataset('pt', test_encodings, test_labels)

    ################################# Load Model #################################
    if training_args.do_train or not args.torchscript:
        with track('Load Model'):
            if args.bf16_ipex_ft:
                with torch.cpu.amp.autocast():
                    model = AutoModelForSequenceClassification \
                            .from_pretrained(args.model_name_or_path) \
                            .to(device=device)
                    model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
            else:
                model = AutoModelForSequenceClassification \
                        .from_pretrained(args.model_name_or_path) \
                        .to(device=device)    
            #model = AutoModelForSequenceClassification \
            #    .from_pretrained(args.model_name_or_path) \
            #    .to(device=device)

        with track("Process int8 model"):
            if args.int8:
                # convert fp32 model to int8
                ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
                conf = ipex.quantization.QuantConf(configure_file=args.model_name_or_path + "/configure.json")
                dumpy_tensor = torch.ones((training_args.per_device_eval_batch_size, max_seq_len), dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)
                if args.int8_bf16:
                    with torch.cpu.amp.autocast():
                        model = ipex.quantization.convert(model, conf, jit_inputs)
                else:
                    model = ipex.quantization.convert(model, conf, jit_inputs)
                with torch.no_grad():
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

        with track("Process bf16 model"):
            if args.ipex_bf16:
                # convert fp32 model to bf16
                with torch.cpu.amp.autocast(), torch.no_grad():
                    torch.jit.load('imdb_bf16model.pt')
                model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
                dumpy_tensor = torch.ones((training_args.per_device_eval_batch_size, max_seq_len), dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)
                with torch.cpu.amp.autocast(), torch.no_grad():
                    model = torch.jit.trace(model, jit_inputs, strict=False)
                    model = torch.jit.freeze(model)
                with torch.no_grad():
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

    ################################ Fine-Tune #################################
    if training_args.do_train:
        with track('Fine-Tune'):
            with track('--------Init Fine-Tuning'):
                batch_size = training_args.per_device_train_batch_size
                model.train()
                weight_decay = 0.0
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
                optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
                if training_args.local_rank != -1:
                   model = torch.nn.parallel.DistributedDataParallel(model)

            with track('--------Training Loop'):
                train_sampler = RandomSampler(train_dataset) if training_args.local_rank == -1 else DistributedSampler(train_dataset)

                for _ in tqdm(range(int(training_args.num_train_epochs)), desc='Epoch', disable=training_args.local_rank not in [-1, 0]):
                    for batch in tqdm(DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size),
                                      desc='Train Step', disable=training_args.local_rank not in [-1, 0]):
                        optim.zero_grad()
                        loss = model(**to_inputs(batch))[0]
                        loss.backward()
                        optim.step()

            with track('--------Save Fine-Tuned Model'):
                if training_args.local_rank in [-1, 0]:
                    # Take care of DDP wrapper
                    model_to_save = model.module if hasattr(model, "module") else model
                    if args.torchscript:
                        with track('--------Save TorchScript model'):
                            model.eval()
                            batch = to_inputs(batch)
                            traced_model = torch.jit.trace(model_to_save, [batch['input_ids'], batch['attention_mask']])
                            torch.jit.save(traced_model, output_dir / "traced_model.pt")
                    else:
                        torch.save(model_to_save.state_dict(), output_dir / "pytorch_model.bin")

    ############################### Inference #################################
    if training_args.do_predict:
        with track('Inference'):
            if args.torchscript:
                with track('--------Load TorchScript model'):
                    model_path = output_dir if training_args.do_train else Path(args.model_name_or_path)
                    model = torch.jit.load(model_path / "traced_model.pt").to(device=device)

            batch_size = training_args.per_device_eval_batch_size
            all_outputs, all_labels = [], []

            def prediction_step(batch, labels):
                all_labels.extend(labels)
                inputs = to_inputs(batch)
                output = model(inputs['input_ids'], inputs['attention_mask']) if args.torchscript \
                    else model(**inputs)
                all_outputs.append(output['logits'].detach().cpu())

            model.eval()
            with torch.no_grad():
                if args.real_time:
                    data_generator = read_dataset(args.dataset, 'test', generator=True, \
                                                  batch_size=batch_size, max_samples=max_test)

                    for texts, labels in tqdm(data_generator, desc='Test Step'):
                        prediction_step(batch=tokenizer(texts, **token_args), labels=labels)

                else:
                    test_sampler = RandomSampler(test_dataset) if training_args.local_rank == -1 else DistributedSampler(test_dataset)

                    for batch in tqdm(DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size), desc='Test Step'):
                        prediction_step(batch=batch, labels=batch.pop('labels'))
                    acc = compute_metrics(PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels))
                    print(f"\n*********** TEST_METRICS ***********\nAccuracy: {acc['acc']}\n")

    bench.summary()


if __name__ == "__main__":
    main()
