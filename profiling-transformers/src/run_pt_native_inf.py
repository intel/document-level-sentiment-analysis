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

import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import intel_extension_for_pytorch as ipex
finally:
    pass

from transformers import (
    logging as hf_logging,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding
)

from utils import (
    Arguments,
    Benchmark,
    compute_metrics,
    PredsLabels,
    check_dataset
)

hf_logging.set_verbosity_info()
logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_train, max_test = args.max_train_samples, args.max_test_samples
    if args.smoke_test:
        training_args.max_steps = 3
        max_train, max_test = 10, 10

    bench = Benchmark()
    track = bench.track
    with track('Total Run'):
        ############################ Load Data ####################################
        with track('Load Data'):
            data = load_dataset(*check_dataset(args.dataset))
            train_all = data['train']
            test_split = 'validation' if args.dataset == 'sst2' else 'test'
            len_train = len(train_all)
            train_data = train_all.select(range(len_train - max_train, len_train)) if max_train else train_all

            # split the Test Data for multi-instance
            if args.multi_instance:
                start_index = (args.instance_index - 1) * args.max_test_samples
                end_index = args.instance_index * args.max_test_samples
                test_data = data[test_split].select(range(start_index, end_index))
                print("start_index is ", start_index)
                print("end_index is ", end_index)
                print("test length is ", len(test_data))
            else:
                test_data = data[test_split].select(range(max_test)) if max_test else data[test_split]

            text_column = [c for c in test_data.column_names if type(test_data[c][0]) != int][0]

        ############################### Pre-process ###############################
        with track('Pre-process'):
            with track('----Init tokenizer'):
                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
                )

            max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)

            with track('----Tokenize + Extract Features'):
                def preprocess(examples):
                    return tokenizer(
                        examples[text_column],
                        padding='max_length',
                        truncation=True,
                        max_length=max_seq_len
                    )

                kwargs = dict(
                    function=preprocess,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=[text_column] + (['idx'] if args.dataset == 'sst2' else []),
                    load_from_cache_file=not args.overwrite_cache)

                train_data = train_data.map(**kwargs) if training_args.do_train else None
                test_data = test_data.map(**kwargs) if training_args.do_predict else None

        ###################### Load Model and Trainer ############################
        with track('Load Model'):
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(device=device)

        with track("Process int8 model"):
            if args.int8:
                # convert fp32 model to int 8
                dumpy_tensor = torch.ones((training_args.per_device_eval_batch_size, max_seq_len), dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)

                if os.path.exists(args.model_name_or_path + "/quantized_model.pt"):
                    print("load int8 model-----------------------")
                    if args.int8_bf16:
                        with torch.cpu.amp.autocast():
                            model = torch.jit.load(args.model_name_or_path + "/quantized_model.pt")
                            model = torch.jit.freeze(model.eval())
                    else:
                        model = torch.jit.load(args.model_name_or_path + "/quantized_model.pt")
                        model = torch.jit.freeze(model.eval())
                else:
                    print("load configure and convert the model")
                    ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
                    from intel_extension_for_pytorch.quantization import prepare, convert
                    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
                    qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                    prepared_model = prepare(model, qconfig, example_inputs=jit_inputs, inplace=False)
                    prepared_model.load_qconf_summary(qconf_summary = args.model_name_or_path + "/int8_configure.json")
                    if args.int8_bf16:
                        with torch.cpu.amp.autocast():
                            model = convert(prepared_model)
                            model = torch.jit.trace(model, jit_inputs, strict=False)
                    else:
                        model = convert(prepared_model)
                        model = torch.jit.trace(model, jit_inputs, strict=False)
                    model = torch.jit.freeze(model)


                with torch.no_grad():
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

    #            model.save("quantized_model.pt")
    #            import sys
    #            sys.exit(0)

        with track("Process bf16 model"):
            if args.ipex_bf16:
                model = ipex.optimize(model, dtype=torch.bfloat16, level='O0')
                dumpy_tensor = torch.ones((training_args.per_device_eval_batch_size, max_seq_len), dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)
                with torch.cpu.amp.autocast(), torch.no_grad():
                    model = torch.jit.trace(model, jit_inputs, strict=False)
                    model = torch.jit.freeze(model)
                with torch.no_grad():
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

        if args.ipex_fp32:
            model = ipex.optimize(model, dtype=torch.float32, level='O1')

        ############################### Inference #################################
        if training_args.do_predict:
            with track('Inference'):
                batch_size = training_args.per_device_eval_batch_size
                all_outputs, all_labels = [], []

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                def to_inputs(batch: dict) -> dict:
                    return {k: (v if torch.is_tensor(v) else tensor(v)).to(device=device) for k, v in batch.items()}

                def prediction_step(batch, labels):
                    all_labels.extend(labels)
                    inputs = to_inputs(batch)
                    output = model(inputs['input_ids'], inputs['attention_mask']) if args.torchscript \
                        else model(**inputs)
                    all_outputs.append(output['logits'].detach().cpu())

                model.eval()

                with torch.no_grad():
                    if args.profiler:
                        with torch.profiler.profile(
                            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler/' + args.profiler_name),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True
                            ) as prof:
                            for batch in tqdm(DataLoader(test_data, batch_size=batch_size,
                                                         collate_fn=DataCollatorWithPadding(tokenizer))):
                                prediction_step(batch=batch, labels=batch.pop('labels'))
                                prof.step()
                    else:
                        for batch in tqdm(DataLoader(test_data, batch_size=batch_size,
                                                     collate_fn=DataCollatorWithPadding(tokenizer))):
                            prediction_step(batch=batch, labels=batch.pop('labels'))

                    acc = compute_metrics(PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels))
                    print(f"\n*********** TEST_METRICS ***********\nAccuracy: {acc['acc']}\n")

    bench.summary()


if __name__ == "__main__":
    main()
