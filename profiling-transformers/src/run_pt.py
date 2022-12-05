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

from datasets import load_dataset
from transformers import (
    logging as hf_logging,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils import (
    Arguments,
    Benchmark,
    compute_metrics,
    save_train_metrics,
    save_test_metrics,
    check_dataset
)

hf_logging.set_verbosity_info()
logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

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
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

            trainer = Trainer(
                model=model,  # the instantiated HF model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=train_data,  # training dataset
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=tokenizer
            )

        ############################### Fine-Tune #################################
        if training_args.do_train:
            with track('Fine-Tune'):
                train_result = trainer.train()
                trainer.save_model()
                save_train_metrics(train_result, trainer, len(train_data))

        ############################### Inference #################################
        test_metrics = ""
        if training_args.do_predict:
            with track('Inference'):
                preds, _, metrics = trainer.predict(test_data)
                test_metrics = save_test_metrics(metrics, len(test_data), training_args.output_dir)

    bench.summary()
    print(test_metrics)


if __name__ == "__main__":
    main()
