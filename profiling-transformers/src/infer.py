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


from datasets import load_dataset
from transformers import AutoTokenizer

from utils import Benchmark



class DlsaInference(object):
    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.training_args = kwargs['training_args']

        self.max_train, self.max_test = self.args.max_train_samples, self.args.max_test_samples
        if self.args.smoke_test:
            self.max_train, self.max_test = 100, 100

        self.bench = Benchmark()
        self.track = self.bench.track

    def e2e_infer(self):
        with self.track('Total Run'):
            self._load_data()
            self._preprocess()
            self._load_model()
            self._do_infer()
            self.bench.summary()

    def _load_data(self):
        with self.track('Load Data'):
            data = load_dataset(self.args.dataset)
            test_split = 'validation' if self.args.dataset == 'sst2' else 'test'
            if self.args.multi_instance:
                start_index = (self.args.instance_index - 1) * self.args.max_test_samples
                end_index = self.args.instance_index * self.args.max_test_samples
                self.test_data = data[test_split].select(range(start_index, end_index))
                print("start_index is ", start_index)
                print("end_index is ", end_index)
                print("test length is ", len(self.test_data))
            else:
                self.test_data = data[test_split].select(range(self.max_test)) if self.max_test else data[test_split]

            self.text_column = [c for c in self.test_data.column_names if type(self.test_data[c][0]) != int][0]

    def _preprocess(self):
        with self.track('Pre-process'):
            with self.track('----Init tokenizer'):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path
                )

            self.max_seq_len = min(self.args.max_seq_len, self.tokenizer.model_max_length)

            with self.track('----Tokenize + Extract Features'):
                def preprocess(examples):
                    return self.tokenizer(
                        examples[self.text_column],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_seq_len
                    )

                kwargs = dict(
                    function=preprocess,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=[self.text_column] + (['idx'] if self.args.dataset == 'sst2' else []),
                    load_from_cache_file=not self.args.overwrite_cache)

                self.test_data = self.test_data.map(**kwargs) if self.training_args.do_predict else None

    def _load_model(self):
        raise NotImplementedError

    def _do_infer(self):
        raise NotImplementedError