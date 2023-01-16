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

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tpp_pytorch_extension import bert as tpp_bert
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    set_seed
)

from finetune_tpp import FinetuneTpp
from utils import compute_metrics, PredsLabels



class FinetuneTppDist(FinetuneTpp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_seed(self.training_args.seed)

        if int(os.environ.get('PMI_SIZE', '0')) > 1 and not self.args.multi_instance:
            if self.args.dist_backend == 'ccl':
                try:
                    import oneccl_bindings_for_pytorch
                except ImportError:
                    print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
                    raise
            elif self.args.dist_backend == 'mpi':
                if not torch.distributed.is_mpi_available():
                    try:
                        import torch_mpi
                    except ImportError:
                        print("MPI backend requested but not available try installing torch_mpi module")
                        raise
            else:
                raise ValueError(f"{self.args.dist_backend} backend requested but not supported")

            os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')
            torch.distributed.init_process_group(backend=self.args.dist_backend)
            self.training_args.local_rank = torch.distributed.get_rank()
            if self.training_args.local_rank == 0:
                print(
                    f"##################Using {self.args.dist_backend.upper()} dist "
                    f"run with {torch.distributed.get_world_size()} ranks",
                    flush=True)

    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()

    def _load_model(self):
        return super()._load_model()

    def _do_finetune(self):
        if self.training_args.do_train:
            with self.track('Fine-Tune'):
                with self.track('--------Init Fine-Tuning'):
                    batch_size = self.training_args.per_device_train_batch_size
                    self.model.train()
                    weight_decay = 0.0
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in self.model.named_parameters() if
                                       not any(nd in n for nd in no_decay)],
                            "weight_decay": weight_decay,
                        },
                        {
                            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                        },
                    ]
                    optim = tpp_bert.AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)
                    if self.training_args.local_rank != -1:
                        self.model = torch.nn.parallel.DistributedDataParallel(self.model)

                with self.track('--------Training Loop'):
                    train_sampler = RandomSampler(
                        self.train_data) if self.training_args.local_rank == -1 else DistributedSampler(
                        self.train_data)

                    for _ in tqdm(range(int(self.training_args.num_train_epochs)), desc='Epoch',
                                  disable=self.training_args.local_rank not in [-1, 0]):
                        for batch in tqdm(DataLoader(self.train_data, sampler=train_sampler, batch_size=batch_size,
                                                     collate_fn=DataCollatorWithPadding(self.tokenizer)),
                                          desc='Train Step', disable=self.training_args.local_rank not in [-1, 0]):
                            optim.zero_grad()
                            loss = self.model(**batch)[0]
                            loss.backward()
                            optim.step()

                with self.track('--------Save Fine-Tuned Model'):
                    torch.save(self.model.state_dict(), self.training_args.output_dir + "/pytorch_model.bin")

    def _do_infer(self):
        if self.training_args.do_predict:
            with self.track('Inference'):
                batch_size = self.training_args.per_device_eval_batch_size
                all_outputs, all_labels = [], []

                def prediction_step(batch, labels):
                    all_labels.extend(labels)
                    inputs = batch
                    output = self.model(**inputs)
                    all_outputs.append(output['logits'].detach().cpu())

                self.model.eval()
                with torch.no_grad():
                    test_sampler = RandomSampler(
                        self.test_data) if self.training_args.local_rank == -1 else DistributedSampler(
                        self.test_data)

                    for batch in tqdm(DataLoader(self.test_data, sampler=test_sampler, batch_size=batch_size,
                                                 collate_fn=DataCollatorWithPadding(self.tokenizer)),
                                      desc='Test Step'):
                        prediction_step(batch=batch, labels=batch.pop('labels'))
                    acc = compute_metrics(PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels))
                    print(f"\n*********** TEST_METRICS ***********\nAccuracy: {acc['acc']}\n")
