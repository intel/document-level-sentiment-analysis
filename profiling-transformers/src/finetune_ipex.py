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


import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from finetune import DlsaFinetune
from utils import compute_metrics, PredsLabels



class FinetuneIpex(DlsaFinetune):
    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()

    def _load_model(self):
        if self.training_args.do_train:
            with self.track('Load Model'):
                if self.args.dtype_ft == "fp32":
                    self.model = AutoModelForSequenceClassification \
                        .from_pretrained(self.args.model_name_or_path)
                    self.model = ipex.optimize(self.model, dtype=torch.float32, level='O1')

                elif self.args.dtype_ft == "bf16":
                    with torch.cpu.amp.autocast():
                        self.model = AutoModelForSequenceClassification \
                            .from_pretrained(self.args.model_name_or_path)
                        self.model = ipex.optimize(self.model, dtype=torch.bfloat16, level='O0')
                else:
                    error_msg = f'Now only support fp32, bf16.Your input datatype is {self.args.dtype_ft}.'
                    raise ValueError(error_msg)

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
                    optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

                with self.track('--------Training Loop'):
                    for _ in tqdm(range(int(self.training_args.num_train_epochs)), desc='Epoch'):
                        for batch in tqdm(DataLoader(self.train_data, batch_size=batch_size, shuffle=True,
                                                     collate_fn=DataCollatorWithPadding(self.tokenizer)),
                                          desc='Train Step'):
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
                    for batch in tqdm(DataLoader(self.test_data, batch_size=batch_size,
                                                 collate_fn=DataCollatorWithPadding(self.tokenizer)), desc='Test Step'):
                        prediction_step(batch=batch, labels=batch.pop('labels'))
                    acc = compute_metrics(PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels))
                    print(f"\n*********** TEST_METRICS ***********\nAccuracy: {acc['acc']}\n")
