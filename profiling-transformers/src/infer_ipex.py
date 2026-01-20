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

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from infer import DlsaInference
from utils import compute_metrics, PredsLabels


class IpexInfer(DlsaInference):
    def _load_data(self):
        return super()._load_data()

    def _preprocess(self):
        return super()._preprocess()    

    def _load_model(self):
        with self.track('Load Model'):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path)

        if self.args.dtype_inf == 'fp32':
            self.model = ipex.optimize(self.model, dtype=torch.float32, level='O1')

        elif self.args.dtype_inf == 'bf16':
            with self.track("Process bf16 model"):
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16, level='O0')
                dumpy_tensor = torch.ones((self.training_args.per_device_eval_batch_size, self.max_seq_len),
                                          dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)
                with torch.cpu.amp.autocast(), torch.no_grad():
                    self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
                    self.model = torch.jit.freeze(self.model)
                with torch.no_grad():
                    y = self.model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = self.model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

        elif self.args.dtype_inf == 'int8':
            with self.track("Process int8 model"):
                # convert fp32 model to int 8
                dumpy_tensor = torch.ones((self.training_args.per_device_eval_batch_size, self.max_seq_len),
                                          dtype=torch.long)
                jit_inputs = (dumpy_tensor, dumpy_tensor, dumpy_tensor)

                if os.path.exists(self.args.model_name_or_path + "/quantized_model.pt"):
                    print("load int8 model-----------------------")
                    with torch.cpu.amp.autocast():
                        self.model = torch.jit.load(self.args.model_name_or_path + "/quantized_model.pt")
                        self.model = torch.jit.freeze(self.model.eval())
                else:
                    print("load configure and convert the model")
                    ipex.nn.utils._model_convert.replace_dropout_with_identity(self.model)
                    from intel_extension_for_pytorch.quantization import prepare, convert
                    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
                    qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                                  qscheme=torch.per_channel_symmetric))
                    prepared_model = prepare(self.model, qconfig, example_inputs=jit_inputs, inplace=False)
                    prepared_model.load_qconf_summary(
                        qconf_summary=self.args.model_name_or_path + "/int8_configure.json")
                    with torch.cpu.amp.autocast():
                        self.model = convert(prepared_model)
                        self.model = torch.jit.trace(self.model, jit_inputs, strict=False)
                    self.model = torch.jit.freeze(self.model)

                with torch.no_grad():
                    y = self.model(dumpy_tensor, dumpy_tensor, dumpy_tensor)
                    y = self.model(dumpy_tensor, dumpy_tensor, dumpy_tensor)

        else:
            error_msg = f'Now only support fp32, bf16 and int8.Your input datatype is {self.args.dtype_inf}.'
            raise ValueError(error_msg)

    def _do_infer(self):
        if self.training_args.do_predict:
            with self.track('Inference'):
                batch_size = self.training_args.per_device_eval_batch_size
                all_outputs, all_labels = [], []

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                def to_inputs(batch: dict) -> dict:
                    return {k: (v if torch.is_tensor(v) else tensor(v)).to(device=device) for k, v in batch.items()}

                def prediction_step(batch, labels):
                    all_labels.extend(labels)
                    inputs = to_inputs(batch)
                    output = self.model(**inputs)
                    all_outputs.append(output['logits'].detach().cpu())

                self.model.eval()

                with torch.no_grad():
                    if self.args.profiler:
                        with torch.profiler.profile(
                                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                    './profiler/' + self.args.profiler_name),
                                record_shapes=True,
                                profile_memory=True,
                                with_stack=True
                        ) as prof:
                            for batch in tqdm(DataLoader(self.test_data, batch_size=batch_size,
                                                         collate_fn=DataCollatorWithPadding(self.tokenizer))):
                                prediction_step(batch=batch, labels=batch.pop('labels'))
                                prof.step()
                    else:
                        for batch in tqdm(DataLoader(self.test_data, batch_size=batch_size,
                                                     collate_fn=DataCollatorWithPadding(self.tokenizer))):
                            prediction_step(batch=batch, labels=batch.pop('labels'))

                    acc = compute_metrics(PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels))
                    print(f"\n*********** TEST_METRICS ***********\nAccuracy: {acc['acc']}\n")
