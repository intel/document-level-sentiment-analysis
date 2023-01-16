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


from transformers import HfArgumentParser, TrainingArguments
from utils import Arguments
from transformers import logging as hf_logging

hf_logging.set_verbosity_info()


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    kwargs = {'args': args, 'training_args': training_args}

    if args.infer_impl == 'trainer':
        from infer_trainer import TrainerInfer
        infer = TrainerInfer(**kwargs)
    elif args.infer_impl == 'ipex':
        from infer_ipex import IpexInfer
        infer = IpexInfer(**kwargs)
    else:
        error_msg = f'Now only support trainer and ipex implementation for DLSA inference pipeline.'
        raise ValueError(error_msg)
    
    infer.e2e_infer()


if __name__ == '__main__':
    main()
