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
from transformers import logging as hf_logging

from utils import Arguments

hf_logging.set_verbosity_info()


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    kwargs = {'args': args, 'training_args': training_args}

    if args.finetune_impl == 'trainer':
        from finetune_trainer import FinetuneTrainer
        finetune = FinetuneTrainer(**kwargs)
    elif args.finetune_impl == 'ipex':
        from finetune_ipex import FinetuneIpex
        finetune = FinetuneIpex(**kwargs)
    elif args.finetune_impl == 'ipex_ccl':
        from finetune_ipex_dist import FinetuneIpexDist
        finetune = FinetuneIpexDist(**kwargs)
    elif args.finetune_impl == 'tpp':
        from finetune_tpp import FinetuneTpp
        finetune = FinetuneTpp(**kwargs)
    elif args.finetune_impl == 'tpp_ccl':
        from finetune_tpp_dist import FinetuneTppDist
        finetune = FinetuneTppDist(**kwargs)
    else:
        error_msg = f'Now only support trainer, ipex, ipex_ccl, tpp and tpp_ccl implementations ' \
                    f'for DLSA fine-tuning pipeline. ' \
                    f'Your input datatype is {args.finetune_impl}.'
        raise ValueError(error_msg)

    finetune.e2e_finetune()


if __name__ == '__main__':
    main()
