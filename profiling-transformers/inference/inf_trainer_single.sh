#!/bin/bash

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

LOG_NAME=$(date "+%m%d-%H%M")
DATASET="sst2"
BATCH_SIZE=8
SEQUENCE_LEN=55
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-bert-large-uncased}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs}"
DTYPE_INF="fp32"
APPEND=""

while [ "$1" != "" ];
do
   case $1 in
    -l | --log_name )
        shift
        LOG_NAME="$1"
        echo "log name is $LOG_NAME"
        ;;
    -d | --dataset )
        shift
        DATASET="$1"
        echo "dataset is : $DATASET"
        ;;
    -b | --batch_size )
        shift
        BATCH_SIZE="$1"
        echo "batch size per instance is : $BATCH_SIZE"
        ;;
    -s | --sequence_len )
        shift
        SEQUENCE_LEN="$1"
        echo "sequence_len is : $SEQUENCE_LEN"
        ;;
    --dtype_inf )
        shift
        DTYPE_INF="$1"
        echo "dtype_inf is : $DTYPE_INF"
        ;;
    -h | --help )
         echo "Usage: $0 [OPTIONS]"
         echo "OPTION includes:"
         echo "   -l | --log_name - the log name of this round"
         echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
         echo "   -b | --batch_size - batch size per instance"
         echo "   -s | --sequence_len - max sequence length"
         echo "   --dtype_inf - data type used for inference"
         echo "   -h | --help - displays this message"
         exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: $0 [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] whether to use imdb or sst2 DATASET"
        echo "   -b | --batch_size - batch size per instance"
        echo "   -s | --sequence_len - max sequence length"
        echo "   --dtype_inf - data type used for inference"
        exit
       ;;
  esac
  shift
done

if [ "$DTYPE_INF" == "bf16" ]; then
    APPEND="--bf16 --use_ipex"
fi

if [ -z "$LOG_NAME" ]; then
    pre=$(date "+%m%d-%H%M")
else
    pre=$LOG_NAME
fi

OUTPUT_DIR=$OUTPUT_DIR'/'$pre'/'$DATASET
echo "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"/output_test


export CUDA_VISIBLE_DEVICES="-1"; \
python ./src/run_infer.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR"/output_test \
        --infer_impl trainer \
        --do_predict \
        --max_seq_len "$SEQUENCE_LEN" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --no_cuda \
        "$APPEND" \
        2>&1 | tee "$OUTPUT_DIR"/test.log \

