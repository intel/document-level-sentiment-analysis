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

export LOG_NAME=`date "+%m%d-%H%M"`
export DATASET="sst2"
export BATCH_SIZE=8
export SEQUENCE_LEN=55
export IPEX_BF16=0
export IPEX_FP32=0
export INT8=0
export INT8_BF16=0
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-bert-large-uncased}"
export OUTPUT_DIR="${OUTPUT_DIR:-./logs}"

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
    --ipex_bf16 )
        IPEX_BF16=1
        echo "ipex_bf16 is : $IPEX_BF16"
        ;;
    --ipex_fp32 )
        IPEX_FP32=1
        echo "ipex_fp32 is : $IPEX_FP32"
        ;;
    --int8 )
        INT8=1
        echo "int8 is : $INT8"
        ;;
    --int8_bf16 )
        INT8_BF16=1
        echo "int8_bf16 is : $INT8_BF16"
        ;;
    -h | --help )
         echo "Usage: ././inference/single_instance.sh [OPTIONS]"
         echo "OPTION includes:"
         echo "   -l | --log_name - the log name of this round"
         echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
         echo "   -b | --batch_size - batch size per instance"
         echo "   -s | --sequence_len - max sequence length"
         echo "   --ipex_bf16 - wether to use ipex_bf16 precision"
         echo "   --ipex_fp32 - wether to use ipex_fp32 precision"
         echo "   --int8 - wether to use int8 precision"
         echo "   --int8_bf16 - wether to use int8_bf16 precision"
         echo "   -h | --help - displays this message"
         exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: ./inference/single_instance.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
        echo "   -b | --batch_size - batch size per instance"
        echo "   -s | --sequence_len - max sequence length"
        echo "   --ipex_bf16 - wether to use ipex_bf16 precision"
        echo "   --ipex_fp32 - wether to use ipex_fp32 precision"
        echo "   --int8 - wether to use int8 precision"
        echo "   --int8_bf16 - wether to use int8_bf16 precision"
        exit
       ;;
  esac
  shift
done

if [ -z "$LOG_NAME" ]; then
    pre=`date "+%m%d-%H%M"`
else
    pre=$LOG_NAME
fi

OUTPUT_DIR=$OUTPUT_DIR'/'$pre'/'$DATASET
echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR


export CUDA_VISIBLE_DEVICES="-1"; \
python ./src/run_pt_native_inf.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --int8 $INT8 \
        --int8_bf16 $INT8_BF16 \
        --ipex_bf16 $IPEX_BF16 \
        --ipex_fp32 $IPEX_FP32 \
        --output_dir $OUTPUT_DIR/output_test \
        --do_predict \
        --max_seq_len $SEQUENCE_LEN \
        --per_device_eval_batch_size $BATCH_SIZE \
	2>&1 | tee $OUTPUT_DIR/test_$i.log


