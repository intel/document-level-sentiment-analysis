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
export BATCH_SIZE=32
export SEQUENCE_LEN=55
export BF16=""
export USE_IPEX=""
export TRAIN_EPOCH=1
export MODEL_NAME_OR_PATH="bert-large-uncased"
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
    --bf16 )
        BF16="--bf16"
        echo "use bf16"
        ;;
    --use_ipex )
        USE_IPEX=1
        echo "use_ipex is : $USE_IPEX"
        ;;
    -h | --help )
         echo "Usage: ./fine-tuning/train_trainer.sh [OPTIONS]"
         echo "OPTION includes:"
         echo "   -l | --log_name - the log name of this round"
         echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
         echo "   -b | --batch_size - batch size per instance"
         echo "   -s | --sequence_len - max sequence length"
         echo "   --bf16 - whether using hf bf16 inference"
         echo "   --use_ipex - whether using ipex"
         echo "   -h | --help - displays this message"
         exit
      ;;
    * )
        echo "Invalid option: $1"
        echo "Usage: ./fine-tuning/train_trainer.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
        echo "   -b | --batch_size - batch size per instance"
        echo "   -s | --sequence_len - max sequence length"
        echo "   --bf16 - whether using hf bf16 inference"
        echo "   --use_ipex - whether using ipex"
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
python ./src/run_pt.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR/output_test \
        --max_seq_len $SEQUENCE_LEN \
	--num_train_epochs $TRAIN_EPOCH \
	--do_train \
	--per_device_train_batch_size $BATCH_SIZE \
        --do_predict \
        --per_device_eval_batch_size 8 \
	--no_cuda \
	$BF16 \
	$USE_IPEX \
	2>&1 | tee $OUTPUT_DIR/test_$i.log


