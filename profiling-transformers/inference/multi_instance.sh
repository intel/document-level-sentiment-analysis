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

#
export KMP_SETTINGS=1
export KMP_BLOCKTIME=1
export OMP_MAX_ACTIVE_LEVELS=1

export LOG_NAME=`date "+%m%d-%H%M"`
export DATASET="sst2"
export NUMBER_OF_INSTANCE_PER_SOCKET=1
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
    -n | --num_of_ins_per_socket )
        shift
        NUMBER_OF_INSTANCE_PER_SOCKET="$1"
        echo "number_of_instance_per_socket is : $NUMBER_OF_INSTANCE_PER_SOCKET"
        ;;
#    -c | --cores_per_instance )
#        shift
#        cores_per_instance="$1"
#        echo "cores_per_instance is : $cores_per_instance"
#        ;;
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
         echo "Usage: ./inference/multi_instance.sh [OPTIONS]"
         echo "OPTION includes:"
         echo "   -l | --log_name - the log name of this round"
         echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
         echo "   -n | --num_of_ins_per_socket - number of instance per socket"
#         echo "   -c | --cores_per_instance - cores per instance"
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
        echo "Usage: inference/multi_instance.sh [OPTIONS]"
        echo "OPTION includes:"
        echo "   -l | --log_name - the log name of this round"
        echo "   -d | --dataset - [imdb|sst2] wether to use imdb or sst2 DATASET"
        echo "   -n | --num_of_ins_per_socket - number of instance per socket"
#        echo "   -c | --cores_per_instance - cores per instance"
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

if [ -z "$DATASET" ]; then
    echo "Error: Please enter the DATASET ot use [imdb|sst2]"
    exit
elif [ $DATASET != "imdb" -a $DATASET != "sst2" ]; then
    echo "Error: The DATASET $DATASET cannot be recognized, please enter 'imdb' or 'sst2'"
    exit
fi

if [ -z "$NUMBER_OF_INSTANCE_PER_SOCKET" ]; then
    echo "Error: Please set the instance number per socket using -n or --num_of_ins_per_socket"
    exit
fi

#if [ -z "$cores_per_instance" ]; then
#    echo "Please set the core number per instance using -c or --cores_per_instance"
#    exit
#fi

if [ $IPEX_BF16 = 1 ]; then
    if [ $INT8 = 1 -o $INT8_BF16 = 1 ]; then
        echo "Error: Cannot set IPEX_BF16 and INT8 at the same time"
        exit
    fi
else
    if [ $INT8 = 0 -a $INT8_BF16 = 1 ]; then
        echo "Error: Cannot set INT8_BF16 without INT8 option"
        exit
    fi
fi

if [ -z "$BATCH_SIZE" ]; then
    echo "Error: Please set the batch size per instance using -b or --BATCH_SIZE"
    exit
fi

if [ -z $SEQUENCE_LEN ]; then    
    if [ $DATASET = 'imdb' ]; then
        SEQUENCE_LEN=512
    elif [ $DATASET = 'sst2' ]; then
        SEQUENCE_LEN=55
    fi
    echo "WARNING: SEQUENCE_LEN is not set, using default DATASET ($DATASET) sequence length $SEQUENCE_LEN"
fi


all_core_number=`cat /proc/cpuinfo |grep "processor"|wc -l`
socket_number=`lscpu | grep "Socket(s)" | awk '{print $2}'`
core_number_per_socket=$(($all_core_number / $socket_number))
instance_number=$(($NUMBER_OF_INSTANCE_PER_SOCKET * $socket_number))

if [ $(($core_number_per_socket % $NUMBER_OF_INSTANCE_PER_SOCKET)) != 0 ]; then
    echo "\`instance_numberi_per_socket($NUMBER_OF_INSTANCE_PER_SOCKET)\` cannot be divisible by \`core_number_per_socket($core_number_per_socket)\`"
    exit
else
    cores_per_instance=$(($core_number_per_socket / $NUMBER_OF_INSTANCE_PER_SOCKET))
fi

if [ $DATASET = 'imdb' ]; then
    max_test_samples=$((25000/$instance_number))
else
    max_test_samples=$((872/$instance_number))
fi

OUTPUT_DIR=$OUTPUT_DIR'/'$pre'/'$DATASET
echo "log directory is $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR


for i in $(seq 1 $instance_number)
do
        export OMP_NUM_THREADS=$cores_per_instance
        start_index=$(( ($i-1) * $cores_per_instance))
        end_index=$(( ($i * $cores_per_instance) -1))
        mem_bind=$(( $start_index / $core_number_per_socket))
        echo "\`start core index\` is $start_index"
        echo "\`end core index \` is $end_index"
        echo "\`memory bind\` is $mem_bind"
        str="numactl -C $start_index-$end_index -m $mem_bind"
        echo $str
        nohup numactl -C $start_index-$end_index -m $mem_bind python ./src/run_pt_native_inf.py \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --dataset $DATASET \
                --int8 $INT8 \
                --int8_bf16 $INT8_BF16 \
                --ipex_bf16 $IPEX_BF16 \
                --ipex_fp32 $IPEX_FP32 \
                --multi_instance \
                --output_dir $OUTPUT_DIR/output_test \
                --do_predict \
                --max_seq_len $SEQUENCE_LEN \
                --instance_index $i \
                --max_test_samples $max_test_samples \
                --per_device_eval_batch_size $BATCH_SIZE \
                > $OUTPUT_DIR/test_$i.log 2>&1 &
done
