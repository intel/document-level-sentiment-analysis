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

# export CUDA_VISIBLE_DEVICES="-1"; \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-bert-large-uncased}"
DATASET="${DATASET:-sst2}"
MAX_SEQ_LEN=55
NUM_TRAIN_EPOCHS=1
OUTPUT_DIR="${OUTPUT_DIR:-fine_tuned}"
TRAINNING_BS=32
INFERENCE_BS=8
    #--bf16_ft \
python src/run_pt_native_ft.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset $DATASET \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_seq_len $MAX_SEQ_LEN \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --per_device_train_batch_size $TRAINNING_BS \
    --do_predict \
    --per_device_eval_batch_size $INFERENCE_BS \
    --logging_strategy epoch \
    $@
